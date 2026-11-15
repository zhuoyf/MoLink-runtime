// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{mpsc, oneshot};
use tracing::trace;

use crate::EngineId;
use crate::client::stream::EngineCoreStreamOutput;
use crate::error::{Error, Result};
use crate::protocol::output::{EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput};
use crate::protocol::stats::SchedulerStats;
use crate::protocol::utility::UtilityOutput;
use crate::transport::ConnectedEngine;

pub type OutputSender = mpsc::UnboundedSender<Result<EngineCoreStreamOutput>>;
pub type OutputReceiver = mpsc::UnboundedReceiver<Result<EngineCoreStreamOutput>>;
pub type UtilitySender = oneshot::Sender<Result<UtilityOutput>>;
pub type UtilityReceiver = oneshot::Receiver<Result<UtilityOutput>>;

#[derive(Debug)]
struct TrackedRequest {
    sender: OutputSender,
    engine_id: EngineId,
    lora: Option<LoraRequestState>,
}

/// Frontend-side view of one LoRA request's scheduling phase.
///
/// The engine's `SchedulerStats` does not carry adapter names, so
/// `vllm:lora_requests_info` must be derived from per-request lifecycle events
/// observed by this client, mirroring `LoRARequestStates` in the Python
/// frontend (`vllm/v1/engine/output_processor.py`).
#[derive(Debug)]
struct LoraRequestState {
    adapter_name: String,
    phase: LoraPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoraPhase {
    Waiting,
    Running,
}

/// The latest real scheduler-side load snapshot observed from one engine.
///
/// These counters come from `scheduler_stats` on the normal engine output path
/// and are the preferred routing signal once available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EngineLoadSnapshot {
    /// Requests still counted on the scheduler's waiting side.
    waiting: usize,
    /// Requests currently counted on the scheduler's running side.
    running: usize,
}

#[derive(Debug, Default)]
struct EngineRoutingState {
    /// Requests admitted by this frontend that have not finished yet.
    ///
    /// This is used both as the bootstrap fallback before real scheduler stats
    /// exist and as a lower bound afterwards so asynchronous scheduler
    /// snapshots cannot erase frontend admission history.
    inflight: usize,
    /// The latest real scheduler snapshot received from this engine, if any.
    last_scheduler_stats: Option<EngineLoadSnapshot>,
}

impl EngineRoutingState {
    /// Compute the routing score used to pick the least-loaded engine.
    ///
    /// Scheduler stats can raise the load estimate above the frontend-local
    /// view, but they should not lower it below requests this frontend has
    /// already admitted.
    fn routing_score(&self) -> usize {
        let Some(stats) = self.last_scheduler_stats else {
            return self.inflight;
        };

        self.inflight.max(stats.running + stats.waiting)
    }

    /// Replace the local routing view with a fresh real scheduler snapshot.
    fn apply_scheduler_counts(&mut self, next: EngineLoadSnapshot) {
        self.last_scheduler_stats = Some(next);
    }
}

/// Internal registry for tracking active requests and their output stream
/// senders.
///
/// This is used to route incoming outputs to the correct request stream, and to
/// ensure proper cleanup of senders when requests finish or the client shuts
/// down.
#[derive(Debug)]
pub struct RequestRegistry {
    closed: bool,
    requests: HashMap<String, TrackedRequest>,
    active_lora_requests: usize,
    routing_per_engine: BTreeMap<EngineId, EngineRoutingState>,
}

impl RequestRegistry {
    pub fn new(engines: &[ConnectedEngine]) -> Self {
        Self {
            closed: false,
            requests: HashMap::default(),
            active_lora_requests: 0,
            routing_per_engine: engines
                .iter()
                .map(|engine| (engine.engine_id.clone(), EngineRoutingState::default()))
                .collect(),
        }
    }

    /// Register a newly added request. Create the per-request output channel
    /// bound to its `request_id` and return the selected engine id.
    ///
    /// When `data_parallel_rank` is provided, the request is routed directly to
    /// the engine at that rank index, bypassing load balancing. Otherwise
    /// the engine with the fewest in-flight requests is chosen.
    pub fn register(
        &mut self,
        request_id: String,
        lora_name: Option<String>,
        data_parallel_rank: Option<u32>,
    ) -> Result<(EngineId, OutputReceiver)> {
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }

        let engine_id = self.choose_engine_for_request(data_parallel_rank)?;
        let (tx, rx) = mpsc::unbounded_channel();
        let lora = lora_name.map(|adapter_name| LoraRequestState {
            adapter_name,
            phase: LoraPhase::Waiting,
        });
        if lora.is_some() {
            self.active_lora_requests += 1;
        }
        self.requests.insert(
            request_id,
            TrackedRequest {
                sender: tx,
                engine_id: engine_id.clone(),
                lora,
            },
        );

        let state = self
            .routing_per_engine
            .get_mut(&engine_id)
            .expect("request registry must track all known engines");
        state.inflight += 1;

        Ok((engine_id, rx))
    }

    fn choose_engine_for_request(&mut self, data_parallel_rank: Option<u32>) -> Result<EngineId> {
        if let Some(rank) = data_parallel_rank {
            let engine_id = u16::try_from(rank).ok().map(EngineId::from_engine_index);
            return engine_id
                .filter(|engine_id| self.routing_per_engine.contains_key(engine_id))
                .ok_or_else(|| Error::InvalidDataParallelRank {
                    rank,
                    connected_ranks: self
                        .routing_per_engine
                        .keys()
                        .filter_map(EngineId::engine_index)
                        .collect(),
                });
        }

        Ok(self
            .routing_per_engine
            .iter()
            .min_by_key(|(_, state)| state.routing_score())
            .map(|(engine_id, _)| engine_id.clone())
            .expect("request registry must contain at least one engine"))
    }

    /// Filter the given request IDs to the subset that are still tracked as
    /// active and can be aborted, grouped by engine.
    pub fn abortable_request_ids(&self, request_ids: &[String]) -> BTreeMap<EngineId, Vec<String>> {
        let mut by_engine = BTreeMap::new();
        for request_id in request_ids {
            let Some(tracked) = self.requests.get(request_id.as_str()) else {
                continue;
            };
            by_engine
                .entry(tracked.engine_id.clone())
                .or_insert_with(Vec::new)
                .push(request_id.clone());
        }
        by_engine
    }

    /// Obtain the stream sender for one output. If it indicates the request is
    /// finished, it will be removed from the registry.
    pub fn sender_for_output(&mut self, output: &EngineCoreOutput) -> Option<OutputSender> {
        self.apply_lora_events(output);
        if output.finished() {
            self.remove(output.request_id.as_str()).map(|tracked| tracked.0)
        } else {
            self.requests
                .get(output.request_id.as_str())
                .map(|tracked| tracked.sender.clone())
        }
    }

    /// Advance the request's LoRA scheduling phase from the engine-core events
    /// attached to one output, mirroring the Python frontend's
    /// `LoRARequestStates.update_from_events`.
    fn apply_lora_events(&mut self, output: &EngineCoreOutput) {
        let Some(events) = output.events.as_ref() else {
            return;
        };
        let Some(lora) = self
            .requests
            .get_mut(output.request_id.as_str())
            .and_then(|tracked| tracked.lora.as_mut())
        else {
            return;
        };
        for event in events {
            lora.phase = match event.r#type {
                EngineCoreEventType::Queued | EngineCoreEventType::Preempted => LoraPhase::Waiting,
                EngineCoreEventType::Scheduled => LoraPhase::Running,
            };
        }
    }

    /// Snapshot the adapter names of tracked LoRA requests as
    /// (running, waiting) sets. Feeds the `vllm:lora_requests_info` gauge.
    pub fn lora_adapter_states(&self) -> (BTreeSet<String>, BTreeSet<String>) {
        if self.active_lora_requests == 0 {
            return (BTreeSet::new(), BTreeSet::new());
        }

        let mut running = BTreeSet::new();
        let mut waiting = BTreeSet::new();
        for lora in self.requests.values().filter_map(|tracked| tracked.lora.as_ref()) {
            let set = match lora.phase {
                LoraPhase::Running => &mut running,
                LoraPhase::Waiting => &mut waiting,
            };
            set.insert(lora.adapter_name.clone());
        }
        (running, waiting)
    }

    /// Obtain stream senders for a whole engine output batch under one
    /// registry lock. Finished outputs are removed before returning.
    pub fn senders_for_outputs<'a>(
        &mut self,
        outputs: impl IntoIterator<Item = &'a EngineCoreOutput>,
    ) -> Vec<Option<OutputSender>> {
        outputs.into_iter().map(|output| self.sender_for_output(output)).collect()
    }

    /// Remove a batch of requests that have finished or aborted, returning
    /// their stream senders.
    pub fn finish_many<'a>(
        &mut self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<OutputSender> {
        request_ids
            .into_iter()
            .filter_map(|request_id| self.remove(request_id.as_str()).map(|tracked| tracked.0))
            .collect()
    }

    /// Apply one scheduler stats update for the given engine to the local
    /// routing state. Returns `false` if the engine is unknown to the
    /// client.
    pub fn apply_scheduler_stats(&mut self, engine_index: u32, stats: &SchedulerStats) -> bool {
        self.apply_scheduler_counts(
            engine_index,
            EngineLoadSnapshot {
                waiting: stats.num_waiting_reqs as usize,
                running: stats.num_running_reqs as usize,
            },
        )
    }

    /// Mark the registry as closed, detach and return all tracked senders.
    pub fn close(&mut self) -> Vec<OutputSender> {
        if self.closed {
            return Vec::new();
        }

        self.closed = true;
        self.active_lora_requests = 0;
        std::mem::take(&mut self.requests)
            .into_values()
            .map(|tracked| tracked.sender)
            .collect()
    }

    /// Finalize client-initiated aborts: remove each request and push a
    /// terminal output with `finish_reason = Abort` down its stream before the
    /// sender drops. Returns the request ids that were still active.
    pub fn abort_many<'a>(
        &mut self,
        request_ids: impl IntoIterator<Item = &'a String>,
        timestamp: f64,
    ) -> Vec<String> {
        let mut aborted = Vec::new();
        for request_id in request_ids {
            let Some((sender, engine_id)) = self.remove(request_id) else {
                continue;
            };
            let output = EngineCoreStreamOutput {
                engine_index: engine_id.engine_index().unwrap_or(0),
                timestamp,
                output: EngineCoreOutput {
                    request_id: request_id.clone(),
                    finish_reason: Some(EngineCoreFinishReason::Abort),
                    ..EngineCoreOutput::default()
                },
            };
            let _ = sender.send(Ok(output));
            aborted.push(request_id.clone());
        }
        aborted
    }

    /// Remove one request from the local registry. Returns the tracked entry if
    /// it exists.
    #[must_use]
    pub fn remove(&mut self, request_id: &str) -> Option<(OutputSender, EngineId)> {
        let tracked = self.requests.remove(request_id)?;
        if tracked.lora.is_some() {
            self.active_lora_requests -= 1;
        }
        self.routing_per_engine
            .get_mut(&tracked.engine_id)
            .expect("request registry must track all known engines")
            .inflight -= 1;
        Some((tracked.sender, tracked.engine_id))
    }

    fn apply_scheduler_counts(&mut self, engine_index: u32, next: EngineLoadSnapshot) -> bool {
        let Ok(engine_index) = u16::try_from(engine_index) else {
            return false;
        };
        let engine_id = EngineId::from_engine_index(engine_index);
        let Some(state) = self.routing_per_engine.get_mut(&engine_id) else {
            return false;
        };

        let previous = state.last_scheduler_stats;
        if previous != Some(next) {
            trace!(
                ?engine_id,
                previous_waiting = previous.map(|stats| stats.waiting),
                previous_running = previous.map(|stats| stats.running),
                waiting = next.waiting,
                running = next.running,
                "updated scheduler routing counts",
            );
        }

        state.apply_scheduler_counts(next);
        true
    }


    pub fn is_closed(&self) -> bool {
        self.closed
    }

}

/// Internal registry for tracking active utility calls and their waiting
/// receivers.
#[derive(Debug)]
pub struct UtilityRegistry {
    closed: bool,
    next_call_id: AtomicU64,
    utility_calls: BTreeMap<u64, UtilitySender>,
}

impl Default for UtilityRegistry {
    fn default() -> Self {
        Self {
            closed: false,
            next_call_id: AtomicU64::new(1),
            utility_calls: BTreeMap::default(),
        }
    }
}

impl UtilityRegistry {
    /// Allocate the next utility `call_id` and register a newly added utility
    /// call.
    pub fn allocate_and_register(&mut self) -> (u64, UtilityReceiver) {
        let call_id = self.next_call_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.utility_calls.insert(call_id, tx);
        (call_id, rx)
    }

    /// Resolve a utility output to its waiting receiver.
    pub fn resolve(&mut self, call_id: &u64) -> Option<UtilitySender> {
        self.utility_calls.remove(call_id)
    }

    /// Drop a batch of registered utility calls without delivering a result.
    /// Used to roll back allocations when the dispatch fan-out fails before
    /// every engine could accept the request.
    pub fn unregister_many(&mut self, call_ids: impl IntoIterator<Item = u64>) {
        for call_id in call_ids {
            self.utility_calls.remove(&call_id);
        }
    }

    /// Mark the registry as closed, detach and return all tracked senders.
    pub fn close(&mut self) -> Vec<UtilitySender> {
        if self.closed {
            return Vec::new();
        }

        self.closed = true;
        std::mem::take(&mut self.utility_calls).into_values().collect()
    }



    pub fn is_closed(&self) -> bool {
        self.closed
    }
}
