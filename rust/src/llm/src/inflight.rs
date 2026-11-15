// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Tracking of the external→internal request-id mapping for in-flight requests.
//!
//! When request-id randomization is enabled (the default), [`crate::Llm`]
//! rewrites the external (user-supplied) request id into a unique internal
//! engine id before reaching engine-core. Engine-core only ever knows the
//! internal id, so aborting a request by its external id requires resolving it
//! back to the internal id(s) first.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

/// external id → internal id → number of live guards holding that edge.
type InflightMap = HashMap<String, HashMap<String, usize>>;

/// Maps external (user-supplied) request ids to the set of live internal engine
/// request ids they currently expand into.
///
/// One external id may map to multiple internal ids: duplicate external ids
/// submitted concurrently each get their own randomized internal id, and an
/// abort by the shared external id must reach all of them. Edges are
/// refcounted: with randomization disabled the same (external, internal) pair
/// can be tracked by several guards in sequence (e.g. a finished request whose
/// stream is still held alongside a fresh submission reusing the id), and the
/// edge must survive until the last guard drops.
#[derive(Default)]
pub(crate) struct InflightRequests {
    map: Arc<Mutex<InflightMap>>,
}

impl InflightRequests {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record that `internal` is now an in-flight engine request for the
    /// `external` request id, returning a guard that removes the edge when the
    /// request's output stream is dropped (on clean finish or cancellation).
    pub(crate) fn track(&self, external: String, internal: String) -> RequestGuard {
        *self
            .map
            .lock()
            .entry(external.clone())
            .or_default()
            .entry(internal.clone())
            .or_insert(0) += 1;
        RequestGuard {
            map: Arc::downgrade(&self.map),
            external,
            internal,
        }
    }

    /// Resolve external request ids to the internal engine ids currently
    /// in-flight for them. Unknown or already-finished ids contribute nothing.
    pub(crate) fn resolve(&self, external_ids: &[String]) -> Vec<String> {
        let map = self.map.lock();
        external_ids
            .iter()
            .filter_map(|external| map.get(external))
            .flat_map(|internal_ids| internal_ids.keys())
            .cloned()
            .collect()
    }

    /// Collect the internal engine ids of every in-flight request. Used to
    /// abort all outstanding requests when no external ids are given.
    pub(crate) fn all_internal_ids(&self) -> Vec<String> {
        let map = self.map.lock();
        map.values().flat_map(|internal_ids| internal_ids.keys()).cloned().collect()
    }

}

/// RAII guard that releases one refcount on a single external→internal edge
/// when dropped, removing the edge once no live guard holds it.
///
/// Held by the per-request output stream, so cleanup runs whether the stream
/// terminates cleanly or is cancelled. A [`Weak`] handle is used so a stream
/// outliving its owning [`InflightRequests`] does not keep the map alive.
pub(crate) struct RequestGuard {
    map: Weak<Mutex<InflightMap>>,
    external: String,
    internal: String,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        let Some(map) = self.map.upgrade() else {
            return;
        };
        let mut map = map.lock();
        if let Some(internal_ids) = map.get_mut(&self.external) {
            if let Some(count) = internal_ids.get_mut(&self.internal) {
                *count -= 1;
                if *count == 0 {
                    internal_ids.remove(&self.internal);
                }
            }
            if internal_ids.is_empty() {
                map.remove(&self.external);
            }
        }
    }
}
