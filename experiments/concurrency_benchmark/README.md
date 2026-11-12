# 并发扫描

从仓库根目录运行。复用 heterogeneous_benchmark 的 Control、KVThreshold、阶段参数及 adaptive_pp 的流式请求和调度器，不修改原始实验数据或 PipeLive。

正式矩阵：并发 2/4/8/12/16/24/32/48/64 × Full/Threshold/Static 4,8,12,16,20,24,28 × 3 次；每组重新启动服务，200 秒/阶段，独立模型副本和缓存。各轮轮转策略顺序。纯测量窗口共 40.5 小时，另加服务启动、预热和 Drain。

输入分布沿用原实验：32 输入、96 输出（确定性退化分布），固定 seed=42。最大运行请求数64，保留128 KV块上限、批Token上限128和512上下文长度。SLO TTFT≤5000ms、逐请求平均TPOT≤100ms。所有条件对策略一致。

窗口 [start,start+600) 内收到的输出Token计入吞吐；Goodput按完整请求SLO分类后只计窗口内Token。跨边界请求在Drain阶段完成，不补充新请求，其尾部Token不计入窗口。TTFT及TPOT分位数包含窗口内发起的全部请求。时间戳为UTC epoch；调度窗口使用单调时钟。phases.jsonl记录计划、调用与应用完成时间，包含接口和迁移互斥引起的延迟，不隐藏切换成本。

samples.jsonl每秒保存服务端Running/Waiting、客户端在途数、GPU利用率、KV占用和分层。调度决策独立采样，不阻塞监控。load.jsonl记录每次请求补充；requests.jsonl包含每个Token与接收时间。请求抢占不作为失败；Token错误、接口失败和迁移失败会停止实验并保留日志。

run_all.sh要求pilot/PASSED以及冻结校验通过；独占顺序运行，不覆盖results。每组server日志、资源和迁移日志保存在同级 *-server/。退出时生成exit_code及completed；成功时自动生成benchmark_report.md、aggregate.json和曲线。异常退出尽力生成部分报告。
