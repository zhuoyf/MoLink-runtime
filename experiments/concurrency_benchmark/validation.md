# 启动前验证

正式实验尚未启动；未生成 pilot/PASSED，run_all.sh 的启动门禁保持关闭。

已通过：Python语法、Ruff、Shell语法、固定窗口/Goodput边界自检、报告与五张曲线的临时数据渲染检查。

GPU短测 full_c4：每阶段40秒，共120秒。40个完整请求，Token mismatch=0；自动迁移2次，A/B/C均按时发起。最大请求补充间隔0.109ms，测量窗口120秒，Drain为4.592秒。完整证据位于pilot/full_c4及pilot/full_c4-server。

GPU短测 threshold_c64：保持64个在途请求，A→B阶段切换遭遇HTTP409：KV shrink would release active request blocks。最后一次监控与失败原因保存在pilot/high_concurrency_failure.json；完整异常在pilot-high.log，原始监控与请求保存在pilot/threshold_c64。

原因：现有molink_simulate先调用molink_resize_kv；plan_resize拒绝删除活跃KV块。闭环高并发使128块池接近满载，无法立即收缩到B阶段要求的容量。没有跳过失败、降低并发、修改模拟参数或改动PipeLive核心。

下一步需要明确允许范围：支持活跃请求抢占的缩容事务，或允许阶段切换时等待活跃请求排空并将延迟计入结果。第二种方式会改变阶段实际生效时刻，不能视为原协议已经通过。
