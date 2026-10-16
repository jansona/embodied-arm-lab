# 可选实验与诊断

这些脚本不参与 `./scripts/run-so101.sh` 的正常启动。从仓库根目录以 `python -m experiments.<模块名>` 运行，避免依赖脚本所在目录的 import 行为。

- `evaluate_so101_landmarks`：读取显式提供的诊断数据，计算离线指标
- `so101_waypoint_diagnostic`：读取显式提供的像素记录，以模拟 Jev 检查物理执行
- `diagnose_so101_resolution`：历史分辨率实验分析，需要另行提供原始实验文件
- `gripper_stability_diagnostic`：Panda 夹爪物理稳定性离线回归
- `direction_audit`：Jev 方向决策研究
- `api_smoke`、`local_target_audit`、`phased_micro_validation`、`physical_microcontrol`：可产生真实 API 请求和费用的旧诊断工具，不属于离线测试入口

部分历史工具依赖仓库外的显式配置和未发布的实验记录。不要为完成普通启动或离线测试运行这些工具；没有原始输入时不能将替代数据作为原实验结果。
