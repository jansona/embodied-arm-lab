# 架构与职责

## 主链路

Web/API → TaskService → Qwen planner → RGB SceneProvider → JevMotionDriver / executor → SO101Simulation。

- `services/` 管理任务 ID、历史、状态事件、预算、取消/重置和能力门控；不依赖 HTTP
- `providers/` 管理固定模型协议、响应校验和连接；密钥只在显式服务器配置中加载
- `perception/` 从 RGB 生成带时间/来源/不确定度的观察；运动使旧观察失效
- `control/` 基于验证过的目标与观察执行有边界的动作；每个方向/夹爪决策须来自声明的 controller
- `simulation/` 提供编码器、运动学、相机、物理步进和停止；oracle 只供独立测试评估
- `runtime/` 管理私有固定协议 IPC、有预算 worker、启动和结束，不自动续期
- `web/` 管理 HTTP 校验和界面，静态资源随包安装
- `legacy/panda/` 保留旧 Panda 执行/恢复实验；它们不是 SO101 的成功证据

现有部分共享配置加载仍复用 Panda runner 的经过校验逻辑，这是已知耦合；本次结构重构没有顺便修改协议或安全边界。

## 资源与入口

- `python -m embodied_arm` 和 `embodied-arm` 是同一个 SO101 入口
- worker 以 `python -m embodied_arm.runtime.<worker>` 启动，不使用文件路径导入
- `paths.py` 集中定位包内资源、源码根目录和 IPC；所有队列共用一个 runtime 根
- 源码配置排除范围仍为整个仓库，不能因为模块移动而只排除一个子目录
- 安装模式的运行数据放在用户目录，不写 site-packages
- `pyproject.toml` 声明依赖、console script、包发现和 URDF/网格/网页资源
- 测试使用真实包导入，测试间 helper 也使用 `tests.*`，没有修改 `sys.path`

## 当前能力边界

结构整理不等于图像控制可靠性提高。抓取/放置继续能力门控，真实模型验收未通过，详见 [当前检查点](PRODUCT-CHECKPOINT.md)。历史实验和数值结果留在独立文档/本地 fixtures；发布源码不带运行时、截图、真实配置或录制的实验 JSON。
