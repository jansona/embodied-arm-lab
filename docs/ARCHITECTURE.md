# SO101 产品结构与迁移

## 面向用户的主流程

一个页面、一条任务输入、一条可追踪执行链。用户不需要理解进程、JSON 或诊断脚本。界面必须明确区分对话、理解、观察、计划、执行、完成、失败和取消；“有回复”不等于“机器人完成任务”。

当前代码仍在迁移，不把下面的目标结构描述成已经完成的功能。

## 最小职责划分

1. **ConversationService**：保留对话上下文，调用 Qwen，输出经过验证的 intent/plan；没有新鲜图像时澄清，不猜测物体坐标
2. **TaskOrchestrator**：任务 ID、阶段、预算、状态事件、停止/重置代际；编排观察、计划和动作；成功必须由明确结果检查得出
3. **SceneProvider**：相机 RGB → 识别/局部图像细化 → 标定几何估计；携带图像时间、来源、不确定度，运动后旧观察失效
4. **JevController**：根据已验证目标和新鲜观察选择有边界的小步动作；拒绝无效/低置信度输出，不由程序伪装模型答案
5. **RobotBackend**：编码器、运动学、限位、物理步进、停止、相机捕获；仿真真值只通过独立评估接口暴露
6. **Web/API**：请求校验、呈现状态和事件；不承担运动阶段分解，不存放密钥

建议依赖方向：Web → ConversationService/TaskOrchestrator → SceneProvider/JevController → RobotBackend。评估器单向读取仿真，不反向为控制提供答案。

## 已完成的整理

- 主入口统一为 `run-so101.sh` / `start_so101.py`
- 根目录测试统一移入 `tests/`，更新静态资源路径
- 9 个独立研究脚本移入 `experiments/`，以模块方式运行
- 源码打包移入 `tools/package_so101.py`，包含新包、文档、实验工具
- Panda 历史 README 独立保存，主 README 专注 SO101

生产后端模块暂保留兼容路径，避免与进行中的执行服务接线产生冲突。

## 目标目录（后续迁移提案）

```text
embodied_arm/
  contracts.py          # 结构化 intent、plan、task、事件
  task_service.py       # 生命周期与编排
  conversation/         # Qwen planner、上下文与校验
  perception/           # RGB、视觉适配器、标定/不确定度
  control/              # Jev、waypoint/pick/place executor
  simulation/           # SO101 后端与独立 evaluator
  runtime/              # bounded workers、IPC、launcher
  web/                  # app、static
assets/so101/
tests/
experiments/             # 分辨率、抓取稳定性、微控制等研究脚本
legacy/panda/            # Panda 基线及其专有工具
docs/
```

不要为了目录整洁一次性移动所有模块。当前 root imports、静态资源路径、worker 脚本命令和资产路径依赖根目录布局；未经测试的搬家会破坏可运行性。

## 小步迁移顺序

1. 先统一 SO101 启动入口和 README，保留旧入口兼容；默认离线预览，真实推理显式开启
2. 增加 `embodied_arm` 的类型契约/任务服务，在现有接口上接通单个 pick/place 的完整生命周期；完整实现前明确拒绝不支持的操作
3. 把 UI 中任务编排移到服务层；stop/reset 取消传播到观察、规划、Jev 和仿真，旧回复不能改变新任务
4. 按职责逐组迁移模块，每组同时改 imports、测试、资产定位和 launcher，旧脚本只保留短兼容包装
5. 最后把诊断脚本移到 experiments，Panda 专有模块移到 legacy；共享物理/适配工具先拆出，不重复复制

## 单一启动与部署契约

主入口 `./run-so101.sh`；默认只启动 loopback 网页/仿真。`--live --config <外部配置>` 在同一受支持主机启动三个固定预算 worker。`--workers-only` 只服务已有的受支持分离部署，不引入代理、隧道或新跨环境授权。

不要将容器化或 pip 安装包装成已完成：当前有根目录模块和相对资源，尚未形成可搬移 wheel。先让干净检出 + Python 3.11 + requirements + 单一启动通过，再增加可安装包。部署还需认证、权限/会话隔离和服务生命周期设计；当前仅用于本机或受控远程桌面。

## 端到端验收标准

- 真实模型：普通对话、指代追问、未知物体澄清，以及输入任务后自动观察/规划/动作
- 真实模拟物理：先单块抓起并保持，再放到目标；后续才是双块堆叠
- 图像边界：遮挡、陈旧图像、低置信度、错误识别必须可见地拒绝；不得读 oracle 纠正动作
- HCI：重复提交、停止、重置、刷新/重连、模型失败、worker 到期均保持可理解的一致状态
- 证据：记录版本、模型、预算/实际调用、图像时间、动作结果与失败原因；不记录密钥；独立真值只用于验收

未通过真实模型的行为只能标记为离线测试或待验证，不能以 Panda 历史结果替代 SO101 的证据。
