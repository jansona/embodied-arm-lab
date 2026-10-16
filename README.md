# Embodied Arm Lab

一个用中文对话控制虚拟 SO-ARM101 的实验项目，运行在 CPU 上，用浏览器查看双目相机和任务状态。

目标链路：**用户输入 → Qwen 理解任务 → 相机图像识别场景 → Jev 选择小步动作 → SO101 物理仿真 → 新图像反馈**。

**当前状态：实验原型。** 已有仿真、图像感知、对话/计划预览和受限悬停执行；完整的 SO101 图像闭环抓取、放置、堆叠尚未通过真实模型端到端验收。Panda 的历史成功记录不代表 SO101 已完成。

本次整理已验证对话/任务状态服务和使用显式假模型的浏览器交互；生产抓取/放置仍禁用。两个独立阻塞尚未解决：

- 当前环境不允许访问模型服务，本轮没有真实 Qwen/Jev 验收
- 离线“真实渲染 RGB → 假 Jev → 物理仿真”整链路在下降时因蓝色障碍物单侧遮挡被视觉校验拒绝，尚未完成抓放

发布前完整离线回归：800 项测试及 121 项子测试通过，16 项跳过（10 项需显式启动的浏览器测试、6 项依赖未发布图像记录）。详细结果与此前出现的 IPC 偶发失败见 [本次检查点](PRODUCT-CHECKPOINT.md)。这次全量通过不代表该偶发问题已被定位修复。新的颜色细化修复排除了黄色夹爪对红色识别的干扰；旧报告描述旧版本，高分辨率/无阴影实验没有成为默认或已验证配置。

## 1. 安装与启动

推荐 Linux + Python 3.11；实时推理 worker 使用 Linux/Unix 进程接口。无需 GPU。

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run-so101.sh
```

打开 http://127.0.0.1:8766 。默认只启动仿真和网页，不启动模型 worker；如果同一账户已有 worker 在运行，网页会显示其真实状态。停止终端中的进程用 Ctrl+C。

部署到自己的 Linux 主机也是这三步。网页仅监听本机地址；远程访问请使用已有的安全隧道或受控桌面浏览器。此原型没有公网身份认证，不要直接公开端口。

## 2. 配置模型

将 `config.example.json` 复制到仓库外，例如 `../arm-provider.json`，由你在本机填入 `api_key`。真实配置不要提交到 Git、贴进聊天或放进网页。

模板固定使用阿里云百炼：

- Qwen 对话/图像理解：`qwen3.8-max`
- Jev 动作决策：`decision-model-preview`
- 服务地址：`https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/systemone`
- `min_confidence`、`timeout` 见模板；这不是经过校准的安全概率

系统验证目标地址和模型，不支持通过网页传入任意 API 地址。运行环境必须被允许访问服务；网络受限时修正环境配置，不绕过限制。

## 3. 使用真实模型

```sh
./run-so101.sh --live --config ../arm-provider.json
```

这一个命令启动网页、Qwen 对话、Qwen 视觉和 Jev 三个 worker。打开网页本身不会调用模型；主动提交对话、观察或动作请求可能产生费用。

每次启动只开放一个有上限的窗口：Qwen 对话 6 次/600 秒，视觉 6 次/600 秒，Jev 12 次/180 秒。时间从 worker 启动开始计算，空闲也会到期；不会自动续期。过期后网页仍可查看，重新启动命令才会开新窗口。

正常操作：

1. 查看相机画面与 worker 状态
2. 输入中文，测试对话；需要场景的任务先做视觉观察
3. 查看计划与错误提示，区分“计划生成”和“动作完成”
4. 需要时点击停止或重置；取消后的旧模型结果不会继续执行

需要受限的图像引导悬停实验时，显式增加 `--enable-waypoint`。悬停不是抓取。旧入口 `run-so101-hover.sh` 保留为兼容命令。

若已有受支持的“网页环境 + 推理环境”分离部署，在网页环境启动 `./run-so101.sh`，在可访问服务的推理环境执行：

```sh
./run-so101.sh --live --workers-only --config ../arm-provider.json
```

两个环境必须使用现有受支持的私有队列共享方式；这不是通用远程部署或网络代理。普通单机不需要这个选项。

## 4. 项目结构

- `app_so101.py`、`static_so101/`：SO101 网页与交互 API
- `brain.py`：Qwen 对话和结构化计划
- `camera_perception.py`、`vision_adapter.py`、`rgb_*.py`：RGB 感知与几何估计
- `so101_control.py`、`so101_executor.py`、`so101_jev.py`：场景、执行状态与 Jev 决策
- `so101_simulation.py`、`so101_robot.py`、`assets/so101/`：物理仿真与机器人模型
- `*_worker.py`、`*_ipc.py`：有预算的推理进程与私有消息队列
- `embodied_arm/`：任务生命周期、图像反馈和抓取/放置服务（正在接入）
- `start_so101.py`：统一启动与进程关闭
- `tests/`：离线回归与浏览器端到端测试
- `experiments/`：独立诊断/研究脚本，不参与正常启动
- `tools/`：源码打包等维护工具
- `app.py`、`static/`：保留的 Panda 基线，非主入口

历史诊断脚本已移到 `experiments/`，以 `python -m experiments.<模块名>` 调用；完整分层迁移见 [架构与迁移](docs/ARCHITECTURE.md)。[Panda 基线](docs/PANDA-BASELINE.md)、[SO101 实验说明](SO101-EXPERIMENT.md) 和 [历史验证范围](SO101-VALIDATION.md) 单独保留。

## 5. 测试与边界

```sh
.venv/bin/python -m pytest -q
```

默认测试使用离线/显式模拟响应，不需要 API 密钥。测试通过只说明软件检查通过，不能替代真实 Qwen → Jev → SO101 的端到端验收。

场景感知使用相机 RGB 图像；控制还可使用关节编码器、正向运动学和夹爪位置。仿真物体真值、深度、接触身份不进入 AI 感知/控制输入，独立评估数据与控制隔离。当前使用冻结观察窗口，不是动态实时视觉控制。

模型失败、低置信度、过期观察和取消都会停止或拒绝动作，不会悄悄切换为脚本控制。物体通过物理接触运动，不用传送或固定到夹爪的方式伪造抓取。本项目不是实机安全系统。
