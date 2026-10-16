# Embodied Arm Lab

用中文要求控制虚拟 SO-ARM101 的 CPU 仿真实验。

当前主流程：**完整相机图像 → Qwen 直接从 33 个动作中选择 → 机械约束校验 → 执行一个有界小步 → 新图像再次交给 Qwen**。Qwen 负责判断是否完成；这属于模型判断，不是独立物理验收。接近、抓取等概念可作为参考，不强制按固定配方执行。

默认入口使用 `embodied_arm.web.app_visual` 和一个 `visual_loop_worker`。旧 CV 跟踪/抓放路径保留为历史实验，不参与主入口。默认 `qwen_direct` 不加载或调用小脑；`two_model` 保留 Qwen + Jev 对照。当前路线尚未完成真实模型抓放验收，只用于模拟器，不支持物理硬件。

## 安装和运行

推荐 Linux + Python 3.11，无需 GPU；worker 使用 Unix 进程接口。

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m embodied_arm
```

打开 http://127.0.0.1:8766 。默认只预览，不启动模型 worker、不调用模型。安装后也可运行 `embodied-arm`。Ctrl+C 关闭本次启动的进程。

部署使用相同步骤，只监听本机。远程访问使用已有安全隧道或受控桌面浏览器；本项目没有公网认证，不要直接公开端口。

## 配置和执行

把 `config/provider.example.json` 复制到仓库外，例如 `../arm-provider.json`。默认只需填写 `brain`（Qwen）的 `base_url`、`model`、`api_key`；`cerebellum` 可省略，或保留模板中的空密钥，直接模式不加载或调用它。对照模式才需要填写 `cerebellum`（Jev）。已有外部 v2 配置不需要换格式；旧单组配置仍兼容，参考 `config/provider-legacy.example.json`。不要提交真实配置、在网页填写或贴进聊天。

直接模式也支持最小配置（保存到仓库外，再自行填写密钥）：

```json
{
  "schema_version": 2,
  "brain": {
    "provider": "aliyun_bailian",
    "base_url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    "model": "qwen3.8-max",
    "api_key": "",
    "timeout": 30
  }
}
```

- 大脑：百炼 `qwen3.8-max`，接收完整图像
- 可选对照小脑：`decision-model-preview`，从有限动作中选择
- 模板使用已支持的百炼基础地址；不代表任意兼容服务都经过验证

```sh
.venv/bin/python -m embodied_arm --live --config ../arm-provider.json
```

默认命令等价于添加 `--mode qwen_direct`。需要 Jev 对照时显式运行：

```sh
.venv/bin/python -m embodied_arm --live --mode two_model --config ../arm-provider.json
```

输入中文要求并提交执行。每个动作后重新观察，拒绝、超时、停止和预算到期会结束本次执行，不会回退到旧 CV 或脚本。主动提交会调用模型并可能计费；空闲不自动请求、不自动续期。

会话最多 Qwen 201 次、600 秒；直接模式 Jev 调用为 0，对照模式最多 200 次。Qwen 多出的一次用于最终观察。单任务最多 200 次动作决策、420 秒，同时受 worker 剩余会话预算限制。会话从 worker 启动计时，等待也占用时间。最终完成还需 Qwen 新图像判断；预算耗尽不能当成完成。

`--workers-only` 只启动 worker，不能开启已运行的预览应用；普通使用直接运行上述单机 `--live`。旧 waypoint 选项已移除。`--cycle-max-calls 32`（1–200）设置显式进度复核间隔，不额外调用模型；每个动作仍重新询问 Qwen。`--camera-interval 1`（0.25–10 秒）只控制本地预览采样，不产生定时模型请求。

可只检查外部配置，不请求模型：

```sh
.venv/bin/python -m embodied_arm.providers.diagnostics --config ../arm-provider.json --component brain
```

当前验证状态见 [主路线检查点](docs/VISUAL-LOOP-CHECKPOINT.md)。旧实验结果仅用于历史参考，不能证明新路线可靠。

## 目录

```text
embodied_arm/
  providers/       # Qwen/Jev 协议、响应校验
  perception/      # 历史 CV 实验，不参与当前主路线
  control/         # 小步执行；旧 waypoint/抓放控制为历史实验
  simulation/      # 机器人模型、物理仿真
  services/        # 任务契约、对话/执行生命周期
  runtime/         # 有预算 worker、IPC、启动/关闭
  web/             # HTTP 接口与页面资源
  assets/          # SO101 URDF、网格及许可证
  legacy/panda/    # 保留的 Panda 基线执行工具
config/            # 空密钥配置模板
scripts/           # Shell 便利入口
experiments/       # 独立研究/诊断，不参与普通启动
tools/            # 源码打包维护工具
tests/            # 离线测试、显式假模型的浏览器测试
docs/             # 架构、验收、历史记录
```

模块全部使用包导入，没有根目录兼容占位文件或 `sys.path` 修改。[本次结构重构验证](docs/PACKAGE-REFACTOR.md)记录完整测试和安装启动检查。详见 [架构](docs/ARCHITECTURE.md) 和 [入口迁移表](docs/MODULE-MIGRATION.md)。

## 测试与边界

当前路线优先运行：

```sh
.venv/bin/python -m pytest -q tests/test_visual_providers.py tests/test_visual_loop_ipc.py tests/test_app_visual.py tests/test_visual_ui.py tests/e2e_conversation/test_generic_visual_loop.py tests/test_visual_mechanical_limits.py tests/test_start_so101.py
```

需要维护历史代码时，可选运行完整旧/新测试集合：`python -m pytest -q`。不需要重跑旧 CV 物理实验才能使用新入口。

默认测试不需要 API 密钥；浏览器测试需要额外安装 Playwright/Chromium，否则明确跳过。离线测试或假模型浏览器检查不能替代真实模型端到端验收。

场景只从 RGB 图像估计；控制可读关节编码器、运动学和夹爪位置。仿真物体真值/接触信息仅用于独立评估，不给 AI 或控制器补答案。失败、低置信度、过期图像和取消会停止/拒绝动作，不会偷偷换成脚本。不是实机安全系统。

源码运行的 IPC 位于仓库 `.runtime/`；安装后的包使用用户目录 `.embodied-arm-runtime/`，不会写入安装目录。真实配置始终留在源码/包之外。

默认使用完整的 observer+wrist 两路 640×480 图像，不需要本地 CV 高分辨率处理。默认由 Qwen 一次请求选择方向和距离，对照模式交由 Jev 选择：XYZ 六个正负方向 × 0.5、1、2、4、8 mm 共 30 个平移动作，加上夹爪开/合各 10 个百分点与保持，共 33 个选项。Python 只校验所选动作及边界；0.5 mm 是指令粒度，不是视觉定位精度。笛卡尔执行器保持工具朝下，不选择手腕朝向。新入口由 `SO101_VISUAL_LIVE` 控制（启动器 `--live` 设置为 1，预览为 0）。

默认模拟场景仍包含原来的积木。新决策合同不绑定固定物体 ID 或物体几何，也不使用旧语义视觉提取器，但这不能证明任意场景可靠；完成仍是 Qwen 的图像判断。最终独立 IPC/物理冒烟验证待完成，不引用旧 CV 回归作为新路线验收。
