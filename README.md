# Embodied Arm Lab

用中文对话控制虚拟 SO-ARM101 的 CPU 仿真实验。

目标：**输入任务 → Qwen 理解 → RGB 图像识别 → Jev 小步决策 → 机械臂动作 → 新图像验证**。

目前支持仿真预览、对话/计划和有预算的悬停实验。**生产抓取/放置仍禁用，真实 Qwen → Jev → SO101 抓放验收未通过。** 离线整链路仍会因下降过程中的障碍物遮挡被视觉校验拒绝；当前云环境还阻止模型网络访问。这两个问题独立存在，详见 [当前验收状态](docs/PRODUCT-CHECKPOINT.md)。

## 安装和运行

推荐 Linux + Python 3.11，无需 GPU。实时 worker 使用 Unix 进程接口。

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m embodied_arm
```

打开 http://127.0.0.1:8766 。默认只启动网页/仿真，不启动模型 worker；同一账户已有 worker 时会显示实际状态。Ctrl+C 关闭本次启动的进程。

安装后也可运行 `embodied-arm`。不需要安装项目时，在仓库根目录安装 `requirements.txt` 后运行 `python -m embodied_arm`。Shell 便利入口放在 `scripts/run-so101.sh`，不再在根目录散放脚本。

部署到 Linux 主机使用相同步骤。仅监听本机；远程访问使用已有的安全隧道或受控桌面浏览器。本项目没有公网认证，不要直接公开端口。

## 配置与真实模型

把 `config/provider-v2.example.json` 复制到仓库外，例如 `../arm-provider.json`。分别填写 `brain`（大脑）和 `cerebellum`（小脑）的 `base_url`、`model`、`api_key`；两组可以使用同一把 key，也可以独立配置。视觉默认使用大脑配置。不要把真实配置提交到 Git、输入网页或贴进聊天。

已有旧版单组配置仍可直接使用：加载时在内存中映射到两组，不重写文件、不要求重新输入 key。当前已验证的适配器仅支持下列百炼地址和模型，不表示任意 OpenAI-compatible 服务都能替换。

- Qwen 对话/视觉：`qwen3.8-max`
- Jev 决策：`decision-model-preview`
- 服务：阿里云百炼，模板填写基础地址；大脑追加 `/chat/completions`，小脑追加 `/systemone`，不会重复拼接路径
- 环境必须获准访问服务；网络被限制时修正环境权限，不绕过限制

```sh
.venv/bin/python -m embodied_arm --live --config ../arm-provider.json
```

这会启动网页和三个有预算的 worker。打开网页本身不调用模型；主动提交对话、观察或动作请求可能产生费用。

可分别检查配置或主动测试连接（没有后台连接探测）：

```sh
# 仅本地校验，不请求模型
.venv/bin/python -m embodied_arm.providers.diagnostics --config ../arm-provider.json --component brain
# 明确发起一次连接测试，可能计费；小脑将 brain 改成 cerebellum
.venv/bin/python -m embodied_arm.providers.diagnostics --config ../arm-provider.json --component brain --test-connection
```

检查结果仅显示组件、目标、状态和调用次数，不输出 key 或响应正文。网络策略禁止时仍会失败；该命令不会绕过环境权限。

每次窗口：对话 6 次/600 秒，视觉 6 次/600 秒，Jev 12 次/180 秒。空闲也计时，不自动续期。到期后网页保留预览，须显式重启才开新窗口。

使用步骤：启动实时会话 → 直接输入中文要求 → 自动获取当前视觉证据 → 查看 Qwen 计划和执行状态。无需先点击识别按钮。当前抓放执行仍未通过验收，因此计划会明确显示为“未执行”，不会把对话完成冒充动作完成。

双目画面默认约每秒本地采样一次，慢帧不重叠、不积压；可用 `--camera-interval 2` 修改间隔（0.25–10 秒）。这只更新本地图像，不按秒调用付费模型。涉及场景的对话会复用有效视觉证据，过期时先尝试本地图像跟踪，再按需发起一次视觉识别；服务不可用、超时或图像不可靠会显示具体原因。手动识别和悬停控件仅保留在诊断面板。停止会取消待处理请求，双目预览仍继续。

`--enable-waypoint` 显式开启受限悬停实验，不能开启抓放。`--workers-only` 用于已有、受支持的同机分离执行环境，只启动 worker；普通单机不需要。

## 目录

```text
embodied_arm/
  providers/       # Qwen/Jev 协议、响应校验
  perception/      # RGB、几何估计、视觉反馈和标记记忆
  control/         # SO101 小步动作、waypoint、实验抓放
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

```sh
.venv/bin/python -m pytest -q
```

默认测试不需要 API 密钥；浏览器测试需要额外安装 Playwright/Chromium，否则明确跳过。离线测试或假模型浏览器检查不能替代真实模型端到端验收。

场景只从 RGB 图像估计；控制可读关节编码器、运动学和夹爪位置。仿真物体真值/接触信息仅用于独立评估，不给 AI 或控制器补答案。失败、低置信度、过期图像和取消会停止/拒绝动作，不会偷偷换成脚本。不是实机安全系统。

源码运行的 IPC 位于仓库 `.runtime/`；安装后的包使用用户目录 `.embodied-arm-runtime/`，不会写入安装目录。真实配置始终留在源码/包之外。
