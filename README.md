# Embodied Arm Lab

用中文对话控制虚拟 SO-ARM101 的 CPU 仿真实验。

目标：**输入任务 → Qwen 理解 → RGB 图像识别 → Jev 小步决策 → 机械臂动作 → 新图像验证**。

目前支持仿真预览、对话/计划、悬停和显式开启的实验性仿真抓放任务。**`--live` 可提交模拟器任务，但真实 Qwen → Jev → SO101 抓放验收仍未通过；不支持物理硬件控制。** 离线整链路仍会因下降过程中的障碍物遮挡被视觉校验拒绝；当前云环境还阻止模型网络访问。这两个问题独立存在，详见 [当前验收状态](docs/PRODUCT-CHECKPOINT.md)。

动态微控制已增加逐步观察、目标纠正和有界修订。局部周期暂停后重新观察、由大脑复核，再从未完成动作继续，不重放已完成抓取或释放。堆叠必须有可见支撑证据；底块被遮住时不能宣告成功。下一阶段运行和验证见 [本机验证计划](docs/NEXT-PHASE-VALIDATION.md)，历史验收边界见 [动态闭环检查点](docs/DYNAMIC-LOOP-CHECKPOINT.md)。

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

把 `config/provider.example.json` 复制到仓库外，例如 `../arm-provider.json`。分别填写 `brain`（大脑）和 `cerebellum`（小脑）的 `base_url`、`model`、`api_key`；两组可以使用同一把 key，也可以独立配置。视觉默认使用大脑配置。不要把真实配置提交到 Git、输入网页或贴进聊天。

`config/provider-legacy.example.json` 仅供旧版单组格式参考。已有旧版单组配置仍可直接使用：加载时在内存中映射到两组，不重写文件、不要求重新输入 key。当前已验证的适配器仅支持下列百炼地址和模型，不表示任意 OpenAI-compatible 服务都能替换。

- Qwen 对话/视觉：`qwen3.8-max`
- Jev 决策：`decision-model-preview`
- 服务：阿里云百炼，模板填写基础地址；大脑追加 `/chat/completions`，小脑追加 `/systemone`，不会重复拼接路径
- 环境必须获准访问服务；网络被限制时修正环境权限，不绕过限制

```sh
.venv/bin/python -m embodied_arm --live --config ../arm-provider.json
```

这会启动网页、三个有预算的 worker，并显式开启实验性 SO101 仿真任务。打开网页本身不调用模型；主动提交对话、观察或动作请求可能产生费用。运行时的遮挡、置信度、预算和取消保护可能停止任务；开启任务不代表已通过可靠性验收。

可分别检查配置或主动测试连接（没有后台连接探测）：

```sh
# 仅本地校验，不请求模型
.venv/bin/python -m embodied_arm.providers.diagnostics --config ../arm-provider.json --component brain
# 明确发起一次连接测试，可能计费；小脑将 brain 改成 cerebellum
.venv/bin/python -m embodied_arm.providers.diagnostics --config ../arm-provider.json --component brain --test-connection
```

检查结果仅显示组件、目标、状态和调用次数，不输出 key 或响应正文。网络策略禁止时仍会失败；该命令不会绕过环境权限。

每次会话窗口：对话 10 次/600 秒（最多 8 次复核），视觉 10 次/600 秒，Jev 200 次/420 秒。单任务中的连续步骤共享最多 200 次 Jev 调用/420 秒；每个局部周期最多 16 次调用，最多 8 次高层复核。会话从 worker 启动计时，任务从提交计时，实际可用时间受剩余会话限制，等待不会重置预算。空闲也计时，不自动调用、不自动续期；到期后须显式重启才开新窗口。 活动 primitive 的 RGB 身份有效期仍为 120 秒，任务窗口和 Qwen 语义刷新不会自动延长它，超期可能提前停止任务。

使用步骤：启动 `--live` 会话 → 输入中文要求 → 点击“提交并执行任务”（或按 Enter）→ 自动观察、规划并尝试执行。Shift+Enter 换行，无需先点击识别按钮。实时会话仅对模拟器开放实验性任务；预览模式不执行抓放。折叠面板里的对话/计划预览不会提交或排队执行，即使模型文案写着“等待执行”。是否已提交及当前进度以界面的运行状态为准，不把模型回复当成动作完成。

双目画面默认约每秒本地采样一次，慢帧不重叠、不积压；可用 `--camera-interval 2` 修改间隔（0.25–10 秒）。这只更新本地图像，不按秒调用付费模型。涉及场景的对话会复用有效视觉证据，过期时先尝试本地图像跟踪，再按需发起一次视觉识别；服务不可用、超时或图像不可靠会显示具体原因。手动识别和悬停控件仅保留在诊断面板。停止会取消待处理请求，双目预览仍继续。

实验性执行使用 **1920×1440 RGB 感知、scaled ROI 局部跟踪和 `shadow_free_rgb_v1` 无阴影渲染**，默认采用 `observer_wrist`：固定观察相机 + 随手腕移动的相机，逐帧使用机器人 FK 更新相机标定；旧相机组合仅供诊断。网页实时预览保持 640×480；界面显示实际感知/预览分辨率、光照和 ROI 配置。高分辨率感知仍会增加 CPU 与处理开销，不能保证实时速度，也不表示遮挡问题已解决。普通预览模式保留原先的低分辨率配置。

`--enable-waypoint` 只开启受限悬停实验；仿真抓放由 `--live` 显式开启。`--workers-only` 用于已有、受支持的同机分离执行环境，只启动 worker，不能把已经运行的预览网页切换为任务模式；普通单机不需要。

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
