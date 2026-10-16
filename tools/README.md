# 维护工具

从仓库根目录运行：

```sh
python -m tools.package_so101 /tmp/embodied-arm-source.zip
```

生成源码快照，包含应用、包、测试、文档和模型资产，不包含虚拟环境、运行时目录和真实配置。`config/provider.example.json` 必须为空密钥模板。
