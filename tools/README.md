# 维护工具

从仓库根目录运行：

```sh
python -m tools.package_so101 /tmp/embodied-arm-source.zip
```

生成源码快照，包含应用、包、测试、文档和模型资产，不包含虚拟环境、运行时目录和真实配置。`config/provider.example.json` 是大脑/小脑分组模板，两组 key 都必须为空；旧格式仅保留为 `config/provider-legacy.example.json`，同样不允许非空 key。
