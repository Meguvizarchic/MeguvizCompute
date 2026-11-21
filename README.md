<p align="center"><img src="docs/brand/MeguvizCompute.svg" width="96" alt="MeguvizCompute"></p>

# MeguvizCompute

面向 C/C++ 应用的计算库（当前版本 `1.2.2`），提供 CPU 与 Vulkan 后端、有界异步任务和 Android `AHardwareBuffer` 硬件缓冲互操作。

- 通过资源（`MgcResource`）和读写租约（`MgcLease`）管理缓冲区生命周期，任务完成后再复用输入与输出。
- 内建字节拷贝、归一化、ReLU、RGB8 缩放与 Float32 张量统计算子，并提供统一 `ncnn` 计算与离线 `MgcShaderpack` 运行时加载入口。
- Android 工程可按设备能力导入或分配 `AHardwareBuffer`，通过同步围栏在外部生产/消费方与 Compute 之间安全交接。

## 快速开始

需要 CMake `3.31+` 与 C++17 工具链。先构建纯 CPU 参考版本：

```sh
cmake -S . -B build/reference -DCMAKE_BUILD_TYPE=Release -DMGC_WITH_NCNN=OFF
cmake --build build/reference --config Release
cmake --install build/reference --config Release --prefix ./sdk
```

在调用方将安装目录加入 `CMAKE_PREFIX_PATH`，包含 `<meguviz/compute.h>` 并链接：

```cmake
find_package(MeguvizCompute 1.2.2 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE MeguvizCompute::core)
```

启用 `ncnn` 或 Android Vulkan 时，按仓库预设配置对应的依赖源码；`MeguvizRIFE` 等上层算法库共用同一套 Compute SDK。详见 [构建预设](CMakePresets.json) 与 [ncnn 依赖与补丁](third_party/locks/ncnn.lock.json)。

## 文档与接入指南

- [AI 接入与迁移指南](.agents/skills/meguviz-compute/SKILL.md)
- [完整 C 调用示例](examples/c_consumer.c)
- [Android 缓冲区与同步接口](include/meguviz/compute_android.h)
- [张量统计 CLI 说明](docs/TENSOR_STATISTICS_CLI.md)
- [私有 SDK 域拥有者契约](docs/DOMAIN_OWNERS.md)
- [私有 SDK 延迟异步任务契约](docs/DEFERRED_JOBS.md)
- [AHardwareBuffer 生产者围栏契约](docs/guarded_producer.md)

[MIT 许可证](LICENSE) · [第三方声明](NOTICE.md)
