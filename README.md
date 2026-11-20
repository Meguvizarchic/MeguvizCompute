<p align="center"><img src="docs/brand/MeguvizCompute.svg" width="96" alt="MeguvizCompute"></p>

# MeguvizCompute

面向 C/C++ 应用的计算库，提供 CPU 与 Vulkan 后端、异步任务和 Android 硬件缓冲区互操作。

- 用资源和读写租约管理缓冲区。提交后等待任务完成再复用其输入和输出。
- 提供图像缩放、张量统计及 ncnn 计算入口。
- Android 项目可按设备能力导入 `AHardwareBuffer`，并使用同步围栏在生产者和消费者之间交接。

## 快速开始

需要 CMake 3.31+ 和 C++17 工具链。先构建 CPU 参考版本。

```sh
cmake -S . -B build/reference -DCMAKE_BUILD_TYPE=Release -DMGC_WITH_NCNN=OFF
cmake --build build/reference --config Release
cmake --install build/reference --config Release --prefix ./sdk
```

在调用方将安装目录加入 `CMAKE_PREFIX_PATH`，包含 `<meguviz/compute.h>` 并链接：

```cmake
find_package(MeguvizCompute 1.2.1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE MeguvizCompute::core)
```

使用 ncnn 或 Android Vulkan 时，按仓库预设配置相应的依赖源码。RIFE 等算法库使用同批 Compute SDK。[构建预设](CMakePresets.json) · [ncnn 依赖与补丁](third_party/locks/ncnn.lock.json)

## 文档

- [完整 C 调用示例](examples/c_consumer.c)
- [Android 缓冲区与同步接口](include/meguviz/compute_android.h)
- [张量统计示例](docs/TENSOR_STATISTICS_CLI.md)
- [Codex 接入与迁移](.agents/skills/meguviz-compute/SKILL.md)

[MIT 许可证](LICENSE) · [第三方声明](NOTICE.md)
