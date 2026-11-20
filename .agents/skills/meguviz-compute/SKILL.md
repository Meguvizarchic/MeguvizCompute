---
name: meguviz-compute
description: 在 C、C++ 或 Android NDK 项目中接入 MeguvizCompute，或迁移已有 CPU/Vulkan 缓冲与任务到其资源、租约和有界提交接口。也适用于评估与 MeguvizMemory 的可选缓冲协作。
---

# 接入 MeguvizCompute

## 选定一条现有数据流

找到调用方的构建目标和准备迁移的一个操作，记录当前分配、最后访问、完成通知和释放位置。优先使用项目已锁定的 Compute SDK。若引用了本 skill 所在仓库之外的消费者，先找本地 SDK 前缀或对应源码 checkout。以所选版本的头文件、CMake 导出和 `build-info.json` 为准，复用现有目标与构建参数。

| 需求 | 接入点 |
| --- | --- |
| CPU 操作或首次接入 | `find_package(MeguvizCompute 1.2.2 CONFIG REQUIRED)` 并链接 `MeguvizCompute::core`。公共头为 `<meguviz/compute.h>`，从 `examples/c_consumer.c` 的 COPY 链路开始。 |
| Android GPU/AHardwareBuffer | 同一 Compute domain 内使用 `<meguviz/compute_android.h>`。检查实际 SDK/ABI、目标缓冲区的格式与 usage、所用同步模式和目标设备能力。保留调用方已有的兼容路径。 |
| 内部 ncnn 算法适配 | 仅在适配器确实需要时链接 `MeguvizCompute::backend_ncnn_sdk`，以该 ABI 的 `MGC_REQUIRED_PRIVATE_SDK_FINGERPRINT` 匹配 Compute SDK。由 Compute 提供唯一 ncnn owner。 |

## 实现最小迁移

1. 先建立 runtime/domain/session，在描述符填 `MGC_HEADER_INIT(类型)` 和必需字段，其余字段置零。为每次写入建立新的 content ID，取得 WRITE 后再写。仅在对应 lease 下使用 map 指针，结束后 unmap；按头文件的握手将已完成的数据交给 READ。
2. 准备单独的提交引用。只有 `MGC_ACCEPTED` 转移这些引用，拒绝时调用方归还。仍需在外部使用的 lease 保留独立引用。完成和上下文释放回调可能在提交函数返回前执行，避免让它们依赖提交函数尚未写回的数据。
3. 在完成后检查终态和状态。取消、超时或设备丢失时停止追加工作并保留仍在使用的资源，直到完成物理退役。成功 drain 才覆盖 session 的已接受工作、回调和提交引用；消费者持有的输出单独释放。Android 外部访问成功取得的 reservation 由同一 lease 归还，遵守头文件里的 owned/borrowed fd 语义。

### 与 Memory 协作时

只有调用方选择 Memory 时才接入适配器，Compute 公共 SDK 可以独立使用。对 Memory 持有的 CPU 数据，核对两个 SDK 的 ABI、工具链及 MSVC CRT，使用 Memory 源码 `adapters/compute_host` 中的 `MeguvizMemoryComputeHostBridge::host_bridge` 和 `mgmem_compute_host`（C++ `mgmem::compute::host`），并确认所选 Compute 提供 `mgc_resource_import_host_v1`。导入省去 CPU backing 的复制；借用和错误回调上下文一直保留至资源、租约及执行引用归还。

从 Compute 向 Memory 交付时，`adapters/compute` 的 `read_bridge` 接收完整且就绪的 CPU READ lease；Arena 需要该 SDK 的独立 `arena_read_bridge` 目标。一次性上传/下载由 `adapters/compute_host/TRANSFER.md` 的桥提交 Compute COPY，GPU 上传/下载可能仍经 staging。对同一份不变上传或重复下载，可按该目录 `REUSE.md` 的实际接口提前准备 staging。GPU-only、受保护内容和 AHB 仍走原设备同步与所有权路径。

## 验证并交付

- 优先运行消费方原有构建和同输入的输出比对。Windows 源码可在仓库根目录运行下列参考检查。其中 `external_host` 覆盖外部 backing，`deferred_host_lifecycle` 覆盖异步退役。

  ```sh
  cmake --preset windows-reference
  cmake --build --preset windows-reference
  ctest --preset windows-reference --no-tests=error -R "^(public_c_copy|external_host|deferred_host_lifecycle)$"
  ```

- 迁移 Android 时，单列交叉构建、真实设备的 AHB/同步验证和回退验证。记录消费者改动、选用的 SDK 身份和实际执行过的检查。
