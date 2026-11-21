# MeguvizCompute API 与模块速查

本文档整理 `MeguvizCompute` `1.2.2` 的 CMake 构建目标、公共 C ABI、Android `AHardwareBuffer` 接口、Shaderpack 加载接口、内部 `ncnn` 私有 SDK，以及与 `MeguvizMemory` 协同的桥接边界。

## 1. 构建要求与预设

- **CMake 与语言标准**：CMake `3.31+`，C11 / C++17（`CMAKE_CXX_STANDARD=17`）。
- **MSVC 运行时**：静态链接 CRT（`MultiThreaded$<$<CONFIG:Debug>:Debug>`），消费方目标须保持一致。
- **Android NDK 工具链**：最低 `android-27`（API 27），`ANDROID_STL=c++_static`。使用 `MGC_ANDROID_STATIC_OPENMP=ON` 时仅支持 `arm64-v8a` 与 `armeabi-v7a`，并校验 NDK 30.0.16248370 静态 `libomp.a` 及 `NOTICE` 的 SHA-256。动态库消费方链接时建议添加 `-Wl,-z,max-page-size=16384`。

| CMake 选项 / 预设 | 说明 | 默认值 / 备注 |
| --- | --- | --- |
| `MGC_WITH_NCNN` | 构建锁定的真实 `ncnn` provider 与私有 SDK | `ON`；设为 `OFF` 可构建纯 CPU 参考版（`windows-reference`） |
| `MGC_VULKAN` | 启用由 `ncnn` 托管的 Vulkan domain | `OFF`；启用时需同时提供 `MGC_GLSLANG_SOURCE` |
| `MGC_NCNN_SOURCE` | 锁定的 `ncnn` 源码路径（commit `7e36d25cb153092ba87a60530455c7376fedf452`） | 配置阶段强制校验 commit 与 `third_party/patches/0001-vulkan-retirement-sync-fd.patch` 哈希 |
| `MGC_NCNN_LAYER_MANIFEST` | 可选的模型与 `ncnn` 算子裁剪清单文件路径 | 启用需 `MGC_WITH_NCNN=ON`，裁剪状态写入 `build-info.json` |
| `MGC_REQUIRED_PRIVATE_SDK_FINGERPRINT` | 期望匹配的私有 SDK 指纹 | 不匹配则在 `find_package` 或配置期报错拒绝 |
| `MGC_GUARDED_PRODUCER_FENCE_EXPERIMENTAL` | 实验性受保护 AHB 生产者交接 | 默认 `OFF`，开启会改变 `build_fingerprint` 并激活能力 `19` |

预设列表（见 `CMakePresets.json`）：`windows-reference`、`windows-cpu`、`android-arm64-api27`、`android-v7-api27`、`linux-cpu`。

## 2. CMake 导入目标

使用 `find_package(MeguvizCompute 1.2.2 CONFIG REQUIRED)` 导入包后，可按需链接以下目标：

| CMake Target | 类型 | 包含头文件 | 适用场景与限制 |
| --- | --- | --- | --- |
| `MeguvizCompute::core` | `STATIC` | `<meguviz/compute.h>`、`<meguviz/compute_android.h>`、`<meguviz/shaderpack.h>` | 公共 C ABI、Runtime/Domain/Resource/Lease/Session、外部主机内存导入与 Shaderpack 加载 |
| `MeguvizCompute::kernels` | `INTERFACE` | `<meguviz/kernels.h>` | 传递链接 `core`，提供 5 个内建计算算子契约与 `MgcStatisticsF32` 结构 |
| `MeguvizCompute::backend_ncnn_sdk` | `INTERFACE` | `<meguviz/backend_ncnn_sdk.hpp>`、`<meguviz/compute_job.h>`、`<meguviz/ncnn_options_json.hpp>` | 仅在 `MGC_WITH_NCNN=ON` 时安装；链接 `core` 与内部 `ncnn`，强制校验 `MGC_PRIVATE_SDK_FINGERPRINT` |

> 注意：`MeguvizCompute` 是工程内唯一可声明 `ncnn` 目标的拥有者。若消费方在 `find_package(MeguvizCompute)` 之前已自行定义了 `ncnn` target，配置将直接报错退出。

## 3. 公共 C ABI 速查（`<meguviz/compute.h>`）

### 3.1 状态码（`MgcStatus`）

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `MGC_SUCCESS` | `0` | 同步调用成功完成 |
| `MGC_ACCEPTED` | `1` | `mgc_submit` 已接收任务并接管该次提交持有的租约引用 |
| `MGC_INVALID_ARGUMENT` | `-1` | 参数空指针、字段越界或句柄无效 |
| `MGC_UNSUPPORTED_ABI` | `-2` | `MgcHeader` 大小或主/次版本不兼容 |
| `MGC_UNSUPPORTED_CAPABILITY` | `-3` | 当前 provider 或硬件不支持所请求能力 |
| `MGC_INCOMPATIBLE_DOMAIN` | `-4` | 跨 domain 提交、线程重入冲突或域不匹配 |
| `MGC_MODEL_MISMATCH` | `-5` | 模型加载或格式不匹配 |
| `MGC_ARTIFACT_MISMATCH` | `-6` | Shaderpack 清单、SPIR-V、反射或 Options 哈希不匹配 |
| `MGC_BUSY` | `-7` | 资源在途、存在外部访问未归还、或从回调内调用了 `drain`/`close` |
| `MGC_BACKPRESSURE` | `-8` | Session 在途任务数达到 `max_in_flight` 上限 |
| `MGC_DEADLINE_REJECTED` | `-9` | 无法满足请求期限 |
| `MGC_OUT_OF_MEMORY` | `-10` | 超出 `memory_budget_bytes` 或底层分配失败 |
| `MGC_DEVICE_LOST` | `-11` | Vulkan 逻辑设备丢失 |
| `MGC_TIMEOUT` | `-12` | `wait_ready` 或 `mgc_session_drain` 等待超时 |
| `MGC_CANCELLED` | `-13` | 任务已被取消 |
| `MGC_INTERNAL_ERROR` | `-14` | 内部状态异常 |
| `MGC_PRODUCER_FAILED` | `-15` | 外部生产者同步围栏失败或清理失败 |

### 3.2 关键枚举与宏

- **ABI 头初始化**：`MGC_HEADER_INIT(type)`（`MGC_ABI_MAJOR = 0u`，`MGC_ABI_MINOR = 1u`）。
- **Provider**：`MGC_PROVIDER_CPU_REFERENCE (1)`、`MGC_PROVIDER_NCNN_CPU (2)`、`MGC_PROVIDER_NCNN_VULKAN (3)`。
- **访问权限**：`MGC_ACCESS_READ (1)`、`MGC_ACCESS_WRITE (2)`。
- **资源用途（`usage`）**：`0` 为可 CPU 映射的显式主机存储；`MGC_RESOURCE_GPU_BUFFER (1)`；`MGC_RESOURCE_ANDROID_AHB (2)`。GPU 分配不可调用 `mgc_lease_map`。
- **数据格式**：`MGC_FORMAT_BYTES (1)`、`MGC_FORMAT_RGB8 (2)`、`MGC_FORMAT_RGBA8 (3)`、`MGC_FORMAT_FLOAT32 (4)`。
- **内建操作（`operation`）**：`MGC_OPERATION_COPY (1)`、`MGC_OPERATION_NORMALIZE_F32 (2)`、`MGC_OPERATION_RESIZE_RGB8 (3)`、`MGC_OPERATION_STATISTICS_F32 (4)`、`MGC_OPERATION_RELU_F32 (5)`。
- **任务终态（`MgcCompletion.terminal`）**：`MGC_TERMINAL_SUCCESS (1)`、`MGC_TERMINAL_FAILED (2)`、`MGC_TERMINAL_CANCELLED (3)`、`MGC_TERMINAL_DEVICE_LOST (4)`。
- **能力编号（`mgc_domain_capability`）**：`MGC_CAP_CPU_REFERENCE (1)` 到 `MGC_CAP_GUARDED_PRODUCER_FENCE (19)`（完整列表见 `include/meguviz/compute.h`）。

### 3.3 核心函数签名

```c
// 构建身份与版本探测
MgcStatus mgc_query_build_info(MgcBuildInfo *info);

// Runtime 与 Domain 生命周期
MgcStatus mgc_runtime_create(const MgcRuntimeDesc *desc, MgcRuntime **runtime);
void      mgc_runtime_retain(MgcRuntime *runtime);
void      mgc_runtime_release(MgcRuntime *runtime);
MgcStatus mgc_runtime_close(MgcRuntime *runtime);

MgcStatus mgc_domain_acquire(MgcRuntime *runtime, const MgcDomainDesc *desc, MgcDomain **domain);
MgcStatus mgc_domain_identity(MgcDomain *domain, uint64_t *domain_id, uint64_t *generation);
MgcStatus mgc_domain_state(MgcDomain *domain, MgcDomainStateInfo *state);
MgcStatus mgc_domain_service(MgcDomain *domain, uint64_t wait_ns);
MgcStatus mgc_domain_capability(MgcDomain *domain, uint32_t feature, MgcCapability *capability);
MgcStatus mgc_domain_allocation_stats(MgcDomain *domain, MgcAllocationStats *stats);
MgcStatus mgc_runtime_allocation_stats(MgcRuntime *runtime, MgcAllocationStats *stats);
void      mgc_domain_retain(MgcDomain *domain);
void      mgc_domain_release(MgcDomain *domain);

// Resource 与外部主机内存导入
MgcStatus mgc_resource_create(MgcDomain *domain, const MgcResourceDesc *desc, MgcResource **resource);
MgcStatus mgc_resource_import_host_v1(MgcDomain *domain, const MgcResourceDesc *desc,
                                      const MgcHostMemoryDesc *backing, MgcResource **resource);
void      mgc_resource_retain(MgcResource *resource);
void      mgc_resource_release(MgcResource *resource);

// Lease 与数据访问
MgcStatus mgc_lease_create(MgcResource *resource, const MgcLeaseDesc *desc, MgcLease **lease);
void      mgc_lease_retain(MgcLease *lease);
void      mgc_lease_release(MgcLease *lease);
MgcStatus mgc_lease_info(MgcLease *lease, MgcLeaseInfo *info);
MgcStatus mgc_lease_resource(MgcLease *lease, MgcResource **retained_resource);
MgcStatus mgc_lease_map(MgcLease *lease, void **data, uint64_t *size_bytes);
MgcStatus mgc_lease_unmap(MgcLease *lease);
MgcStatus mgc_lease_wait_ready(MgcLease *lease, uint64_t timeout_ns);
MgcStatus mgc_lease_publish_read(MgcLease *writing_lease, MgcLease **read_lease);

// Session、提交与 Ticket 控制
MgcStatus mgc_session_create(MgcDomain *domain, const MgcSessionDesc *desc, MgcSession **session);
MgcStatus mgc_session_close(MgcSession *session);
MgcStatus mgc_submit(MgcSession *session, const MgcSubmitDesc *desc, MgcTicket **ticket);
void      mgc_ticket_retain(MgcTicket *ticket);
void      mgc_ticket_release(MgcTicket *ticket);
MgcStatus mgc_ticket_cancel(MgcTicket *ticket);
MgcStatus mgc_ticket_query(MgcTicket *ticket, MgcCompletion *completion);
MgcStatus mgc_ticket_execution_info(MgcTicket *ticket, MgcExecutionInfo *info);
MgcStatus mgc_session_drain(MgcSession *session, uint64_t timeout_ns);
```

## 4. 内建算子与 Shaderpack 接口

- **`<meguviz/kernels.h>`**：
  - `MGC_OPERATION_COPY`：按 `byte_count` 拷贝（`input` 为 `READ`，`output` 为 `WRITE`）。
  - `MGC_OPERATION_NORMALIZE_F32`：`byte_count` 须为 4 的倍数，输出 `(x - min) / (max - min)`（`min == max` 时输出 0）。
  - `MGC_OPERATION_RELU_F32`：`byte_count` 须为 4 的倍数，逐元素计算 `max(0, x)`。
  - `MGC_OPERATION_STATISTICS_F32`：输出结构体 `MgcStatisticsF32`（含 `count`、`finite_count`、`sum`、`minimum`、`maximum`）。
  - `MGC_OPERATION_RESIZE_RGB8`：**`byte_count` 必须为 `0`**，尺寸与行步长以资源描述符为准；GPU 路径需设备开启 `shaderFloat64`，否则纯主机资源回退 CPU、仅设备端资源返回 `MGC_UNSUPPORTED_CAPABILITY`。
- **`<meguviz/shaderpack.h>`**：
  - Profile 常量：`MGC_SHADER_PROFILE_NCNN_COMPAT (1)`、`MGC_SHADER_PROFILE_NATIVE_VULKAN (2)`。
  - 接口：`mgc_shaderpack_open`、`mgc_shaderpack_retain`、`mgc_shaderpack_release`、`mgc_shaderpack_domain_identity`、`mgc_shaderpack_select`。
  - `mgc_shaderpack_open` 会校验路径不逃逸 `root_path_utf8`、清单哈希与 SPIR-V/反射哈希；`MgcShaderArtifactView.spirv_words` 指针仅借用到 `mgc_shaderpack_release` 为止。

## 5. Android `AHardwareBuffer` 扩展（`<meguviz/compute_android.h>`）

- 同步模式：`MGC_ANDROID_HOST_WAIT_COMPAT (0)`、`MGC_ANDROID_GPU_SYNC_FD (1)`。
- 布局常量：`MGC_ANDROID_LAYOUT_GENERAL (1)`（对应 Vulkan `GENERAL` 与 `VK_QUEUE_FAMILY_FOREIGN_EXT`，字段置 0 即可自动选择默认外部交接参数，无需包含 Vulkan 头文件）。
- 核心函数：
  - `mgc_android_import(domain, desc, &lease)` / `mgc_android_allocate(domain, desc, &lease)`
  - `mgc_android_get_buffer(lease, &borrowed_buffer)`（借用指针，存活至 lease 释放）
  - `mgc_android_wait_ready(lease, timeout_ns)`
  - `mgc_android_release_to_external(lease, &owned_fd)`（别名 `mgc_android_export_release_fence`）：启动一次有界外部访问并转移新的 `owned_fd`（`-1` 表示无挂起依赖）。
  - `mgc_android_set_acquire_fence(lease, borrowed_fd)`：结束同 lease 的外部访问（未下发 GPU 工作须传 `-1`；内部复制 fd，调用方保留原 fd 所有权）。
  - 实验性受保护交接（仅当 `MGC_CAP_GUARDED_PRODUCER_FENCE` 启用时可用）：`mgc_android_set_guarded_producer_fence(lease, borrowed_fd, &generation)` 与 `mgc_android_finish_guarded_producer(lease, generation, cleanup_status)`。

## 6. 私有 `ncnn` 源级 SDK（仅内部算法适配）

仅在链接 `MeguvizCompute::backend_ncnn_sdk` 且校验通过 `MGC_REQUIRED_PRIVATE_SDK_FINGERPRINT` 时可用：

- `<meguviz/backend_ncnn_sdk.hpp>`：
  - `meguviz::with_ncnn_domain` / `meguviz::with_ncnn_domain_cleanup`
  - `meguviz::NcnnDomainOwner` 管理（`ncnn_domain_owner_create`、`with_ncnn_domain_owner`、`try_with_ncnn_domain_owner`、`try_with_ncnn_domain_owner_retire`、`ncnn_domain_owner_pending`、`ncnn_domain_owner_release`，同一物理 `VkDevice` 上最多 16 个 owner）
  - `meguviz::with_ncnn_gpu`（同步等待结束）与 `meguviz::with_ncnn_gpu_owned`（配合延迟异步任务与 `keepalive`）
  - `meguviz::ncnn_shaderpack_pipeline`（需传入 `const ncnn::Option &effective_options`，旧版无 `Option` 重载返回 `MGC_UNSUPPORTED_ABI`）、`ncnn_shader_target_facts`、`ncnn_shader_options_json`、`private_sdk_fingerprint`
- `<meguviz/compute_job.h>`：
  - `MgcHostJobDesc`、`mgc_private_submit_host_job`、`mgc_private_ticket_defer`、`mgc_private_ticket_release_retired_write`、`mgc_private_session_cancel`、`mgc_private_session_domain`。

## 7. 与 `MeguvizMemory` 协作的可选桥接

下列桥接包全部位于 `MeguvizMemory` 仓库的 `adapters/` 目录，均以独立 CMake 包形式提供，Compute 本身不依赖 `MeguvizMemory`：

| 协作需求 | `MeguvizMemory` 适配器目录 | CMake 包与目标 | 关键入口与前提 |
| --- | --- | --- | --- |
| Memory CPU 缓冲零复制导入 Compute | `adapters/compute_host` | `MeguvizMemoryComputeHostBridge::host_bridge` | `<mgmem/compute_host.h>`（`mgmem_compute_host`），需 Compute 提供 `mgc_resource_import_host_v1` |
| Memory 与 Compute 间上传/下载 | `adapters/compute_host` | `MeguvizMemoryComputeHostBridge::host_bridge` | `<mgmem/compute_transfer.h>`（`mgmem_upload` / `mgmem_download`）与 `<mgmem/compute_stage.h>`（`prepare_upload` / `send`），复用 Compute `COPY`，不支持 AHB |
| Compute CPU READ lease 交回 Memory | `adapters/compute` | `MeguvizMemoryComputeReadBridge::read_bridge` / `arena_read_bridge` | `<mgmem/compute_read.h>`（`mgmem_compute_read` / `mgmem_compute_read_arena`） |
| Compute AHB 映射为 Memory CPU 读/写缓冲 | `adapters/compute_ahb` | `MeguvizMemoryAhbReadBridge::read_bridge` | `<mgmem/ahb_read.h>` / `ahb_slot.h` / `ahb_map.h`（Android API 26+，单层 RGBA8） |
| Memory 写入 GPU-only AHB 协调 | `adapters/compute_gpu_ahb` | `MeguvizMemoryGpuAhbBridge::gpu_ahb` | `<mgmem/gpu_ahb.h>`（`begin` / `publish` / `retire` / `cancel`） |
