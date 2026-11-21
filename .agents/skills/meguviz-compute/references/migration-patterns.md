# MeguvizCompute 迁移对照与排障

本文档提供存量代码向 `MeguvizCompute` 迁移的常见对照写法、并发与所有权规则、设备丢失恢复流程以及高频错误排查表。

## 1. 常见代码迁移范式

### 范式 A：外部已有 CPU 缓冲免复制导入（`mgc_resource_import_host_v1`）

当调用方已有自己管理的 CPU 内存池或外部缓冲时，无需新建 Compute 托管资源再 `memcpy`，直接通过 `mgc_resource_import_host_v1` 导入：

**迁移前（自管缓冲直接传入算子）**：
```c
// 旧代码假设 data 在异步处理期间由调用方手动加锁保活，容易发生悬垂指针或提前释放
submit_custom_worker(user_buf, user_bytes, on_done);
```

**迁移后（绑定引用计数回调的外部主机资源）**：
```c
typedef struct UserBacking {
    void *data;
    uint64_t bytes;
    // ... 引用计数（须保证线程安全） ...
} UserBacking;

static void retain_backing(void *ctx) { /* 原子递增引用计数，严禁抛异常 */ }
static void release_backing(void *ctx) { /* 原子递减引用计数，归零时安全释放 */ }

MgcCapability cap = {0}; cap.header = (MgcHeader)MGC_HEADER_INIT(MgcCapability);
if (mgc_domain_capability(domain, MGC_CAP_EXTERNAL_HOST_MEMORY, &cap) == MGC_SUCCESS && cap.enabled) {
    MgcResourceDesc rdesc = {0}; rdesc.header = (MgcHeader)MGC_HEADER_INIT(MgcResourceDesc);
    rdesc.capacity_bytes = backing->bytes; rdesc.format = MGC_FORMAT_BYTES;
    rdesc.allocation_id = next_alloc_id();

    MgcHostMemoryDesc hdesc = {0}; hdesc.header = (MgcHeader)MGC_HEADER_INIT(MgcHostMemoryDesc);
    hdesc.data = backing->data; hdesc.capacity_bytes = backing->bytes;
    hdesc.access = MGC_ACCESS_READ; // 只读导入禁止随后创建 WRITE lease
    hdesc.context = backing; hdesc.retain = retain_backing; hdesc.release = release_backing;

    MgcResource *res = NULL;
    if (mgc_resource_import_host_v1(domain, &rdesc, &hdesc, &res) == MGC_SUCCESS) {
        // 导入成功后 Compute 已保留一次 backing 引用；
        // 直至该资源及由此派生的全部 lease、已接收任务都结束后才调用 release_backing
    }
}
```

### 范式 B：Android `AHardwareBuffer` 跨端同步交接

当 GLES/Camera 生产者写入 AHB 后交由 Compute 计算，或 Compute 输出 AHB 给外部渲染时，不能仅靠持有 `AHardwareBuffer *` 指针就读写像素，必须走成对预约与 sync fd 交接：

1. **外部生产 → Compute 消费**：
   - 将 AHB 与生产者 fence 导入：`mgc_android_import(domain, &import_desc, &read_lease)`。
   - 调用方保留原始 `AHardwareBuffer` 引用并在合适时机释放/关闭自己的原始 `acquire_fence_fd`（`mgc_android_import` 内部会复制一份）。
   - Compute 异步任务完成后释放 `read_lease`。
2. **Compute 输出 → 外部 GLES/渲染消费**：
   - 对只读输出 lease 调用 `mgc_android_release_to_external(out_lease, &ready_fd)`，取得 `ready_fd` 及 `GENERAL` / `VK_QUEUE_FAMILY_FOREIGN_EXT` 所有权。
   - 外部消费方导入 `ready_fd`（若消费 API 会接管 fd 所有权，须先 `dup`）等待 GPU 完成后再读取，并在归还时调用 `mgc_android_set_acquire_fence(out_lease, done_fd)`（未发起新 GPU 工作则传 `-1`）。
   - **成对铁律**：每次成功的 `mgc_android_release_to_external` 都会额外保留该内容租约，必须由同一个 `out_lease` 调用 `mgc_android_set_acquire_fence` 结束预约，否则资源会一直被锁定且后续提交返回 `MGC_BUSY`。

### 范式 C：消除工程中的重复 `ncnn` 目标

若已有算法模块直接 `find_package(ncnn)` 或用 `add_subdirectory` 编译了私有的 `ncnn` 静态库：

**迁移前**：
```cmake
add_subdirectory(third_party/ncnn)
target_link_libraries(my_algo PRIVATE ncnn)
```

**迁移后**：
```cmake
find_package(MeguvizCompute 1.2.2 CONFIG REQUIRED)
# 若使用 backend_ncnn_sdk，务必校验构建指纹与安装 SDK 一致
# -DMGC_REQUIRED_PRIVATE_SDK_FINGERPRINT=<build-info.json 中的 private_sdk_fingerprint>
target_link_libraries(my_algo PRIVATE MeguvizCompute::backend_ncnn_sdk)
```
在算法代码中使用 `meguviz::with_ncnn_domain`（或用于模型常驻上传的 `meguviz::NcnnDomainOwner`、`with_ncnn_gpu_owned`），避免绕过 Compute 私自创建独立的 Vulkan 设备实例。

### 范式 D：在现有架构中对接 `Shaderpack`

```c
MgcShaderpackDesc sdesc = {0}; sdesc.header = (MgcHeader)MGC_HEADER_INIT(MgcShaderpackDesc);
sdesc.root_path_utf8 = pack_dir; sdesc.root_path_length = (uint32_t)strlen(pack_dir);
sdesc.profile = MGC_SHADER_PROFILE_NCNN_COMPAT;
MgcShaderpack *pack = NULL;
if (mgc_shaderpack_open(domain, &sdesc, &pack) == MGC_SUCCESS) {
    MgcShaderArtifactRequest req = {0};
    req.header = (MgcHeader)MGC_HEADER_INIT(MgcShaderArtifactRequest);
    req.logical_shader_id = "my_app/kernel";
    req.logical_shader_id_length = (uint32_t)strlen(req.logical_shader_id);
    req.variant_key = expected_variant_hash;
    req.effective_options_hash = expected_options_hash;
    MgcShaderArtifactView view = {0}; view.header = (MgcHeader)MGC_HEADER_INIT(MgcShaderArtifactView);
    if (mgc_shaderpack_select(pack, &req, &view) == MGC_SUCCESS) {
        // view.spirv_words 在 mgc_shaderpack_release(pack) 前一直有效
    }
    mgc_shaderpack_release(pack);
}
```

## 2. 所有权、并发与生命周期核心规则

| 环节 | 必须遵守的规则 | 违规后果 |
| --- | --- | --- |
| `mgc_submit` 提交引用 | 调用前对每个非空 `input` / `output` 租约单独 `mgc_lease_retain`；仅返回 `MGC_ACCEPTED` 时引用归 Compute 托管 | 拒绝提交时未归还导致泄漏；未额外 retain 导致提前析构崩溃 |
| 完成回调与上下文 | `retain_context` 与 `release_context` 必须成对提供或同为空；回调可能在 `mgc_submit` 返回前执行；回调内严禁抛异常 | 悬垂上下文访问或跨越 C ABI 异常导致进程直接终止 |
| 回调内操作限制 | 回调内可查询、取消或发起新 `mgc_submit`，**严禁**调用 `mgc_session_drain` 或 `mgc_session_close` | 触发同线程自死锁防护，返回 `MGC_BUSY` |
| `mgc_lease_map` / `unmap` | 仅限 `usage == 0` 的 CPU 资源；使用返回的指针后立刻 `unmap`；不要 `free()` 映射指针；不要在未 `unmap` 时发布或提交 | 返回 `MGC_UNSUPPORTED_CAPABILITY` 或 `MGC_BUSY` |
| 资源内容复用 | 每次重写同一 `MgcResource` 时，必须递增或更换新的 `content_id` 并申请新的 `WRITE` 租约 | 读写版本混用，或被内容版本保护机制拒绝 |
| 取消与设备丢失 | 取消或发生设备丢失时，停止提交并保留资源，调用 `mgc_session_drain` 或 `mgc_domain_service` 确认后台工作全部退出后再销毁对象 | `mgc_runtime_close` / `mgc_session_close` 返回 `MGC_BUSY` |

## 3. 设备丢失（`MGC_DEVICE_LOST`）恢复与安全回退

1. **禁止原地重绑旧设备**：发生 `MGC_DEVICE_LOST` 后，原有 `MgcDomain` 绝不会重新绑定到新的 `VkDevice`。
2. **推进并确认旧工作结束**：调用 `mgc_domain_state(domain, &state)` 查看 `lost`、`device_retired` 与 `detached` 标志，并通过 `mgc_domain_service(domain, wait_ns)`（及如有外部 GL/AHB 预约则完成其清理）推进后台退出。
3. **重建域或回退**：旧域完全退出清理后，保留原有输出以完成排障并释放旧句柄，再重新 `mgc_domain_acquire` 获取新域，或回退到 `MGC_PROVIDER_CPU_REFERENCE` / 消费方既有软件实现。

## 4. 高频错误码与构建失败排查表

| 现象 / 返回码 | 常见根因 | 排查与修复动作 |
| --- | --- | --- |
| CMake 报错 `private SDK fingerprint mismatch` | `MGC_REQUIRED_PRIVATE_SDK_FINGERPRINT` 与安装的 SDK 不一致 | 读取 `<prefix>/lib/cmake/MeguvizCompute/build-info.json` 中的 `private_sdk_fingerprint` 重新配置，或重新编译消费方 |
| CMake 报错 `Preexisting ncnn target is not owned...` | 消费方在 `find_package(MeguvizCompute)` 前已引入私有 `ncnn` | 移除消费方自己的 `ncnn` 依赖，全部改为链接 `MeguvizCompute::backend_ncnn_sdk` |
| `MGC_UNSUPPORTED_ABI (-2)` | 结构体未清零或遗漏 `MGC_HEADER_INIT(Type)` | 检查所有 `Mgc*Desc`、`MgcCompletion`、`MgcCapability`、`MgcBuildInfo` 的 `header` 初始化 |
| `MGC_BUSY (-7)` | 存在未 `unmap` 的租约、未结束的外部 AHB 预约、在回调中 `drain`/`close`，或资源仍在途 | 检查 `mgc_android_release_to_external` 是否已调用 `set_acquire_fence` 配对结束；确认 `drain` 在工作线程回调外调用 |
| `MGC_OUT_OF_MEMORY (-10)` | 累计注册资源超出 `MgcRuntimeDesc.memory_budget_bytes` | 及时 `mgc_resource_release` 不再使用的资源，或合理调高预算；注意 AHB 按 `max(row_extent, allocationSize)` 计入 |
| `MGC_ARTIFACT_MISMATCH (-6)` | Shaderpack 路径、文件哈希、`variant_key` 或 `effective_options_hash` 不匹配 | 使用 `mgst verify --pack <dir>` 校验证据，并核对运行期传入的 `ncnn_shader_options_json` 哈希 |
| `MGC_PRODUCER_FAILED (-15)` | Android AHB 外部生产者 fence 报错或 guarded cleanup 确认失败 | 丢弃受影响的输出内容，保留未确信安全的生产者资源，不复用该输出内容 |
| Windows 链接报 CRT 冲突（`LNK2038` / `RuntimeLibrary`） | 消费方使用了动态 `/MD`，而 Compute 使用静态 `/MT` | 在消费方 CMake 中设置 `CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"` |
