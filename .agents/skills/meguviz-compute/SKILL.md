---
name: meguviz-compute
description: 在 C、C++ 或 Android NDK 项目中新接入 MeguvizCompute（1.2.2），或将已有 CPU/Vulkan 缓冲区、张量算子、AHardwareBuffer 同步流及 ncnn 计算迁入其资源租约与有界异步提交接口。当涉及 MgcRuntime、MgcLease、AHardwareBuffer 同步围栏、MgcShaderpack 加载或评估与 MeguvizMemory 协作时触发；纯 CPU 内存池管理转交 MeguvizMemory，离线 GLSL 编译与打包转交 MeguvizShaderTools，视频插帧与媒体编解码分别转交 MeguvizRIFE 与 MeguvizMedia。
---

# MeguvizCompute 接入与迁移

`MeguvizCompute`（`1.2.2`）为 C11/C++17 与 Android NDK 应用提供统一的计算资源租约、有界异步提交、内建 CPU/Vulkan 算子、`MgcShaderpack` 运行时加载以及 `AHardwareBuffer` 跨端同步接口。

## 第 1 阶段：盘点工程现状

修改代码前，先读取消费方的构建配置与当前版本信息，并输出简短评估：

1. **检查构建配置与安装前缀**：读取消费方 `CMakeLists.txt`、`CMakePresets.json`、`build.gradle.kts` 及目标 SDK 前缀中的 `lib/cmake/MeguvizCompute/build-info.json`，确认已安装的头文件、`MeguvizComputeConfig.cmake` 与实际工具链匹配。
2. **核对版本与工具链基线**：
   - 需要 CMake `3.31+` 与 C++17 编译器；Windows 端 Compute 默认使用静态 CRT（`MultiThreaded$<$<CONFIG:Debug>:Debug>`），消费方须保持一致。
   - Android 端最低支持 `android-27`（`ANDROID_STL=c++_static`），动态库链接建议启用 16 KiB ELF 页对齐（`-Wl,-z,max-page-size=16384`）。
3. **梳理目标数据流与 `ncnn` 归属**：记录待迁移缓冲的分配、最后写入、任务触发、完成通知与释放点；检查工程中是否已引入其他 `ncnn` 目标。`MeguvizCompute` 独占 `ncnn` 实例，若外部已存在同名 `ncnn` target，配置阶段会直接报错。
4. **明确迁移与回退方案**：先选定一条独立数据流做最小改造，并保留原有 CPU 参考实现或宿主既有路径作为回退，确认方案后再修改源码。

## 第 2 阶段：选择模块与接入依赖

在消费方 `CMakeLists.txt` 中通过 `find_package(MeguvizCompute 1.2.2 CONFIG REQUIRED)` 引入，并仅链接实际需要的目标：

| 使用场景 | CMake Target | 公共头文件 | 边界与前提 |
| --- | --- | --- | --- |
| CPU 拷贝、外部内存导入与通用调度 | `MeguvizCompute::core` | `<meguviz/compute.h>` | 公开 C ABI，不暴露 `ncnn`，可完全独立接入 |
| 归一化、ReLU、统计与 RGB 缩放算子 | `MeguvizCompute::kernels` | `<meguviz/kernels.h>` | 传递链接 `core`，`RESIZE_RGB8` GPU 路径需设备支持 `shaderFloat64` |
| 运行时加载并校验离线着色器包 | `MeguvizCompute::core` | `<meguviz/shaderpack.h>` | 配合 `MeguvizShaderTools` 产出的 pack 目录使用 |
| Android `AHardwareBuffer` 与同步围栏 | `MeguvizCompute::core` | `<meguviz/compute_android.h>` | 先查询域能力，按结果启用同步或回退路径 |
| 内部 `ncnn` 图推理与自定义扩展 | `MeguvizCompute::backend_ncnn_sdk` | `<meguviz/backend_ncnn_sdk.hpp>` | 仅限内部算法适配，须通过 `-DMGC_REQUIRED_PRIVATE_SDK_FINGERPRINT` 锁定指纹 |

**兄弟库与桥接边界**：
- **纯 CPU 内存复用**：不涉及异步任务或 GPU/ncnn 计算时，转交 `MeguvizMemory`。
- **与 `MeguvizMemory` 协作**：仅在消费方明确使用 `MeguvizMemory` 时，按需引入 `MeguvizMemory` 仓库 `adapters/` 下的独立桥接包（`compute_host`、`compute`、`compute_ahb`、`compute_gpu_ahb`），不要在未使用 Memory 时提前引入适配器依赖。
- **离线编译着色器**：转交 `MeguvizShaderTools`（`mgst` 或 `mgst_compile_shaderpack`），Compute 仅负责包加载校验与执行。

## 第 3 阶段：执行最小迁移与生命周期对齐

### 最小 C ABI 接入骨架（COPY 链路）

```c
#include <meguviz/compute.h>
#include <string.h>

// 1. 初始化 runtime / domain / session（所有描述符先清零再用 MGC_HEADER_INIT 初始化）
MgcRuntimeDesc rd = {0}; rd.header = (MgcHeader)MGC_HEADER_INIT(MgcRuntimeDesc);
rd.provider = MGC_PROVIDER_CPU_REFERENCE; rd.memory_budget_bytes = 1024 * 1024;
MgcRuntime *rt = NULL; mgc_runtime_create(&rd, &rt);

MgcDomainDesc dd = {0}; dd.header = (MgcHeader)MGC_HEADER_INIT(MgcDomainDesc);
MgcDomain *dom = NULL; mgc_domain_acquire(rt, &dd, &dom);

MgcSessionDesc sd = {0}; sd.header = (MgcHeader)MGC_HEADER_INIT(MgcSessionDesc);
sd.max_in_flight = 4;
MgcSession *sess = NULL; mgc_session_create(dom, &sd, &sess);

// 2. 创建资源与 WRITE 租约，写入后 unmap 并发布为 READ（每次重写必须递增 content_id）
MgcLease *in_write = NULL, *in_read = NULL, *out_write = NULL;
// ... 通过 mgc_resource_create 与 mgc_lease_create 创建缓冲与租约 ...
void *ptr = NULL; uint64_t bytes = 0;
mgc_lease_map(in_write, &ptr, &bytes);
memcpy(ptr, src_buf, len);
mgc_lease_unmap(in_write);
mgc_lease_publish_read(in_write, &in_read);
mgc_lease_release(in_write);

// 3. 为本次提交单独保留租约引用；仅 MGC_ACCEPTED 时交由 Compute 托管
mgc_lease_retain(in_read);
mgc_lease_retain(out_write);
MgcSubmitDesc job = {0}; job.header = (MgcHeader)MGC_HEADER_INIT(MgcSubmitDesc);
job.operation = MGC_OPERATION_COPY; job.input = in_read; job.output = out_write;
job.byte_count = len; job.request_id = 1;
MgcTicket *ticket = NULL;
MgcStatus st = mgc_submit(sess, &job, &ticket);
if (st != MGC_ACCEPTED) {
    mgc_lease_release(in_read);
    mgc_lease_release(out_write);
} else {
    mgc_session_drain(sess, 1000000000ULL);
    MgcCompletion done = {0}; done.header = (MgcHeader)MGC_HEADER_INIT(MgcCompletion);
    mgc_ticket_query(ticket, &done);
    // 校验 done.terminal == MGC_TERMINAL_SUCCESS && done.status == MGC_SUCCESS
    mgc_ticket_release(ticket);
}

// 4. 关闭顺序：先排空并关闭 session，消费方读取输出后再释放 lease / resource / domain / runtime
mgc_session_close(sess);
// ... 读取 out_write 后释放 in_read、out_write、资源与 domain ...
mgc_runtime_close(rt);
mgc_runtime_release(rt);
```

### 生命周期、线程与所有权铁律

1. **描述符合同与内容版本**：结构体先清零再填 `MGC_HEADER_INIT(类型)`；GPU 资源（`usage != 0`）不支持 CPU 映射。每次重新填充同一资源前，必须使用新的 `content_id` 申请新 `WRITE` 租约。
2. **异步回调时序**：完成回调和上下文释放回调在 session 工作线程执行，且**可能在 `mgc_submit` 返回前已经触发**；回调上下文须事先就绪且回调内禁止抛出 C++ 异常。严禁在完成回调内部调用 `mgc_session_drain` 或 `mgc_session_close`（会返回 `MGC_BUSY`）。
3. **取消、排空与资源保留**：取消（`mgc_ticket_cancel`）、超时或出现 `MGC_DEVICE_LOST` 时，应停止追加新任务并保留仍在使用的资源，直到后台任务彻底结束。`mgc_session_drain` 成功表示已接收任务、回调与提交引用均已结束，消费方持有的输出租约仍须自行释放。
4. **Android `AHardwareBuffer` 与同步围栏**：`mgc_android_import` 仅借用调用方传入的 AHB 与 fd（内部会增加引用并复制 fd），原句柄仍由调用方关闭或释放；通过 `mgc_android_release_to_external` 取得的外部访问必须由同一 lease 调用 `mgc_android_set_acquire_fence` 归还（未下发 GPU 工作时传 `-1`）。
5. **动态库卸载安全**：Linux/Android 动态库消费方须与 Compute 处于同一导出 `meguviz_code_image_anchor_v1` 的模块内，且宿主须保留动态库加载引用直至所有 session 与 owner 成功关闭。

> **按需深入阅读**：完整 C/C++ API 签名、状态码、Android AHB 扩展、`backend_ncnn_sdk` 与 `MeguvizMemory` 桥接详情见 [API 与模块速查](references/api-and-modules.md)；存量代码迁移对照、外部主机内存导入、设备丢失恢复及排障表见 [迁移对照与排障](references/migration-patterns.md)。

## 第 4 阶段：验证与交付清单

1. **消费方验证**：先运行接入工程原有的编译与单测，对同一组输入执行迁移前后的输出比对，确认数值与资源计数无泄漏。
2. **本库源码自检**：在 `MeguvizCompute` 源码根目录按当前配置运行对应验证预设：
   - **CPU 参考版本（Windows）**：
     ```sh
     cmake --preset windows-reference
     cmake --build --preset windows-reference
     ctest --preset windows-reference --no-tests=error -R "^(public_c_copy|external_host|deferred_host_lifecycle)$"
     ```
   - **完整开发 SDK 验证（含 `ncnn` 安装与消费方反向校验）**：运行 `python tools/verify_development_sdk.py --preset windows-cpu`（或 `--preset windows-reference`）。
   - **Android 交叉构建与 ELF 检查**：运行 `python tools/verify_android_sdk.py --abi arm64`（或 `v7`），再在真实设备上补充 `AHardwareBuffer`、同步围栏与设备丢失恢复测试。
3. **失败与回退核对**：确认 `MGC_BUSY`、`MGC_TIMEOUT`、`MGC_DEVICE_LOST` 分支均能正确保留资源不崩溃，且保留的 CPU 参考路径或原有实现可安全回退。
4. **交付摘要**：记录修改文件清单、选用的 Compute SDK 版本与 `private_sdk_fingerprint`（如使用私有 SDK）、实际执行的命令结果，并明确标注哪些结论仅覆盖宿主测试、哪些已通过真机验证。
