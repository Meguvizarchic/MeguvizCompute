<p><img src="docs/brand/MeguvizCompute.svg" width="96" alt="MeguvizCompute 图标"></p>

# MeguvizCompute

当前候选已实际实现 Vulkan buffer、ncnn 小图与通用 GPU 核、RGBA8 AHB 互操作、最终提交异步退役和运行时 shaderpack。代码已冻结交协调者统一验证；`STATUS.json` 标记 implemented-unverified。仅完成必要 Android arm64 功能对象编译，不代表真机资格或性能成绩。

公开头位于 `include/meguviz`。`usage=0` 是显式 CPU staging，`MGC_RESOURCE_GPU_BUFFER` 为真实 VkMat；GPU/AHB 不能 map。`mgc_lease_publish_read` 在完成写入后封存内容，所有引用转为只读。`mgc_android_import` 独立 acquire AHB/dup fd，内容 retain/release 回调直到最终引用结束，不能用对象 refcount 代替 producer 内容复用权限。

私有 SDK 的 `with_ncnn_gpu` 仅允许同域作用域：callback 录制命令，外层统一提交、物理退役和 host finalize。view.buffer 是完整 allocation，shader 必须应用 info.byte_offset/byte_length。AHB view 为 typed RGBA8 storage image，逐资源检查格式能力；Compute 负责 GENERAL/FOREIGN ownership，外部消费者每次读完通过 set_acquire_fence 注册完成依赖。HOST_WAIT_COMPAT 与 binary GPU_SYNC_FD 分别实现，缺能力明确拒绝。

ncnn 原始 pin 保持 `7e36d25cb153092ba87a60530455c7376fedf452`，glslang 保持 `780b6a209132dc74e99458f62ebe5fa7e57c41cf`。构建消费独立 `../dependencies/ncnn-meguviz` fork，精确 patch 字节在 `third_party/patches` 和 `third_party/locks/ncnn.lock.json` 登记并进入私有 SDK fingerprint。共享 `../dependencies/ncnn` 未修改。fork 保留 recording hidden submit/wait 阈值，不能宣称 FULL_RECORD_ASYNC。

Compute 默认保持 OpenMP OFF；Android ARM ncnn 构建可显式设置 `MGC_ANDROID_STATIC_OPENMP=ON`，使用锁定的 NDK 静态 `libomp.a`。启用时实际 ABI 归档与 LLVM NOTICE 的 SHA-256 必须匹配锁并进入私有 SDK 指纹；安装包包含相应通知。运行库或通知身份改变时构建会拒绝，而不是沿用旧指纹。

GPU kernels 支持 copy、有限 FP32 ReLU/normalize/statistics 与半像素 RGB8 resize。统计回读有界分块小摘要，GPU sum 使用 FP32 缩放补偿归约，设备数值容差需最终 oracle 验证；不把它伪装成 CPU double 累加的逐位相等。公开 GPU buffer 子区间须满足设备 storage offset 对齐；私有完整 allocation view 支持 shader 显式 offset。shaderpack open/select 可独立在 host 验证，ncnn pipeline 缓存与 native-vulkan artifact view 分开；runtime glslang 保留。

独立消费者：`mgc_vulkan_chain`（无设备返回77）、`mgc_shaderpack_consumer pack-root variant-key effective-options-hash`、既有 public C/private graph/tensor_statistics。可重复构建命令为 CMake presets `windows-reference`、`windows-cpu`、`android-arm64-api27`、`android-v7-api27`。最终 build/install/运行由协调者统一执行，原证据与失败日志均保留在 `build/reports`。

SDK 是 STATIC/PIC development package；正式单 DSO 组装由 Media distribution 负责。不含研究模型、生产权重或自动下载器；未外部发布、未推送。设备缺失只延后资格，不代表以上代码未实施。

C4 候选支持 Vulkan 1.0/1.1/1.3 与 SPIR-V 1.0/1.3/1.6，实际放行依赖 vkCreateDevice 已启用快照：FP16/int8/int16 storage 与 arithmetic、subgroup、BF16/cooperative/dot、浮点控制、原子作用域、工作组/共享内存/描述符/图像格式分别检查。新 artifact 保存完整 effective_options、target_facts 与 facts hash，并在执行时和真实 typed Option、actual domain 比较；旧包仍可 host inspect，缺执行绑定时不会创建 GPU pipeline。来源 schema 由 ShaderTools 唯一拥有并进入工具实现 fingerprint，runtime compiler 保留。
