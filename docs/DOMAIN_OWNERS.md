# 私有 SDK 域拥有者（`NcnnDomainOwner`）契约

`NcnnDomainOwner` 是 `ncnn` 私有源码 SDK 为独立模型上传提供的常驻所有权句柄，不属于公共 C ABI，也不是异步 ticket 接口。接入该接口的消费方必须使用匹配版本的 `MGC_REQUIRED_PRIVATE_SDK_FINGERPRINT` 重新编译。

## 1. 实例绑定与数量限制

- 每个 `NcnnDomainOwner` 会持续保留其所属的计算 domain、`ncnn` 实例/设备以及当前模块的代码镜像租约。
- 同一物理 `VkDevice` 上最多允许创建 `16` 个 owner。即使通过多个不同的 `MgcRuntime` 创建，只要底层共享同一个 `VkDevice`，它们就会共享任务接收与挂起门控状态。
- 调用方必须持续持有 owner 句柄，直到 `ncnn_domain_owner_release` 返回 `MGC_SUCCESS`。所有针对同一 owner 句柄的操作与销毁动作均需由调用方外部串行化。

## 2. 挂起标记（Pending）与互斥规则

- 调用 `with_ncnn_domain_owner`（或 `try_with_ncnn_domain_owner`）进入回调前，系统会保守地先将设备标记为 pending。
- 只有当回调明确确认其拥有的全部工作均已执行完毕或从未提交过时，才可在同一个 owner 作用域内调用 `ncnn_domain_owner_pending(owner, false)` 清除挂起标记；若回调抛出异常，pending 状态会被自动恢复。仅靠回调返回错误码或成功状态并不足以证明硬件工作已经结束，API 本身也不会主动检查调用方的 fence、命令对象或延迟主机目标。
- 设备处于 pending 状态时，同一设备上的普通作用域、新 GPU 资源创建及相关 Android 提交都会返回 `MGC_BUSY`，`ncnn_domain_owner_release` 同样返回 `MGC_BUSY` 并保持原 owner 对象完整有效。此时只有该 owner 自身可进入其 `try_with_ncnn_domain_owner_retire` 收尾作用域。
- Owner 作用域禁止在活动 ticket 内部执行，也不允许嵌套进入另一个 owner 作用域；活动 ticket 内部的延迟工作由 `DEFERRED_JOBS` 机制单独承接。

## 3. 作用域串行化、清理与资源保活

- GPU 作用域维持既有的共享递归串行顺序；CPU domain 使用独立执行槽位，互不阻塞。跨不同执行槽位的常规嵌套调用会在尝试获取第二把互斥锁之前直接返回 `MGC_INCOMPATIBLE_DOMAIN`。
- 消费方必须在回调内对 fence 使用有限时间等待，并显式推动挂起中的 owner 完成收尾。
- `with_ncnn_domain_cleanup` 清理作用域仅用于销毁已彻底空闲、且由调用方独占的资源，可在普通新任务被阻塞期间运行，但**严禁**在其中提交新 GPU 工作、触发用户回调或回收仍在挂起状态的资源。
- 消费方须自行确保相关联的 `ncnn::Net`、分配器、管线、Shaderpack、模型字节、输入/输出租约及延迟主机目标在对应工作彻底结束前始终存活，仅持有 `MgcDomain` 引用并不能代替这些对象的生命周期管理。

## 4. 动态库卸载安全

Linux 与 Android 平台的动态库消费方必须在与 Compute 相同的镜像中导出 `meguviz_code_image_anchor_v1` 符号（官方组合默认导出）。宿主必须在 `ncnn_domain_owner_release` 成功返回之后才能卸载动态库，模块内部的自身引用无法安全卸载正在执行 release 函数的最后一份代码副本；独立拆分的消费方 `.so` 需要有各自的镜像生命周期约束。

## 适用范围与验证说明

本契约不承诺中断卡死的图形驱动调用、硬实时墙钟超时或自动回收孤儿 owner；同时也不自动修复遗留裸 `VkCompute` 的隐式提交失败、AHB/FD 独立收尾或忽略 `close` 返回 `MGC_BUSY` 的调用逻辑，这些路径需各自接入完整的所有权与收尾协议。
