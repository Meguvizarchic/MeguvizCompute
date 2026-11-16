# Private domain owner contract

The exact-build ncnn source SDK provides `NcnnDomainOwner` for standalone model
uploads. It is not a public C ABI and is not an asynchronous ticket API.
Consumers must rebuild against the matching private SDK fingerprint.

An owner retains its Compute domain, the ncnn instance/device and this module's
code-image lease. At most 16 owners may be created per actual VkDevice. Domains
sharing that device share admission state, even when created by separate
runtimes. A caller must retain the owner handle until release succeeds.

`with_ncnn_domain_owner` conservatively marks the device pending before invoking
the callback. The callback may call `ncnn_domain_owner_pending(owner, false)`
only after establishing that all work it owns is physically retired or was
never submitted. A callback exception restores pending. Merely returning an
error or success does not establish retirement. The API does not inspect a
consumer's fences, command objects or delayed host targets.

While pending, ordinary scopes, new GPU resources and relevant Android
admission return BUSY for the same device. `ncnn_domain_owner_release` returns
BUSY and leaves its complete object valid. Only that owner may enter its
retirement scope. Owner scopes cannot run inside an active ticket or recursively
inside another owner scope; accepted ticket deferral is a separate requirement.

GPU scopes preserve their existing shared recursive serialization order. CPU
domains have separate execution slots. Normal nested entry on a different slot
is rejected before acquiring a second mutex. A consumer must use finite fence
waits and explicitly service pending owners; driver calls are not interrupted
and no wall-clock API deadline or automatic orphan service is promised.

The cleanup scope bypasses admission only for exclusive, already-retired
objects. It must not submit work, call users or reclaim pending resources. The
consumer must retain all associated Net, allocator, pipeline, shaderpack, model
bytes, input/output leases and delayed host objects until their relevant work
retires. A domain reference alone does not establish those lifetimes.

Dynamic consumers on Linux/Android must export the exact
`meguviz_code_image_anchor_v1` symbol from the same image as Compute. The official
composition does so. The host must retain its loader reference through successful
owner release and the API's return. A module-local reference cannot safely
unload the last copy of the code currently executing its release function.
Separate consumer DSOs need their own code-image lifetime agreement.

This contract does not fix legacy VkCompute hidden-submit failures, AHB/FD
retirement, unified device-loss recovery or consumers that ignore close-BUSY.
Those paths require their own complete ownership and retirement integration.
