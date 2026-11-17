#ifndef MEGUVIZ_COMPUTE_ANDROID_H
#define MEGUVIZ_COMPUTE_ANDROID_H
#include "compute.h"

#ifdef __cplusplus
extern "C" {
#endif
struct AHardwareBuffer;
enum { MGC_ANDROID_HOST_WAIT_COMPAT = 0, MGC_ANDROID_GPU_SYNC_FD = 1 };
/* Values match Vulkan GENERAL and VK_QUEUE_FAMILY_FOREIGN_EXT. Zero fields
 * select these defaults; callers never need Vulkan headers for the C ABI. */
enum { MGC_ANDROID_LAYOUT_GENERAL = 1 };
typedef struct MgcAndroidImportDesc {
    MgcHeader header;
    struct AHardwareBuffer *buffer;
    int32_t acquire_fence_fd;
    uint32_t required_flags;
    uint64_t allocation_id;
    uint64_t allocation_generation;
    uint64_t content_id;
    uint32_t access;
    uint32_t reserved[3];
    uint32_t layout;
    uint32_t external_queue_family;
    uint32_t sync_mode;
    uint32_t reserved_tail;
    void *content_context;
    MgcContextFn retain_content;
    MgcContextFn release_content;
} MgcAndroidImportDesc;
typedef struct MgcAndroidAllocateDesc {
    MgcHeader header;
    uint32_t width;
    uint32_t height;
    uint64_t usage;
    uint64_t content_id;
    uint32_t access;
    uint32_t sync_mode;
    uint32_t required_flags;
    uint32_t reserved[3];
} MgcAndroidAllocateDesc;
/* Synchronous call borrows caller AHB/fd. Success acquires an independent AHB
 * object reference and duplicates fd; caller retains both originals. Failure
 * unwinds only internal acquisitions. -1 means dependency already satisfied.
 * Object acquisition never grants producer content-reuse permission. */
MGC_API MgcStatus mgc_android_import(MgcDomain *domain, const MgcAndroidImportDesc *desc, MgcLease **lease);
/* Creates an owned RGBA8 AHB (sampled + color-output usage). The returned WRITE
 * content lease pins it until all consumers retire. No producer may rewrite a
 * content version merely because it retained the AHB object. */
MGC_API MgcStatus mgc_android_allocate(MgcDomain *domain, const MgcAndroidAllocateDesc *desc, MgcLease **lease);
/* Buffer is borrowed while lease remains held. An object reference alone does
 * not prove content readiness or grant direct CPU/GPU access: external users
 * must reserve access and honor its ready fence and any producer cleanup gate.
 * Use AHardwareBuffer_acquire only for an independent object reference; it does
 * not grant content reuse. */
MGC_API MgcStatus mgc_android_get_buffer(MgcLease *lease, struct AHardwareBuffer **borrowed_buffer);
MGC_API MgcStatus mgc_android_wait_ready(MgcLease *lease, uint64_t timeout_ns);
/* Ends the same lease's outstanding external reservation. READ registers an
 * external consumer fence; WRITE registers this content's producer fence.
 * Borrows/duplicates fd; -1 acknowledges physical completion/no issued work.
 * A registered READ remains pinned and blocks Compute until wait_ready retires
 * it; a registered WRITE may feed the same published lease's GPU dependency.
 * Failed registration preserves the reservation. No last-fd replacement. */
MGC_API MgcStatus mgc_android_set_acquire_fence(MgcLease *lease, int32_t borrowed_fd);
/* Optional guarded producer handoff. A successful registration borrows the
 * fence, returns the reservation generation and leaves an independent cleanup
 * gate closed. Submissions may be accepted but cannot execute until BOTH the
 * sync-file is verified successful and finish_guarded confirms host cleanup.
 * The original lease and its owning code/context must survive until finish.
 * A stale generation is rejected. On unknown cleanup do not finish: preserve
 * the resource and reservation. This additive API is usable only when
 * MGC_CAP_GUARDED_PRODUCER_FENCE reports enabled on the same domain. */
MGC_API MgcStatus mgc_android_set_guarded_producer_fence(MgcLease *lease, int32_t borrowed_fd,
    uint64_t *generation);
/* cleanup_status MGC_SUCCESS confirms successful cleanup. MGC_PRODUCER_FAILED
 * reports failed or uncertain cleanup to queued consumers, while retaining
 * the producer reservation and pin: it is not proof of physical retirement. */
MGC_API MgcStatus mgc_android_finish_guarded_producer(MgcLease *lease, uint64_t generation,
    MgcStatus cleanup_status);
/* Begins one bounded external access and retains the content lease. Success
 * transfers a ready fd and GENERAL/FOREIGN ownership. Every success must end
 * with set_acquire_fence on the same lease, even when no GL work was issued.
 * FOREIGN idle accesses use only a resource try-lock; active content, existing
 * external access or a busy device-transition slot returns BUSY. Other actual
 * readers are serialized instead of replacing their pending fence. */
MGC_API MgcStatus mgc_android_release_to_external(MgcLease *lease, int32_t *owned_fd);
/* Alias of release_to_external, including its begin/end reservation contract.
 * Success transfers a new owned fd; -1 means dependency already satisfied. */
MGC_API MgcStatus mgc_android_export_release_fence(MgcLease *lease, int32_t *owned_fd);
#ifdef __cplusplus
}
#endif
#endif
