#ifndef MEGUVIZ_COMPUTE_H
#define MEGUVIZ_COMPUTE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define MGC_API __attribute__((visibility("default")))
#else
#define MGC_API
#endif
#define MGC_ABI_MAJOR 0u
#define MGC_ABI_MINOR 1u
#define MGC_HEADER_INIT(type) { (uint32_t)sizeof(type), MGC_ABI_MAJOR, MGC_ABI_MINOR }

typedef int32_t MgcStatus;
enum {
    MGC_SUCCESS = 0, MGC_ACCEPTED = 1,
    MGC_INVALID_ARGUMENT = -1, MGC_UNSUPPORTED_ABI = -2,
    MGC_UNSUPPORTED_CAPABILITY = -3, MGC_INCOMPATIBLE_DOMAIN = -4,
    MGC_MODEL_MISMATCH = -5, MGC_ARTIFACT_MISMATCH = -6,
    MGC_BUSY = -7, MGC_BACKPRESSURE = -8, MGC_DEADLINE_REJECTED = -9,
    MGC_OUT_OF_MEMORY = -10, MGC_DEVICE_LOST = -11, MGC_TIMEOUT = -12,
    MGC_CANCELLED = -13, MGC_INTERNAL_ERROR = -14,
    MGC_PRODUCER_FAILED = -15
};

typedef struct MgcHeader {
    uint32_t struct_size;
    uint16_t abi_major;
    uint16_t abi_minor;
} MgcHeader;

typedef struct MgcHash { uint8_t bytes[32]; } MgcHash;
typedef struct MgcRuntime MgcRuntime;
typedef struct MgcDomain MgcDomain;
typedef struct MgcResource MgcResource;
typedef struct MgcLease MgcLease;
typedef struct MgcSession MgcSession;
typedef struct MgcTicket MgcTicket;
/* User-provided callbacks must not throw exceptions across this C ABI.
 * Retain/release callbacks run without the session lock; a close during submit
 * preparation returns BUSY. If retain throws after partial user-side work,
 * the library cannot infer or undo that work. */
typedef void (*MgcContextFn)(void *context);

enum { MGC_PROVIDER_CPU_REFERENCE = 1, MGC_PROVIDER_NCNN_CPU = 2, MGC_PROVIDER_NCNN_VULKAN = 3 };
enum {
    MGC_CAP_CPU_REFERENCE = 1, MGC_CAP_VULKAN_COMPUTE = 2, MGC_CAP_AHB_IMPORT = 3,
    MGC_CAP_EGL_AHB_RENDERABLE = 4, MGC_CAP_EGL_AHB_SAMPLEABLE = 5,
    MGC_CAP_VULKAN_AHB_SAMPLEABLE = 6, MGC_CAP_VULKAN_AHB_STORAGE = 7,
    MGC_CAP_EXTERNAL_YCBCR = 8, MGC_CAP_SYNC_FD_IMPORT = 9, MGC_CAP_SYNC_FD_EXPORT = 10,
    MGC_CAP_HOST_JOB_ASYNC = 11, MGC_CAP_FINAL_SUBMIT_ASYNC = 12,
    MGC_CAP_FULL_RECORD_ASYNC = 13, MGC_CAP_GPU_TIMESTAMP = 14,
    MGC_CAP_FP16_STORAGE = 15, MGC_CAP_FP16_ARITHMETIC = 16,
    MGC_CAP_APPROVED_SHADER_PROFILE = 17, MGC_CAP_EXTERNAL_HOST_MEMORY = 18,
    MGC_CAP_GUARDED_PRODUCER_FENCE = 19
};
enum { MGC_ACCESS_READ = 1, MGC_ACCESS_WRITE = 2 };
/* usage=0 is explicit host storage. GPU allocations are never CPU mapped. */
enum { MGC_RESOURCE_GPU_BUFFER = 1, MGC_RESOURCE_ANDROID_AHB = 2 };
enum { MGC_FORMAT_BYTES = 1, MGC_FORMAT_RGB8 = 2, MGC_FORMAT_RGBA8 = 3, MGC_FORMAT_FLOAT32 = 4 };
enum { MGC_OPERATION_COPY = 1, MGC_OPERATION_NORMALIZE_F32 = 2, MGC_OPERATION_RESIZE_RGB8 = 3, MGC_OPERATION_STATISTICS_F32 = 4, MGC_OPERATION_RELU_F32 = 5 };
enum { MGC_DEPENDENCY_READY = 0, MGC_DEPENDENCY_HOST_WAIT = 1, MGC_DEPENDENCY_SYNC_FD = 2 };
enum { MGC_TERMINAL_SUCCESS = 1, MGC_TERMINAL_FAILED = 2, MGC_TERMINAL_CANCELLED = 3, MGC_TERMINAL_DEVICE_LOST = 4 };
enum { MGC_EVENT_ACCEPTED = 1, MGC_EVENT_GPU_SUBMITTED = 2, MGC_EVENT_GPU_DEPENDENCY_READY = 3, MGC_EVENT_EXECUTION_RETIRED = 4, MGC_EVENT_OUTPUT_RELEASED = 5 };
enum { MGC_MEASURE_GPU_TIMESTAMPS = 1 };
typedef struct MgcBatchEvent {
    MgcHeader header;
    uint64_t request_id, batch_id, monotonic_ns;
    uint32_t kind, reserved;
} MgcBatchEvent;
typedef void (*MgcBatchEventFn)(void *context, const MgcBatchEvent *event);

typedef struct MgcRuntimeDesc {
    MgcHeader header;
    uint32_t provider;
    uint32_t required_flags;
    /* Admission budget for registered Resource ranges, including allocator
     * alignment (AHB conservatively uses max(row extent, allocationSize)).
     * Private ncnn caches, commands/queries and driver memory are separate. */
    uint64_t memory_budget_bytes;
    uint32_t reserved[4];
    uint32_t measurement_flags;
    uint32_t measurement_reserved;
} MgcRuntimeDesc;

typedef struct MgcDomainDesc {
    MgcHeader header;
    uint32_t device_index;
    uint32_t required_flags;
    MgcHash private_sdk_fingerprint;
    uint32_t reserved[4];
} MgcDomainDesc;

typedef struct MgcResourceDesc {
    MgcHeader header;
    uint64_t allocation_id;
    uint64_t allocation_generation;
    uint64_t capacity_bytes;
    uint32_t format;
    uint32_t usage;
    uint32_t width;
    uint32_t height;
    uint64_t row_stride_bytes;
    uint32_t required_flags;
    uint32_t reserved[3];
} MgcResourceDesc;

/* Optional externally owned CPU backing. The caller keeps its reference until
 * this call returns. Compute retains context once on admission and releases it
 * after the final resource/lease/task reference. The callback functions must
 * be thread-safe, no-throw and safe to call outside Compute locks. Data must
 * remain stable for that lifetime. READ forbids creation of WRITE leases. */
typedef struct MgcHostMemoryDesc {
    MgcHeader header;
    void *data;
    uint64_t capacity_bytes;
    uint32_t access;
    uint32_t required_flags;
    uint32_t reserved[4];
    void *context;
    MgcContextFn retain;
    MgcContextFn release;
} MgcHostMemoryDesc;

typedef struct MgcLeaseDesc {
    MgcHeader header;
    uint64_t content_id;
    uint64_t byte_offset;
    uint64_t byte_length;
    uint32_t access;
    uint32_t dependency_kind;
    uint64_t dependency_id;
    uint32_t required_flags;
    uint32_t reserved[3];
    /* Content lifetime, independent of a cached allocation reference. Retain
     * once on successful creation; release after this lease's final reference.
     * publish_read preserves the same content owner and reference accounting. */
    void *content_context;
    MgcContextFn retain_content;
    MgcContextFn release_content;
} MgcLeaseDesc;

typedef struct MgcImageView {
    MgcHeader header;
    MgcLease *lease;
    uint64_t byte_offset;
    uint64_t row_stride_bytes;
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint32_t plane;
    uint32_t layout;
    uint32_t required_flags;
    uint32_t reserved[2];
} MgcImageView;

typedef struct MgcLeaseInfo {
    MgcHeader header;
    uint64_t domain_id;
    uint64_t domain_generation;
    uint64_t allocation_id;
    uint64_t allocation_generation;
    uint64_t content_id;
    uint64_t capacity_bytes;
    uint32_t access;
    uint32_t dependency_kind;
    uint32_t reserved[4];
    uint32_t format;
    uint32_t width;
    uint32_t height;
    uint32_t reserved_tail;
    uint64_t row_stride_bytes;
    uint64_t byte_offset;
    uint64_t byte_length;
} MgcLeaseInfo;

typedef struct MgcCapability {
    MgcHeader header;
    uint32_t feature;
    uint32_t advertised;
    uint32_t enabled;
    int32_t probe_result;
    uint32_t workload_qualified;
    int32_t reason;
    uint32_t reserved[4];
} MgcCapability;
/* A lost domain is never rebound to a replacement VkDevice. After service has
 * retired/detached its old device, acquire a NEW domain/model/plan. Old handles
 * remain valid only for diagnostics and origin-owned release. */
typedef struct MgcDomainStateInfo {
    MgcHeader header;
    uint64_t domain_id, domain_generation, device_epoch;
    uint32_t lost, device_retired, detached;
    int32_t last_error;
    uint32_t reserved[4];
} MgcDomainStateInfo;

typedef struct MgcCompletion {
    MgcHeader header;
    uint64_t request_id;
    uint32_t terminal;
    int32_t status;
    uint32_t execution_retired;
    uint32_t host_finalized;
    uint32_t result_discarded;
    uint32_t reserved[3];
    /* Actual worker thread CPU between job execution entry and host finalize;
     * excludes user callback. observed=0 means unavailable, not zero CPU use. */
    uint64_t worker_cpu_ns;
    uint32_t worker_cpu_observed;
    uint32_t worker_cpu_reserved;
    uint64_t gpu_elapsed_ns;
    uint64_t gpu_batch_count;
    uint64_t gpu_observed_batch_count;
    uint32_t gpu_elapsed_observed;
    uint32_t gpu_timestamp_valid_bits;
    uint32_t gpu_timestamp_period_bits;
    uint32_t batch_event_coverage_observed;
} MgcCompletion;

typedef void (*MgcCompletionFn)(void *context, const MgcCompletion *completion);

typedef struct MgcSessionDesc {
    MgcHeader header;
    uint32_t max_in_flight;
    uint32_t required_flags;
    uint32_t reserved[4];
} MgcSessionDesc;

/* submit copies this descriptor. Each non-null lease is one distinct submission
 * reference: ACCEPTED transfers it, rejection leaves it with the caller.
 * callback context retain/release must be supplied as a pair (or both null).
 * Accepted context is retained before submit returns; release occurs after the
 * final callback returns. Completion runs on the session worker, never inline
 * in submit and never under its lock. The callback may query/cancel or submit;
 * drain/close from that same callback returns BUSY. In an experimental guarded
 * producer build the worker executes the first ready ticket, so completion
 * order is readiness order rather than submission order; tickets that share a
 * resource are still serialized by acceptance. */
typedef struct MgcSubmitDesc {
    MgcHeader header;
    uint64_t request_id;
    uint32_t operation;
    uint32_t required_flags;
    MgcLease *input;
    MgcLease *output;
    uint64_t byte_count;
    MgcCompletionFn completion;
    void *context;
    MgcContextFn retain_context;
    MgcContextFn release_context;
    uint32_t reserved[4];
    /* Bounded recorded events dispatch after execution-slot exit, before the
     * final callback; same retained context. Overflow clears coverage flag. */
    MgcBatchEventFn batch_event;
} MgcSubmitDesc;
typedef struct MgcAllocationStats {
    MgcHeader header;
    uint64_t domain_id;
    uint64_t managed_live_bytes, managed_highwater_bytes, managed_live_count, managed_highwater_count;
    uint64_t imported_live_bytes, imported_highwater_bytes, imported_live_count, imported_highwater_count;
    uint64_t total_allocation_count;
    uint32_t resource_ranges_observed;
    uint32_t private_reserved_observed;
    uint64_t private_reserved_bytes;
    uint32_t private_live_observed;
    uint32_t reserved;
    uint64_t private_live_bytes;
} MgcAllocationStats;

typedef struct MgcBuildInfo {
    MgcHeader header;
    uint32_t provider;
    uint32_t reserved;
    MgcHash private_sdk_fingerprint;
    MgcHash build_fingerprint;
} MgcBuildInfo;
typedef struct MgcExecutionInfo {
    MgcHeader header;
    uint64_t submitted_batches;
    uint64_t retired_batches;
    uint64_t finalized_batches;
    uint64_t recording_submitted_batches;
    uint32_t final_submit_async;
    uint32_t full_record_async;
    uint32_t reserved[4];
} MgcExecutionInfo;

MGC_API MgcStatus mgc_query_build_info(MgcBuildInfo *info);
MGC_API MgcStatus mgc_runtime_create(const MgcRuntimeDesc *desc, MgcRuntime **runtime);
MGC_API void mgc_runtime_retain(MgcRuntime *runtime);
MGC_API void mgc_runtime_release(MgcRuntime *runtime);
/* close returns BUSY while domains/tasks/consumer outputs remain live. */
MGC_API MgcStatus mgc_runtime_close(MgcRuntime *runtime);
MGC_API MgcStatus mgc_domain_acquire(MgcRuntime *runtime, const MgcDomainDesc *desc, MgcDomain **domain);
MGC_API MgcStatus mgc_domain_identity(MgcDomain *domain, uint64_t *domain_id, uint64_t *generation);
MGC_API MgcStatus mgc_domain_state(MgcDomain *domain, MgcDomainStateInfo *state);
/* Worker service; NULL explicitly visits this module's standalone GPU owners.
 * A zero wait uses zero-time fence queries and try-locks where possible;
 * driver and origin-cleanup calls may still block.
 * SUCCESS is not an assertion that caller-owned sessions/outputs are closed. */
MGC_API MgcStatus mgc_domain_service(MgcDomain *domain, uint64_t wait_ns);
/* Explicit opt-in, sampled per accepted ticket. Default zero emits no GPU
 * timestamp commands. Period is raw IEEE754 float bits, not host wall time. */
MGC_API MgcStatus mgc_domain_set_measurement_flags(MgcDomain *domain, uint32_t flags);
/* Resource allocation ranges only; private ncnn live/reserved, command/query
 * and driver allocations stay unknown. Counts are not total GPU/DRAM usage.
 * Imported includes AHB Vulkan imports, even when the AHB was allocated here. */
MGC_API MgcStatus mgc_domain_allocation_stats(MgcDomain *domain, MgcAllocationStats *stats);
MGC_API MgcStatus mgc_runtime_allocation_stats(MgcRuntime *runtime, MgcAllocationStats *stats);
MGC_API void mgc_domain_retain(MgcDomain *domain);
MGC_API void mgc_domain_release(MgcDomain *domain);
MGC_API MgcStatus mgc_domain_capability(MgcDomain *domain, uint32_t feature, MgcCapability *capability);
MGC_API MgcStatus mgc_resource_create(MgcDomain *domain, const MgcResourceDesc *desc, MgcResource **resource);
MGC_API MgcStatus mgc_resource_import_host_v1(MgcDomain *domain, const MgcResourceDesc *desc,
    const MgcHostMemoryDesc *backing, MgcResource **resource);
MGC_API void mgc_resource_retain(MgcResource *resource);
MGC_API void mgc_resource_release(MgcResource *resource);
MGC_API MgcStatus mgc_lease_create(MgcResource *resource, const MgcLeaseDesc *desc, MgcLease **lease);
MGC_API void mgc_lease_retain(MgcLease *lease);
MGC_API void mgc_lease_release(MgcLease *lease);
MGC_API MgcStatus mgc_lease_info(MgcLease *lease, MgcLeaseInfo *info);
/* Returns one retained origin resource reference; no lease reference moves.
 * Reuse still requires a successful new WRITE lease and a newer content_id. */
MGC_API MgcStatus mgc_lease_resource(MgcLease *lease, MgcResource **retained_resource);
/* Explicit CPU mapping is unavailable for protected or GPU-only resources.
 * A successful map is borrowed until unmap; never free() its pointer. */
MGC_API MgcStatus mgc_lease_map(MgcLease *lease, void **data, uint64_t *size_bytes);
MGC_API MgcStatus mgc_lease_unmap(MgcLease *lease);
/* Waits for the lease's producer dependency; timeout never consumes it. */
MGC_API MgcStatus mgc_lease_wait_ready(MgcLease *lease, uint64_t timeout_ns);
/* Seals a completed WRITE lease. All references to this lease become READ;
 * returns one additional owned reference. The caller still releases its old
 * reference. No maps or foreign in-flight submissions may exist. */
MGC_API MgcStatus mgc_lease_publish_read(MgcLease *writing_lease, MgcLease **read_lease);
MGC_API MgcStatus mgc_session_create(MgcDomain *domain, const MgcSessionDesc *desc, MgcSession **session);
MGC_API MgcStatus mgc_session_close(MgcSession *session);
MGC_API MgcStatus mgc_submit(MgcSession *session, const MgcSubmitDesc *desc, MgcTicket **ticket);
MGC_API void mgc_ticket_retain(MgcTicket *ticket);
MGC_API void mgc_ticket_release(MgcTicket *ticket);
/* Cancellation requested before completion discards the result. A confirmed
 * MGC_DEVICE_LOST or MGC_PRODUCER_FAILED remains the completion status even
 * when cancellation is concurrent. For an experimental guarded producer job,
 * once a private retired-WRITE release has transferred an output, cancellation
 * before completion returns MGC_BUSY: that output cannot be retracted. */
MGC_API MgcStatus mgc_ticket_cancel(MgcTicket *ticket);
MGC_API MgcStatus mgc_ticket_query(MgcTicket *ticket, MgcCompletion *completion);
MGC_API MgcStatus mgc_ticket_execution_info(MgcTicket *ticket, MgcExecutionInfo *info);
/* timeout_ns=0 polls. Timeout preserves every live object. drain waits for all
 * accepted execution, host finalize, callbacks and borrowed submission leases;
 * consumer-owned output leases are outside its scope and stay valid. */
MGC_API MgcStatus mgc_session_drain(MgcSession *session, uint64_t timeout_ns);

#ifdef __cplusplus
}
#endif
#endif
