#ifndef MEGUVIZ_COMPUTE_PRIVATE_JOB_H
#define MEGUVIZ_COMPUTE_PRIVATE_JOB_H
#include <meguviz/compute.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Private source SDK, exact build fingerprint required. Runs on the domain's
 * serial execution slot. The closure owns all other objects until release.
 * The array is copied; each entry transfers one reference only on ACCEPTED.
 * No GPU capability is implied by a host job. */
typedef MgcStatus (*MgcHostJobFn)(void *context, MgcTicket *ticket);
/* Return SUCCESS only after physical retirement, host finalization and internal
 * owned-batch cleanup. Every other result retains the task for another poll.
 * The callback's context is retained by the accepted job, not by this pointer. */
typedef MgcStatus (*MgcHostRetireFn)(void *context, MgcTicket *ticket);
/* The GPU closure explicitly serializes every device operation through owned
 * scopes. CPU domains still keep their serial execution slot. */
enum { MGC_HOST_JOB_EXPLICIT_GPU_SCOPES = 1u };
typedef struct MgcHostJobDesc {
    MgcHeader header;
    uint64_t request_id;
    MgcLease *const *leases;
    uint32_t lease_count;
    uint32_t required_flags;
    MgcHostJobFn execute;
    MgcCompletionFn completion;
    void *context;
    MgcContextFn retain_context;
    MgcContextFn release_context;
    MgcBatchEventFn batch_event;
    MgcHostJobFn resume;
    /* Internal context finalization after retirement and before completion;
     * runs outside internal locks. It must not publish an algorithm result. */
    MgcContextFn finalize;
} MgcHostJobDesc;
MgcStatus mgc_private_submit_host_job(MgcSession *session, const MgcHostJobDesc *desc, MgcTicket **ticket);
/* Only the accepting session may cancel through this entry. A foreign ticket
 * returns INVALID_ARGUMENT without changing it, including after its original
 * session was closed. Both handles must be valid; session close is externally
 * synchronized as with the other session operations. */
MgcStatus mgc_private_session_cancel(MgcSession *session, MgcTicket *ticket);
uint32_t mgc_private_ticket_cancelled(MgcTicket *ticket);
/* Active worker only, one deferred batch per task. No allocation is performed.
 * resume_allowed requires a fully recorded, successful callback. failure_status
 * is used if the task cannot resume; a poll error never proves retirement. */
MgcStatus mgc_private_ticket_defer(MgcTicket *ticket, MgcHostRetireFn poll,
    void *context, uint32_t resume_allowed, MgcStatus failure_status);
uint32_t mgc_private_ticket_is_deferred(MgcTicket *ticket);
/* Active worker only. After all physical/host writes are retired, relinquish
 * one unique internal WRITE submission reference before transferring an owned
 * output to its consumer. Caller keeps a separate WRITE reference. No pending
 * batch, map, alias submission or other in-flight use may remain. Input refs
 * keep their normal through-callback lifetime. */
MgcStatus mgc_private_ticket_release_retired_write(MgcTicket *ticket, MgcLease *lease);
/* Returned domain is retained by the caller, allowing immutable plans to pin
 * it. It is not a borrowed raw VkDevice. */
MgcStatus mgc_private_session_domain(MgcSession *session, MgcDomain **domain);
#ifdef __cplusplus
}
#endif
#endif
