# Deferred accepted jobs in the private source SDK

Private SDK 0.2.6 retains the 0.2.4 `MgcHostJobDesc` `resume` and `finalize` tail
callbacks. Old complete descriptor prefixes default these callbacks to null;
partial pointer fields and unknown required flags are rejected. The public C
submit/completion structures keep their existing ownership contract.

`mgc_private_ticket_defer` registers one retirement owner for the active job.
Until its poll returns SUCCESS, the worker keeps the accepted task, session
in-flight count, submission leases and context references. Cancellation marks
the result discarded but cannot skip physical retirement, run the final
callback early or make the polling backoff spin. Session drain/close still
report TIMEOUT/BUSY while that ownership remains.

Only a fully recorded successful batch may resume the algorithm. A callback
that stopped inside graph extraction is a failed operation even if its earlier
GPU batch later retires. It is never replayed or treated as a complete graph.
Final internal context cleanup precedes the one completion callback; both
context finalization and user completion execute outside worker device locks.

An internal output WRITE reference may be transferred out of the ticket after
all writes retire. `mgc_private_ticket_release_retired_write` requires the active
ticket, a unique unmapped WRITE submission, no pending batch and a separate
caller-owned writer reference. It removes only that submission reference and
its in-flight count. The caller then publishes an owned READ result. Input
references keep their normal through-callback lifetime; consumer release can
therefore make a one-slot output pool reusable during the callback.

GPU jobs using `with_ncnn_gpu_owned` set
`MGC_HOST_JOB_EXPLICIT_GPU_SCOPES`. Every device operation is serialized by its
own explicit scope; the worker does not wrap the whole algorithm in the GPU
mutex. CPU-domain jobs keep their own execution slot. This permits retirement
backoff without holding the GPU slot and keeps unrelated CPU work progressing.

A GPU session acquires a verified code-image reference at creation. An owned
batch borrows it only when its active ticket, session identity, worker and
domain all match; standalone and synthetic scopes retain their own references.
The session keeps its reference through accepted work, retirement and callback
completion. `mgc_session_close` joins the worker before releasing it; if release
fails, the closed session is retained for retry. The caller must keep a separate
host loader reference through the successful close call and its return.

The owned scope reserves its context before recording: heap command, retained
algorithm state, all lease/view references, and Android acquire/release
transitions. A normal initial wait is finite; an unresolved operation continues
through zero-wait polls. A producer-readiness wait with no GPU work in use keeps
its resource leases without quarantining an idle device. Possibly submitted
work keeps the shared-device gate until physical confirmation.

AHB transitions retain their own pools, fences, semaphores and FDs. They do not
repeat accepted submissions. Failed, definitely unsubmitted release attempts
may retry their metadata object; logical failure remains recorded. Resource
ownership is returned only after the actual transition and metadata application
are confirmed. Producer references are dropped outside GPU scopes.

The managed ncnn command retains its buffer/image/host references and records
CPU access/layout snapshots before modifying them. A definitely unsubmitted
tail can be discarded with snapshot rollback; pending/uncertain submissions
cannot. Host finalization caches failure so partial host copies are not replayed.
Consumers still own Net, pipeline, allocator, model and external host-target
lifetimes through the supplied strong context.

Completed GPU cleanup and clearing this owner's pending state precede DONE.
Metadata-only owner release retries bypass ordinary record admission, so another
owner cannot block an already-retired task's callback through that gate.

The generic GPU COPY, ReLU, normalize, statistics and supported RGB resize
operations use owned, staged work; intermediate network transfers are retained
with the same job. RGB resize uses binary64 shaders only when enabled by the
logical device. Otherwise host-to-host resize uses the CPU reference and GPU-only
or AHB storage returns unsupported. Shader rounding remains device-dependent and
must be validated for each workload. Standalone capability and AHB-transition
work uses bounded retained jobs.
The older private synchronous scope waits for retirement before returning, so its
borrowed caller context is never released while its command remains pending.

Device loss is shared by domains on the same logical device. It stops admission
and invalidates failed output, but does not by itself prove that retained work
has retired. Service calls keep the old generation until an actual device-idle
result establishes retirement; only then may a new domain use a fresh logical
device. Old domains remain usable for diagnostics and origin-owned cleanup, not
new GPU operations. `mgc_domain_state` reports the state and
`mgc_domain_service` advances retirement; callers must also keep external AHB,
producer and GL ownership until their independent obligations complete.
Capability records distinguish a successful retired submission from workload
qualification, and report device loss for Vulkan-dependent features after loss.

These rules do not make arbitrary raw command consumers safe automatically.
Neither a hard driver-call wall-clock deadline nor Android hardware, complete
workload or performance qualification is implied by these source contracts.
