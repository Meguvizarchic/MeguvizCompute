# AHB producer fence admission

Android AHB producer dependencies are retained by the resource content, so a
new READ lease for the same content cannot bypass the original import fence.
An accepted session job verifies the fence status without holding the GPU
execution slot. Only a confirmed successful producer can be consumed; a
negative status reports `MGC_PRODUCER_FAILED`, invalidates the affected output
and retains uncertain producer resources. Unsupported or uninspectable status
is an error, not successful data. Native sync-fd previously allowed a Vulkan
acquire command to be submitted before userspace confirmed producer success;
this repair deliberately waits before the actual GPU acquire instead. Its
device latency and throughput have not been qualified.

The additive guarded producer handoff is experimental and OFF by default. Its
build option is recorded in the build fingerprint; capability 19 and its two
entry points fail closed unless that build option and the Android Vulkan/AHB
requirements are met. Registration transfers no caller FD ownership, retains
the exact producer lease, and returns the reservation generation. Consumers
may be accepted while the producer and cleanup are pending, but cannot run
until the fence reports success and the matching generation's cleanup gate is
confirmed. A failed cleanup is reported to accepted consumers while retaining
the producer pin and FD; it is not physical retirement. Unknown outcomes stay
isolated. Never complete a different generation or release backing based on
submission, cancellation or a failed fence alone.

An accepted guarded ticket that is cancelled invalidates its WRITE outputs
before releasing their in-flight protection; callers need a new content
version. In private owned GPU batches every guarded input and every WRITE
output must already be listed in the accepted ticket, otherwise the batch
fails before recording. A successful private retired-WRITE release is an
irrevocable publication point for a guarded ticket: cancellation before
that point prevents transfer, while cancellation afterward returns BUSY until
completion. These added restrictions do not apply to ordinary non-guarded
tickets. The original producer pin remains controlled by its own fence and
cleanup state, independently of ticket cancellation.

In an experimental guarded build, a cleanup completion, reported failure,
cancellation, acceptance or session close notifies a shared session wake
object. Resources hold a small bounded array of weak references to that wake
(two slots). accept() rejects a second ticket while a resource is in flight; a
private batch may also register a requested lease that is not counted in-flight,
so contention beyond the capacity falls back to the bounded watchdog. In the
Android guarded build the array resides in the already-required per-AHB
allocator; ordinary CPU/GPU-buffer resources do not carry it and no additional
per-AHB allocation is introduced. A guarded-build session embeds its wake in
the existing shared session identity: on Android it is an eventfd, where a
notify is a counter increment so a signal racing with poll() cannot be lost,
while the host test build keeps an epoch and condition variable. The wake uses
an aliasing shared pointer and introduces no additional per-session allocation;
that identity allocation is larger and can outlive the session while a ticket
remains alive. Completed tasks unregister their waiter.

The worker probes every accepted ticket without holding the session lock and
executes the first one whose producer fence and cleanup gate are ready. A
ticket whose producer is still pending stays in the worker's pending list (it
does not re-enter s->queue), so an older unready producer no longer blocks a
newer independent ticket: completion order is readiness order, not submission
order. accept() still rejects a second
ticket that references an in-flight resource, which is what makes the reorder
safe. A deferred GPU job still occupies the single worker while it retires.

While no ticket is ready, the worker blocks in poll() on the session eventfd
plus duplicated external fence descriptors, so an unsignaled external fence
wakes it through the kernel instead of a timer. A satisfied fence behind a
still-closed cleanup gate is not polled (that would spin); the gate-opening
call notifies the eventfd. A one-second watchdog covers error paths that have
neither a pollable fd nor a notify (device loss, an unverifiable fence). The
host test build keeps a 5 ms bounded wait because it has no fd to poll. The OS
may schedule a longer delay. Waiting does not hold the GPU slot. No device or
emulator result is implied by host tests or Android cross-compilation.
