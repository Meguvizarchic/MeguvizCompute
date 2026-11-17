/* Included in the implementation namespace, after opaque object definitions. */
#if defined(__ANDROID__)
MgcStatus wait_fd(int fd, uint64_t timeout_ns)
{
    if (fd == -1) return MGC_SUCCESS;
    if (fd < -1) return MGC_INVALID_ARGUMENT;
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        int timeout_ms = -1;
        if (timeout_ns != UINT64_MAX) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            if (timeout_ns && uint64_t(elapsed) >= timeout_ns) return MGC_TIMEOUT;
            const uint64_t left = timeout_ns ? timeout_ns - uint64_t(elapsed) : 0;
            const uint64_t ms = left / 1000000 + (left % 1000000 != 0);
            timeout_ms = int(std::min<uint64_t>(ms, INT32_MAX));
        }
        const int status = meguviz_compute_detail::sync_file_wait(fd, timeout_ms);
        if (status == 0) return MGC_SUCCESS;
        if (status == -2) return MGC_PRODUCER_FAILED;
        if (status == 2) return MGC_BUSY;
        if (status != 1) return MGC_INTERNAL_ERROR;
        if (!timeout_ns || timeout_ns == UINT64_MAX) return MGC_TIMEOUT;
        // Continue if the platform capped a much longer requested timeout.
    }
}
#endif
struct ScopedLeaseCall {
    MgcLease *lease;
    explicit ScopedLeaseCall(MgcLease *value):lease(value) { mgc_lease_retain(lease); }
    ~ScopedLeaseCall() { mgc_lease_release(lease); }
};
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
MgcLease *detach_external_status_locked(MgcResource *r, MgcStatus ready)
{
    if (r->external_access_state != 2 && r->external_access_state != 3) return nullptr;
    if (ready == MGC_PRODUCER_FAILED && r->external_access_state == 3) {
        r->content_invalid = true; r->content_failure = MGC_PRODUCER_FAILED;
        r->producer_guard = 3;
    }
    // A failed fence is not sufficient proof of physical retirement. Retain
    // the exact fd, lease pin and backing for both readers and producers.
    if (ready != MGC_SUCCESS) return nullptr;
    if (r->producer_guard == 1 || r->producer_guard == 3) return nullptr;
    if (r->external_access_fd >= 0) close(r->external_access_fd);
    r->external_access_fd = -1;
    if (r->external_access_state == 3 && r->external_access_lease && r->external_access_lease->acquire_fd >= 0) {
        close(r->external_access_lease->acquire_fd);
        r->external_access_lease->acquire_fd = -1;
        r->external_access_lease->desc.dependency_kind = MGC_DEPENDENCY_READY;
    }
    r->external_access_state = 0;
    r->producer_guard = 0;
    auto *pin = r->external_access_lease;
    r->external_access_lease = nullptr;
    return pin;
}
MgcLease *detach_external_ready_locked(MgcResource *r)
{
    if (r->external_access_state != 2 && r->external_access_state != 3) return nullptr;
    return detach_external_status_locked(r, wait_fd(r->external_access_fd, 0));
}
void reap_external_ready(MgcResource *r)
{
    int copy = -1;
    uint64_t generation = 0;
    uint32_t state = 0;
    MgcLease *pin = nullptr;
    {
        std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
        if (!lock.owns_lock() || (r->external_access_state != 2 && r->external_access_state != 3)) return;
        // A closed or quarantined guarded cleanup gate cannot be detached;
        // avoid caller-side fd syscalls. The worker independently probes the
        // producer so a confirmed error is still propagated promptly.
        if (r->producer_guard == 1 || r->producer_guard == 3) return;
        state = r->external_access_state; generation = r->external_access_generation;
        if (r->external_access_fd >= 0) {
            copy = dup(r->external_access_fd);
            if (copy < 0) return;
        }
    }
    const auto status = wait_fd(copy, 0);
    if (copy >= 0) close(copy);
    {
        std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
        if (lock.owns_lock() && r->external_access_generation == generation && r->external_access_state == state)
            pin = detach_external_status_locked(r, status);
    }
    if (pin) mgc_lease_release(pin);
}
MgcStatus reserve_external_locked(MgcLease *lease, int32_t *out)
{
    auto *r = lease->resource;
    if (!r->ahb) return MGC_UNSUPPORTED_CAPABILITY;
    if (r->content_invalid) return r->content_failure;
    if (r->maps || r->in_flight || r->ownership_uncertain || r->external_transition || r->external_access_state || !r->externally_owned)
        return MGC_BUSY;
    if (lease->acquire_fd >= 0)
    {
        auto ready = wait_fd(lease->acquire_fd, 0);
        if (ready) return ready == MGC_TIMEOUT ? MGC_BUSY : ready;
        close(lease->acquire_fd); lease->acquire_fd = -1;
        lease->desc.dependency_kind = MGC_DEPENDENCY_READY;
    }
    if (r->external_access_generation == UINT64_MAX) return MGC_BACKPRESSURE;
    int fd = r->release_fd < 0 ? -1 : dup(r->release_fd);
    if (r->release_fd >= 0 && fd < 0) return MGC_INTERNAL_ERROR;
    mgc_lease_retain(lease);
    r->external_access_lease = lease;
    r->external_access_state = 1;
    ++r->external_access_generation;
    *out = fd;
    return MGC_SUCCESS;
}
#endif
MgcStatus wait_ready(MgcLease *l, uint64_t timeout_ns)
{
    if (!l) return MGC_INVALID_ARGUMENT;
#if defined(__ANDROID__)
    ScopedLeaseCall call(l);
    const auto start = std::chrono::steady_clock::now();
    auto *r = l->resource;
    for (unsigned attempt = 0; attempt < 8; ++attempt) {
        int copy = -1, original = -1;
        bool external = false;
        uint64_t generation = 0;
        uint32_t access_state = 0, guard = 0;
        {
            std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
            if (!lock.owns_lock()) return MGC_BUSY;
            if (r->content_invalid) return r->content_failure;
#if MGC_HAVE_NCNN && NCNN_VULKAN
            if (r->external_transition || r->external_access_state == 1) return MGC_BUSY;
            if (r->external_access_state == 2 || r->external_access_state == 3) {
                external = true; original = r->external_access_fd;
                generation = r->external_access_generation; access_state = r->external_access_state;
                guard = r->producer_guard;
            } else
#endif
            {
                original = l->acquire_fd;
#if MGC_HAVE_NCNN && NCNN_VULKAN
                generation = r->external_access_generation;
#endif
            }
            if (original < 0 && !external) return MGC_SUCCESS;
            if (original >= 0) { copy = dup(original); if (copy < 0) return MGC_INTERNAL_ERROR; }
        }
        uint64_t remaining = timeout_ns;
        if (timeout_ns != UINT64_MAX && timeout_ns) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            if (uint64_t(elapsed) >= timeout_ns) { if (copy >= 0) close(copy); return MGC_TIMEOUT; }
            remaining -= uint64_t(elapsed);
        }
        MgcStatus status;
#if MGC_HAVE_NCNN && NCNN_VULKAN
        if (external && guard && remaining == 0) {
            const int probed = meguviz_compute_detail::sync_file_probe(copy);
            status = probed == 0 ? MGC_SUCCESS : probed == 1 ? MGC_TIMEOUT :
                probed == 2 ? MGC_BUSY : probed == -2 ? MGC_PRODUCER_FAILED : MGC_INTERNAL_ERROR;
        } else
#endif
            status = wait_fd(copy, remaining);
        if (copy >= 0) close(copy);
        if (status != MGC_SUCCESS && status != MGC_PRODUCER_FAILED) return status;
        MgcLease *pin = nullptr;
        bool guard_pending = false, stale = false;
        {
            std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
            if (!lock.owns_lock()) return MGC_BUSY;
#if MGC_HAVE_NCNN && NCNN_VULKAN
            if (external) {
                if (r->external_access_state != access_state || r->external_access_generation != generation) stale = true;
                else {
                    if (status == MGC_PRODUCER_FAILED && access_state == 3) {
                        r->content_invalid = true; r->content_failure = MGC_PRODUCER_FAILED;
                    }
                    guard_pending = meguviz_compute_detail::producer_gate_status(
                        r->producer_guard, r->content_invalid, r->content_failure) != MGC_SUCCESS;
                    pin = detach_external_status_locked(r, status);
                }
            } else
#endif
            if (l->acquire_fd == original
#if MGC_HAVE_NCNN && NCNN_VULKAN
                && r->external_access_generation == generation
#endif
            ) {
                if (status == MGC_PRODUCER_FAILED) {
                    r->content_invalid = true; r->content_failure = status;
                    // Preserve a legacy lease-only FD on failure. The owning
                    // lease remains caller-managed; new AHB imports use the
                    // resource-level producer pin instead.
                } else {
                    if (l->acquire_fd >= 0) close(l->acquire_fd);
                    l->acquire_fd = -1; l->desc.dependency_kind = MGC_DEPENDENCY_READY;
                }
            } else stale = true;
        }
        if (pin) mgc_lease_release(pin);
        if (stale) continue;
        if (status == MGC_PRODUCER_FAILED) return status;
        if (guard_pending) return MGC_BUSY;
        return MGC_SUCCESS;
    }
    return MGC_BUSY;
#else
    (void)timeout_ns;
    return l->desc.dependency_kind == MGC_DEPENDENCY_READY ? MGC_SUCCESS : MGC_UNSUPPORTED_CAPABILITY;
#endif
}
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
// Return a duplicated descriptor that poll() can use to wait for this lease's
// external producer dependency: >= 0 is an active fence, -1 means the fence is
// already satisfied so only an internal wake can make progress, and -2 means
// the dependency cannot be inspected and a bounded backstop is required. A
// satisfied sync-file behind a still-closed cleanup gate must not be polled:
// poll would return immediately and spin. The gate is opened by an internal
// call that notifies the session wake instead.
int dup_producer_poll_fd(MgcLease *l) noexcept
{
    if (!l) return -1;
    auto *r = l->resource;
    int copy = -1;
    {
        // Dup while holding the resource lock: another thread may close and
        // reuse the descriptor number between an unlocked read and dup().
        std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
        if (!lock.owns_lock()) return -2;
        if (r->content_invalid) return -1;
        const int original = (r->external_access_state == 2 || r->external_access_state == 3)
            ? r->external_access_fd : l->acquire_fd;
        if (original < 0) return -1;
        copy = dup(original);
        if (copy < 0) return -2;
    }
    // sync_file_probe: 1 active, 0 satisfied, negative failed/unverifiable.
    const int status = meguviz_compute_detail::sync_file_probe(copy);
    if (status != 1) { close(copy); return status == 0 ? -1 : -2; }
    return copy;
}
#else
inline int dup_producer_poll_fd(MgcLease *) noexcept { return -1; }
#endif
#if MGC_HAVE_NCNN && NCNN_VULKAN
void gpu_batch_observer(void *context, uint64_t, uint32_t event)
{
    auto *t = static_cast<MgcTicket *>(context); if (!t) return;
    if (event == 1) { uint64_t id = ++t->submitted_batches; if (!t->final_submitting) ++t->recording_batches; if (++t->gpu_pending > 1) { t->gpu_complete = false; t->event_complete = false; } record_event(t, MGC_EVENT_GPU_SUBMITTED, id); }
    if (event == 2) { ++t->retired_batches; if (t->gpu_pending) --t->gpu_pending; record_event(t, MGC_EVENT_GPU_DEPENDENCY_READY, t->submitted_batches.load()); }
    if (event == 3) ++t->finalized_batches;
}
void gpu_timestamp_observer(void *context, uint64_t, uint64_t ns, uint32_t observed, uint32_t bits, uint32_t period)
{ observe_gpu(static_cast<MgcTicket *>(context), ns, observed, bits, period); }
void configure_gpu_measurement(ncnn::VkCompute &command, MgcDomain *domain, MgcTicket *ticket)
{
    const uint32_t flags = ticket ? ticket->measurement_flags : domain->measurement_flags.load();
    if (flags & MGC_MEASURE_GPU_TIMESTAMPS) {
        command.set_gpu_timestamp_observer(gpu_timestamp_observer, ticket);
        command.enable_gpu_timestamps(true); // unsupported leaves every batch unobserved
    }
}
MgcStatus finish_gpu_command(ncnn::VkCompute &command, MgcTicket *ticket)
{
    if (ticket) ticket->final_submitting = true;
    int submitted = command.submit_async();
    if (ticket) { ticket->final_submitting = false; if (!submitted) ticket->used_final_async = true; }
    if (submitted) return MGC_DEVICE_LOST;
    for (;;) { int state = command.wait(10000000); if (!state) break; if (state < 0) return MGC_DEVICE_LOST; }
    return command.host_finalize() ? MGC_INTERNAL_ERROR : MGC_SUCCESS;
}
ncnn::Option gpu_options(MgcDomain *d)
{
    ncnn::Option o;
    o.num_threads = 1; o.use_vulkan_compute = true;
    o.use_fp16_storage = o.use_fp16_packed = o.use_fp16_arithmetic = false;
    o.use_packing_layout = false;
    o.blob_vkallocator = o.workspace_vkallocator = d->blob_allocator;
    o.staging_vkallocator = d->staging_allocator;
    return o;
}
ncnn::VkMat gpu_slice(const ncnn::VkMat &allocation, int first, int count)
{
    ncnn::VkMat view = allocation;
    view.dims = 1; view.w = count; view.h = view.d = view.c = 1;
    view.cstep = count;
#if NCNN_BATCH
    view.n = 1; view.nstep = count;
#endif
    view.offset += size_t(first) * 4;
    return view;
}
#if defined(__ANDROID__)
MgcStatus probe_sync_fd(MgcDomain *domain);
MgcStatus image_ownership(MgcLease *lease,bool acquire,bool own_transition=false);

#endif
#include "gpu_kernels.inc"
#include "gpu_operations_owned.inc"
#endif
