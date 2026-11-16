/* Included in the implementation namespace, after opaque object definitions. */
#if defined(__ANDROID__)
MgcStatus wait_fd(int fd, uint64_t timeout_ns)
{
    return meguviz_compute_detail::wait_fd(fd, timeout_ns,
        [] { return std::chrono::steady_clock::now(); },
        [](pollfd *p, int ms) { return poll(p, 1, ms); });
}
#endif
struct ScopedLeaseCall {
    MgcLease *lease;
    explicit ScopedLeaseCall(MgcLease *value):lease(value) { mgc_lease_retain(lease); }
    ~ScopedLeaseCall() { mgc_lease_release(lease); }
};
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
MgcLease *detach_external_ready_locked(MgcResource *r)
{
    if ((r->external_access_state != 2 && r->external_access_state != 3) || wait_fd(r->external_access_fd, 0) != MGC_SUCCESS) return nullptr;
    if (r->external_access_fd >= 0) close(r->external_access_fd);
    r->external_access_fd = -1;
    r->external_access_state = 0;
    auto *pin = r->external_access_lease;
    r->external_access_lease = nullptr;
    return pin;
}
void reap_external_ready(MgcResource *r)
{
    MgcLease *pin = nullptr;
    {
        std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
        if (lock.owns_lock()) pin = detach_external_ready_locked(r);
    }
    if (pin) mgc_lease_release(pin);
}
MgcStatus reserve_external_locked(MgcLease *lease, int32_t *out)
{
    auto *r = lease->resource;
    if (!r->ahb) return MGC_UNSUPPORTED_CAPABILITY;
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
    ScopedLeaseCall call( l );
    const auto start = std::chrono::steady_clock::now();
    auto *r = l->resource;
    // Snapshot/dup under the resource lock, but never hold that lock while
    // waiting. The resource generation also guards the ordinary acquire fd:
    // a later external reservation can install a different producer fence
    // with the same descriptor number while this waiter is outside the lock.
    for (unsigned attempt = 0; attempt < 8; ++attempt)
    {
        int copy = -1, original = -1;
        bool external = false;
        uint64_t generation = 0;
        {
            std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
            if (!lock.owns_lock()) return MGC_BUSY;
#if MGC_HAVE_NCNN && NCNN_VULKAN
            if (r->external_transition || r->external_access_state == 1) return MGC_BUSY;
            if (r->external_access_state == 2 || r->external_access_state == 3)
            { external = true; original = r->external_access_fd; generation = r->external_access_generation; }
            else
#endif
            {
                original = l->acquire_fd;
#if MGC_HAVE_NCNN && NCNN_VULKAN
                generation = r->external_access_generation;
#endif
            }
            if (original < 0) return MGC_SUCCESS;
            copy = dup(original);
            if (copy < 0) return MGC_INTERNAL_ERROR;
        }
        uint64_t remaining = timeout_ns;
        if (timeout_ns != UINT64_MAX && timeout_ns)
        {
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            if (uint64_t(elapsed) >= timeout_ns) { close(copy); return MGC_TIMEOUT; }
            remaining -= uint64_t(elapsed);
        }
        auto status = wait_fd(copy, remaining); close(copy);
        if (status) return status;
        MgcLease *pin = nullptr;
        {
            std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
            if (!lock.owns_lock()) return MGC_BUSY;
#if MGC_HAVE_NCNN && NCNN_VULKAN
            if (external)
            {
                if ((r->external_access_state == 2 || r->external_access_state == 3) && r->external_access_generation == generation)
                {
                    if (r->external_access_fd >= 0) close(r->external_access_fd);
                    r->external_access_fd = -1; r->external_access_state = 0;
                    pin = r->external_access_lease; r->external_access_lease = nullptr;
                }
            }
            else
#endif
            if (l->acquire_fd == original
#if MGC_HAVE_NCNN && NCNN_VULKAN
                && r->external_access_generation == generation
#endif
            )
            { close(l->acquire_fd); l->acquire_fd = -1; l->desc.dependency_kind = MGC_DEPENDENCY_READY; }
        }
        if (pin) mgc_lease_release(pin);
        (void)external; (void)generation;
    }
    return MGC_BUSY;
#else
    (void)timeout_ns;
    return l->desc.dependency_kind == MGC_DEPENDENCY_READY ? MGC_SUCCESS : MGC_UNSUPPORTED_CAPABILITY;
#endif
}
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
