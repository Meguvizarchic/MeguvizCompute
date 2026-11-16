    MgcStatus mgc_lease_wait_ready(MgcLease *lease, uint64_t timeout_ns)
    { return wait_ready(lease, timeout_ns); }
    MgcStatus mgc_lease_publish_read(MgcLease *lease, MgcLease **out)
    {
        if (!lease || !out) return MGC_INVALID_ARGUMENT;
        *out = nullptr;
        auto *r = lease->resource;
        std::lock_guard<std::mutex> lock(r->mutex);
        if(r->content_invalid)return MGC_DEVICE_LOST;
#if MGC_HAVE_NCNN && NCNN_VULKAN
        if(r->desc.usage&&domain_is_lost(r->domain))return MGC_DEVICE_LOST;
#endif
        if (r->maps || (r->in_flight && !ticket_owns(lease))) return MGC_BUSY;
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
        if (r->external_transition || r->external_access_state == 1 || r->external_access_state == 2) return MGC_BUSY;
#endif
        if (lease->desc.access != MGC_ACCESS_WRITE) return MGC_INVALID_ARGUMENT;
        --r->writers; ++r->readers;
        lease->desc.access = MGC_ACCESS_READ;
        mgc_lease_retain(lease); *out = lease;
        return MGC_SUCCESS;
    }
    MgcStatus mgc_android_import(MgcDomain *domain, const MgcAndroidImportDesc *input, MgcLease **out)
    {
        if (!out) return MGC_INVALID_ARGUMENT;
        *out = nullptr;
        MgcAndroidImportDesc d{};
        auto rc = read_desc(input, d, offsetof(MgcAndroidImportDesc, reserved));
        if (rc) return rc;
        if (input->header.struct_size > offsetof(MgcAndroidImportDesc, content_context) &&
            input->header.struct_size < offsetof(MgcAndroidImportDesc, release_content) + sizeof(d.release_content))
            return MGC_UNSUPPORTED_ABI;
        rc = flags(d.required_flags, d.reserved, sizeof(d.reserved));
        if (rc) return rc;
        if (!domain || !d.buffer || !d.content_id || d.acquire_fence_fd < -1 ||
            (d.access != MGC_ACCESS_READ && d.access != MGC_ACCESS_WRITE) ||
            bool(d.retain_content) != bool(d.release_content) || d.reserved_tail)
            return MGC_INVALID_ARGUMENT;
        if ((d.layout && d.layout != MGC_ANDROID_LAYOUT_GENERAL) ||
            (d.external_queue_family && d.external_queue_family != uint32_t(-3)) ||
            d.sync_mode > MGC_ANDROID_GPU_SYNC_FD) return MGC_UNSUPPORTED_CAPABILITY;
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
        if (!domain->device || !domain->device->info.support_VK_ANDROID_external_memory_android_hardware_buffer() ||
            !domain->device->info.support_VK_EXT_queue_family_foreign()) return MGC_UNSUPPORTED_CAPABILITY;
        if (d.sync_mode == MGC_ANDROID_GPU_SYNC_FD &&
            (!domain->device->vkImportSemaphoreFdKHR || !domain->device->vkGetSemaphoreFdKHR)) return MGC_UNSUPPORTED_CAPABILITY;
        AHardwareBuffer_Desc desc{}; AHardwareBuffer_describe(d.buffer, &desc);
        if (!desc.width || !desc.height || desc.width > INT32_MAX || desc.height > INT32_MAX || desc.layers != 1 || desc.format != AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM ||
            (desc.usage & AHARDWAREBUFFER_USAGE_PROTECTED_CONTENT) ||
            !(desc.usage & AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE) || !(desc.usage & AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT))
            return MGC_UNSUPPORTED_CAPABILITY;
        if (uint64_t(desc.stride) * 4 > UINT64_MAX / desc.height) return MGC_INVALID_ARGUMENT;
        try
        {
            DomainScope slot(domain);
            if (slot.status) return slot.status;
            auto res = std::make_unique<MgcResource>();
            res->ahb_allocator = std::make_unique<AhbAllocator>(domain->device, d.buffer);
            auto import_failure = [&]() -> MgcStatus {
                const auto status = res->ahb_allocator->last_result;
                if (status == VK_ERROR_DEVICE_LOST) { report_device_lost(domain); return MGC_DEVICE_LOST; }
                if (status == VK_ERROR_OUT_OF_HOST_MEMORY || status == VK_ERROR_OUT_OF_DEVICE_MEMORY) return MGC_OUT_OF_MEMORY;
                if (status == VK_ERROR_FORMAT_NOT_SUPPORTED || status == VK_ERROR_FEATURE_NOT_PRESENT ||
                    status == VK_ERROR_EXTENSION_NOT_PRESENT || status == VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR)
                    return MGC_UNSUPPORTED_CAPABILITY;
                return status == VK_SUCCESS ? MGC_UNSUPPORTED_CAPABILITY : MGC_INTERNAL_ERROR;
            };
            if (!res->ahb_allocator->valid) return import_failure();
            res->gpu_image.create(int(desc.width), int(desc.height), size_t(4), 1, res->ahb_allocator.get());
            if (res->gpu_image.empty()) return import_failure();
            MgcResourceDesc resource{}; resource.header = MGC_HEADER_INIT(MgcResourceDesc);
            resource.allocation_id = d.allocation_id ? d.allocation_id : next_allocation++;
            resource.allocation_generation = d.allocation_generation ? d.allocation_generation : 1;
            resource.capacity_bytes = std::max<uint64_t>(uint64_t(desc.stride) * desc.height * 4, res->ahb_allocator->properties.allocationSize);
            resource.format = MGC_FORMAT_RGBA8; resource.usage = MGC_RESOURCE_ANDROID_AHB;
            resource.width = desc.width; resource.height = desc.height; resource.row_stride_bytes = uint64_t(desc.stride) * 4;
            {
                std::lock_guard<std::mutex> identity(domain->allocations_mutex);
                if (domain->allocations.count({resource.allocation_id, resource.allocation_generation}) || domain->ahb_allocations.count(d.buffer))
                    return MGC_BUSY;
                std::lock_guard<std::mutex> lock(domain->runtime->mutex);
                if (domain->runtime->closed) return MGC_BUSY;
                if (resource.capacity_bytes > domain->runtime->budget - domain->runtime->live_bytes) return MGC_OUT_OF_MEMORY;
                domain->allocations.insert({resource.allocation_id, resource.allocation_generation});
                try { domain->ahb_allocations.insert(d.buffer); }
                catch (...) { domain->allocations.erase({resource.allocation_id, resource.allocation_generation}); throw; }
                domain->runtime->live_bytes += resource.capacity_bytes;
            }
            res->desc = resource; res->domain = domain; res->ahb = d.buffer;
            res->budget_bytes = resource.capacity_bytes;
            res->accounted_bytes = res->ahb_allocator->properties.allocationSize;
            res->imported_allocation = true;
            { std::lock_guard<std::mutex> lock(domain->runtime->mutex); allocation_change(domain, res->accounted_bytes, true, true); }
            res->ahb_sync_mode = d.sync_mode;
            res->external_family = d.external_queue_family ? d.external_queue_family : VK_QUEUE_FAMILY_FOREIGN_EXT;
            AHardwareBuffer_acquire(d.buffer); mgc_domain_retain(domain);
            MgcResource *owned = res.release();
            MgcLeaseDesc ld{}; ld.header = MGC_HEADER_INIT(MgcLeaseDesc);
            ld.content_id = d.content_id; ld.byte_length = resource.capacity_bytes; ld.access = d.access;
            ld.content_context = d.content_context; ld.retain_content = d.retain_content; ld.release_content = d.release_content;
            rc = mgc_lease_create(owned, &ld, out);
            if (rc) { mgc_resource_release(owned); return rc; }
            if (d.acquire_fence_fd >= 0)
            {
                (*out)->acquire_fd = dup(d.acquire_fence_fd);
                if ((*out)->acquire_fd < 0)
                { mgc_lease_release(*out); *out = nullptr; mgc_resource_release(owned); return MGC_INTERNAL_ERROR; }
                (*out)->desc.dependency_kind = d.sync_mode ? MGC_DEPENDENCY_SYNC_FD : MGC_DEPENDENCY_HOST_WAIT;
            }
            (*out)->sync_mode = d.sync_mode;
            mgc_resource_release(owned);
            return MGC_SUCCESS;
        }
        catch (const std::bad_alloc &) { return MGC_OUT_OF_MEMORY; }
        catch (...) { return MGC_INTERNAL_ERROR; }
#else
        return MGC_UNSUPPORTED_CAPABILITY;
#endif
    }
    MgcStatus mgc_android_allocate(MgcDomain *domain, const MgcAndroidAllocateDesc *input, MgcLease **out)
    {
        if (!out) return MGC_INVALID_ARGUMENT;
        *out = nullptr;
        MgcAndroidAllocateDesc d{};
        auto rc = read_desc(input, d, offsetof(MgcAndroidAllocateDesc, reserved));
        if (rc) return rc;
        rc = flags(d.required_flags, d.reserved, sizeof(d.reserved)); if (rc) return rc;
        if (!domain || !d.width || !d.height || !d.content_id || d.width > INT32_MAX || d.height > INT32_MAX ||
            d.access != MGC_ACCESS_WRITE) return MGC_INVALID_ARGUMENT;
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
        AHardwareBuffer_Desc desc{}; desc.width = d.width; desc.height = d.height; desc.layers = 1;
        desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
        desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT | d.usage;
        if (desc.usage & AHARDWAREBUFFER_USAGE_PROTECTED_CONTENT) return MGC_UNSUPPORTED_CAPABILITY;
        AHardwareBuffer *buffer = nullptr;
        if (AHardwareBuffer_allocate(&desc, &buffer)) return MGC_OUT_OF_MEMORY;
        MgcAndroidImportDesc import{}; import.header = MGC_HEADER_INIT(MgcAndroidImportDesc);
        import.buffer = buffer; import.acquire_fence_fd = -1; import.access = d.access;
        import.content_id = d.content_id; import.sync_mode = d.sync_mode;
        rc = mgc_android_import(domain, &import, out);
        if (rc == MGC_SUCCESS) {
            (*out)->resource->externally_owned = false;
            (*out)->resource->gpu_image.data->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        }
        AHardwareBuffer_release(buffer);
        return rc;
#else
        return MGC_UNSUPPORTED_CAPABILITY;
#endif
    }
    MgcStatus mgc_android_get_buffer(MgcLease *lease, AHardwareBuffer **out)
    {
        if (!out || !lease) return MGC_INVALID_ARGUMENT;
        *out = nullptr;
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
        if (!lease->resource->ahb) return MGC_UNSUPPORTED_CAPABILITY;
        *out = lease->resource->ahb; return MGC_SUCCESS;
#else
        return MGC_UNSUPPORTED_CAPABILITY;
#endif
    }
    MgcStatus mgc_android_wait_ready(MgcLease *lease, uint64_t ns) { return mgc_lease_wait_ready(lease, ns); }
    MgcStatus mgc_android_set_acquire_fence(MgcLease *lease, int32_t fd)
    {
        if (!lease || fd < -1) return MGC_INVALID_ARGUMENT;
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
        ScopedLeaseCall call(lease);
        auto *r = lease->resource;
        MgcLease *pin = nullptr;
        {
            std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
            if (!lock.owns_lock()) return MGC_BUSY;
            if (!r->ahb) return MGC_UNSUPPORTED_CAPABILITY;
            if (r->in_flight || r->maps || r->ownership_uncertain || r->external_transition || r->external_access_state != 1 ||
                r->external_access_lease != lease || !r->externally_owned) return MGC_BUSY;
            int copy = fd < 0 ? -1 : dup(fd);
            if (fd >= 0 && copy < 0) return MGC_INTERNAL_ERROR; // reservation survives failure
            if (copy >= 0)
            {
                int producer = lease->desc.access == MGC_ACCESS_WRITE ? dup(copy) : -1;
                if (lease->desc.access == MGC_ACCESS_WRITE && producer < 0) { close(copy); return MGC_INTERNAL_ERROR; }
                r->external_access_fd = copy;
                r->external_access_state = lease->desc.access == MGC_ACCESS_WRITE ? 3 : 2;
                if (producer >= 0) {
                    lease->acquire_fd = producer;
                    lease->desc.dependency_kind = lease->sync_mode ? MGC_DEPENDENCY_SYNC_FD : MGC_DEPENDENCY_HOST_WAIT;
                }
            }
            else
            {
                pin = r->external_access_lease;
                r->external_access_lease = nullptr;
                r->external_access_state = 0;
            }
        }
        if (pin) mgc_lease_release(pin);
        return MGC_SUCCESS;
#else
        return MGC_UNSUPPORTED_CAPABILITY;
#endif
    }
    MgcStatus mgc_android_release_to_external(MgcLease *lease, int32_t *out)
    {
        if (!out || !lease) return MGC_INVALID_ARGUMENT;
        *out = -1;
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
        ScopedLeaseCall call(lease);
        auto *r = lease->resource;
        reap_external_ready(r);
        const auto pending_status = service_standalone_image(lease, false);
        if (pending_status != MGC_SUCCESS) return pending_status;
        if (domain_is_lost(r->domain)) return MGC_DEVICE_LOST;
        if (!r->domain->admission) return MGC_UNSUPPORTED_CAPABILITY;
        {
            // Metadata-only gate preserves the idle FOREIGN fast path: it does
            // not wait for an unrelated NN execution slot. Reservation and
            // quarantine publication have a defined admission order.
            std::lock_guard<std::mutex> gate(r->domain->admission->mutex);
            if (r->domain->admission->lost.load()) return MGC_DEVICE_LOST;
            if (r->domain->admission->pending && r->domain->admission->pending != active_owner) return MGC_BUSY;
            std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
            if (!lock.owns_lock()) return MGC_BUSY;
            if (!r->ahb) return MGC_UNSUPPORTED_CAPABILITY;
            if (r->maps || r->in_flight || r->ownership_uncertain || r->external_transition || r->external_access_state) return MGC_BUSY;
            if (r->externally_owned) return reserve_external_locked(lease, out);
        }
        // The ordinary FOREIGN/idle path never takes the NN execution slot.
        // A real ownership transition may use the slot only if immediately free.
        DomainScope slot(r->domain, true);
        if (slot.status) return slot.status;
        {
            std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
            if (!lock.owns_lock() || r->maps || r->in_flight || r->ownership_uncertain || r->external_transition || r->external_access_state) return MGC_BUSY;
            if (r->externally_owned) return reserve_external_locked(lease, out);
            r->external_transition = true;
        }
        auto status = image_ownership(lease, false, true);
        {
            std::lock_guard<std::mutex> lock(r->mutex);
            r->external_transition = false;
            if (!status) status = reserve_external_locked(lease, out);
        }
        return status;
#else
        return MGC_UNSUPPORTED_CAPABILITY;
#endif
    }
    MgcStatus mgc_android_export_release_fence(MgcLease *lease, int32_t *out)
    { return mgc_android_release_to_external(lease, out); }
