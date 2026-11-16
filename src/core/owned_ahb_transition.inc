/* Included in the implementation namespace after gpu_runtime.inc.
 * The enclosing owned batch pins lease/domain and serializes every call in an
 * authorized device scope. It must keep this object until retired() is true.
 * take_released_pin() transfers the detached producer reference to the batch,
 * which releases that reference outside its internal/device locks. */
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
class OwnedAhbTransition final {
public:
    OwnedAhbTransition(MgcLease *lease, bool acquire, bool own_transition = false)
        : lease_(lease), resource_(lease ? lease->resource : nullptr),
          ticket_(active_ticket), acquire_(acquire), own_transition_(own_transition) {}
    OwnedAhbTransition(const OwnedAhbTransition &) = delete;
    OwnedAhbTransition &operator=(const OwnedAhbTransition &) = delete;

    ~OwnedAhbTransition() {
        // A pending object is never a permission to destroy its Vulkan state.
        // Root ownership must retain the entire object until this is false.
        if (!may_be_pending_) cleanup();
    }

    bool retired() const noexcept { return done_; }
    MgcStatus outcome() const noexcept { return outcome_; }
    bool may_be_pending() const noexcept { return may_be_pending_; }
    void retire_device_lost() {
        if(!device_loss_retired(resource_?resource_->domain:nullptr))return;
        remember(MGC_DEVICE_LOST);
        may_be_pending_=false;done_=true;
        if(resource_&&submitted_once_){std::lock_guard<std::mutex> lock(resource_->mutex);resource_->ownership_uncertain=true;}
        // Do not close the original producer FD, detach its reservation, or
        // report FOREIGN/READY. External GLES/producer retirement is separate.
        cleanup();
    }
    MgcLease *take_released_pin() noexcept {
        auto *pin = released_pin_; released_pin_ = nullptr; return pin;
    }

    // Reuse the preallocated wrapper only when no batch could still execute.
    // A successful fence or applied ownership is never submitted a second time.
    // Keep the original error: eventual physical cleanup does not make the
    // surrounding job successful after an earlier failed release attempt.
    bool retry_unsubmitted() {
        if (may_be_pending_ || gpu_ready_ || state_applied_) return false;
        try { cleanup(); }
        catch (...) { remember(MGC_INTERNAL_ERROR); return false; }
        done_ = false; submitted_once_ = false; observed_submit_ = false;
        signal_semaphore_ = false; export_attempted_ = false; export_succeeded_ = false;
        fd_published_ = false; observation_finished_ = false;
        barrier_ = {}; original_acquire_fd_ = -1; acquire_generation_ = 0;
        queue_family_ = 0;
        return true;
    }

    // Cancellation before this conversion's first possible submit requires no
    // wait on a producer the job has never read. It does not cancel GPU work.
    bool discard_unsubmitted() {
        if (may_be_pending_ || submitted_once_) return false;
        if (!done_) { remember(MGC_CANCELLED); done_ = true; cleanup(); }
        return true;
    }

    MgcStatus advance(uint64_t wait_ns) {
        if (wait_ns == UINT64_MAX) return MGC_INVALID_ARGUMENT;
        if (done_) return outcome_;
        if (resource_ && domain_is_lost(resource_->domain)) return MGC_BUSY;
        try {
            if (!submitted_once_) {
                const auto status = prepare_and_submit(wait_ns);
                if (resource_ && domain_is_lost(resource_->domain)) return MGC_BUSY;
                if (done_ || status != MGC_SUCCESS) return status;
            }
            if (!gpu_ready_) {
                const auto result = ncnn::vkWaitForFences(device_->vkdevice(), 1, &fence_, VK_TRUE, wait_ns);
                if (result == VK_TIMEOUT || result == VK_NOT_READY) return MGC_BUSY;
                if (result != VK_SUCCESS) { remember(vk_error(result)); return MGC_BUSY; }
                gpu_ready_ = true;
                may_be_pending_ = false;
            }
            if (resource_ && domain_is_lost(resource_->domain)) return MGC_BUSY;
            // An uncertain submit has no known signal operation to export
            // before retirement. Only a confirmed fence permits this attempt.
            if (signal_semaphore_ && !export_attempted_) export_release_fd();
            if (resource_ && domain_is_lost(resource_->domain)) return MGC_BUSY;
            finish_retirement();
            return outcome_;
        } catch (const std::bad_alloc &) {
            return exception(MGC_OUT_OF_MEMORY);
        } catch (...) {
            return exception(MGC_INTERNAL_ERROR);
        }
    }

private:
    MgcLease *lease_ = nullptr;
    MgcResource *resource_ = nullptr;
    MgcTicket *ticket_ = nullptr;
    const ncnn::VulkanDevice *device_ = nullptr;
    bool acquire_ = false, own_transition_ = false, done_ = false, submitted_once_ = false;
    bool may_be_pending_ = false, gpu_ready_ = false, observed_submit_ = false;
    bool signal_semaphore_ = false, export_attempted_ = false, export_succeeded_ = false, fd_published_ = false;
    bool state_applied_ = false, observation_finished_ = false;
    MgcStatus outcome_ = MGC_SUCCESS;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
    VkQueue borrowed_queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VkImageMemoryBarrier barrier_{};
    int acquire_copy_ = -1, exported_fd_ = -1, original_acquire_fd_ = -1;
    uint64_t acquire_generation_ = 0;
    MgcLease *released_pin_ = nullptr;
    std::unique_ptr<ncnn::VkBatchTimestamp> measurement_;

    static MgcStatus vk_error(VkResult result) {
        if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
            return MGC_OUT_OF_MEMORY;
        return result == VK_ERROR_DEVICE_LOST ? MGC_DEVICE_LOST : MGC_INTERNAL_ERROR;
    }
    void remember(MgcStatus status) noexcept {
        if (status == MGC_SUCCESS || status == MGC_BUSY || status == MGC_TIMEOUT) return;
        // Do not let a later generic error or cancellation hide device loss.
        if (outcome_ == MGC_SUCCESS || status == MGC_DEVICE_LOST) outcome_ = status;
        if(status==MGC_DEVICE_LOST&&resource_)report_device_lost(resource_->domain);
    }
    MgcStatus finish_unsubmitted(MgcStatus status) {
        if (outcome_ == MGC_SUCCESS || status == MGC_DEVICE_LOST) outcome_ = status;
        if (status == MGC_DEVICE_LOST && resource_) report_device_lost(resource_->domain);
        done_ = true; may_be_pending_ = false; cleanup(); return outcome_;
    }
    MgcStatus exception(MgcStatus status) {
        remember(status);
        if (may_be_pending_) return MGC_BUSY;
        // Fence success proves physical safety, but ownership bookkeeping must
        // finish before the enclosing batch can return the resource. Keep the
        // wrapper and its uncertain state for a later metadata-only attempt.
        if (gpu_ready_ && !state_applied_) return MGC_BUSY;
        if (ticket_ && gpu_ready_) { ticket_->gpu_complete = false; ticket_->event_complete = false; }
        done_ = true; cleanup(); return outcome_;
    }
    void cleanup() {
        if (may_be_pending_ || !device_) return;
        const auto vk = device_->vkdevice();
        if (borrowed_queue_) { device_->reclaim_queue(queue_family_, borrowed_queue_); borrowed_queue_ = VK_NULL_HANDLE; }
        measurement_.reset();
        if (semaphore_) ncnn::vkDestroySemaphore(vk, semaphore_, nullptr);
        if (fence_) ncnn::vkDestroyFence(vk, fence_, nullptr);
        if (pool_) ncnn::vkDestroyCommandPool(vk, pool_, nullptr);
        semaphore_ = VK_NULL_HANDLE; fence_ = VK_NULL_HANDLE; pool_ = VK_NULL_HANDLE; command_ = VK_NULL_HANDLE;
        if (acquire_copy_ >= 0) close(acquire_copy_);
        if (exported_fd_ >= 0) close(exported_fd_);
        acquire_copy_ = exported_fd_ = -1;
    }
    void export_release_fd() {
        export_attempted_ = true;
        VkSemaphoreGetFdInfoKHR get{}; get.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        get.semaphore = semaphore_; get.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        int fd = -1;
        const auto status = device_->vkGetSemaphoreFdKHR(device_->vkdevice(), &get, &fd);
        if (status != VK_SUCCESS) { remember(vk_error(status)); return; }
        if (fd < -1) { remember(MGC_INTERNAL_ERROR); return; }
        // -1 is accepted only from a successful Vulkan SYNC_FD export; no
        // application-generated satisfied FD substitutes for a failed export.
        export_succeeded_ = true; exported_fd_ = fd;
    }
    MgcStatus prepare_and_submit(uint64_t wait_ns) {
        if (!lease_ || !resource_) return finish_unsubmitted(MGC_INVALID_ARGUMENT);
        if (!resource_->ahb) return finish_unsubmitted(MGC_SUCCESS);
        device_ = resource_->domain->device;
        if (!device_ || !device_->info.support_VK_EXT_queue_family_foreign() || !resource_->gpu_image.data)
            return finish_unsubmitted(MGC_UNSUPPORTED_CAPABILITY);
        {
            std::lock_guard<std::mutex> lock(resource_->mutex);
            if (resource_->ownership_uncertain || (resource_->external_transition && !own_transition_) ||
                resource_->external_access_state == 1 || resource_->external_access_state == 2 ||
                (resource_->external_access_state == 3 && (!acquire_ || resource_->external_access_lease != lease_)))
                return finish_unsubmitted(MGC_BUSY);
            if (acquire_ != resource_->externally_owned) return finish_unsubmitted(MGC_SUCCESS);
        }
        const bool native = lease_->sync_mode == MGC_ANDROID_GPU_SYNC_FD;
        if (native && (!device_->vkImportSemaphoreFdKHR || !device_->vkGetSemaphoreFdKHR))
            return finish_unsubmitted(MGC_UNSUPPORTED_CAPABILITY);
        if (acquire_ && !native) {
            const auto ready = wait_ready(lease_, wait_ns);
            if (ready == MGC_TIMEOUT || ready == MGC_BUSY) return MGC_BUSY;
            if (ready != MGC_SUCCESS) return finish_unsubmitted(ready);
        }
        const auto vk = device_->vkdevice();
        if ((ticket_ ? ticket_->measurement_flags : resource_->domain->measurement_flags.load()) & MGC_MEASURE_GPU_TIMESTAMPS) {
            measurement_ = std::make_unique<ncnn::VkBatchTimestamp>(device_); measurement_->configure(true);
        }
        VkCommandPoolCreateInfo pool{}; pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT; pool.queueFamilyIndex = device_->info.compute_queue_family_index();
        auto status = ncnn::vkCreateCommandPool(vk, &pool, nullptr, &pool_);
        if (status != VK_SUCCESS) return finish_unsubmitted(vk_error(status));
        VkCommandBufferAllocateInfo allocation{}; allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocation.commandPool = pool_; allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocation.commandBufferCount = 1;
        status = ncnn::vkAllocateCommandBuffers(vk, &allocation, &command_);
        if (status != VK_SUCCESS) return finish_unsubmitted(vk_error(status));
        VkFenceCreateInfo fence{}; fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        status = ncnn::vkCreateFence(vk, &fence, nullptr, &fence_);
        if (status != VK_SUCCESS) return finish_unsubmitted(vk_error(status));
        if (acquire_) {
            std::lock_guard<std::mutex> lock(resource_->mutex);
            original_acquire_fd_ = lease_->acquire_fd;
            acquire_generation_ = resource_->external_access_generation;
            if (native && original_acquire_fd_ >= 0) {
                acquire_copy_ = dup(original_acquire_fd_);
                if (acquire_copy_ < 0) return finish_unsubmitted(MGC_INTERNAL_ERROR);
            }
        }
        const bool wait_semaphore = acquire_copy_ >= 0;
        signal_semaphore_ = native && !acquire_;
        if (wait_semaphore || signal_semaphore_) {
            VkExportSemaphoreCreateInfo export_info{}; export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
            export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
            VkSemaphoreCreateInfo semaphore{}; semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            if (signal_semaphore_) semaphore.pNext = &export_info;
            status = ncnn::vkCreateSemaphore(vk, &semaphore, nullptr, &semaphore_);
            if (status != VK_SUCCESS) return finish_unsubmitted(vk_error(status));
            if (wait_semaphore) {
                VkImportSemaphoreFdInfoKHR import{}; import.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
                import.semaphore = semaphore_; import.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
                import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT; import.fd = acquire_copy_;
                status = device_->vkImportSemaphoreFdKHR(vk, &import);
                if (status != VK_SUCCESS) return finish_unsubmitted(status == VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR ? MGC_UNSUPPORTED_CAPABILITY : vk_error(status));
                acquire_copy_ = -1;
            }
        }
        VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        status = ncnn::vkBeginCommandBuffer(command_, &begin);
        if (status != VK_SUCCESS) return finish_unsubmitted(vk_error(status));
        if (measurement_) measurement_->begin(command_);
        const auto *image = resource_->gpu_image.data;
        barrier_.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier_.srcAccessMask = acquire_ ? 0 : image->access_flags;
        barrier_.dstAccessMask = acquire_ ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT : 0;
        barrier_.oldLayout = acquire_ ? VK_IMAGE_LAYOUT_GENERAL : image->image_layout;
        barrier_.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier_.srcQueueFamilyIndex = acquire_ ? resource_->external_family : pool.queueFamilyIndex;
        barrier_.dstQueueFamilyIndex = acquire_ ? pool.queueFamilyIndex : resource_->external_family;
        barrier_.image = image->image; barrier_.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ncnn::vkCmdPipelineBarrier(command_, acquire_ ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            acquire_ ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier_);
        if (measurement_) measurement_->end(command_);
        status = ncnn::vkEndCommandBuffer(command_);
        if (status != VK_SUCCESS) return finish_unsubmitted(vk_error(status));
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &command_;
        if (wait_semaphore) { submit.waitSemaphoreCount = 1; submit.pWaitSemaphores = &semaphore_; submit.pWaitDstStageMask = &stage; }
        if (signal_semaphore_) { submit.signalSemaphoreCount = 1; submit.pSignalSemaphores = &semaphore_; }
        queue_family_ = pool.queueFamilyIndex;
        borrowed_queue_ = device_->acquire_queue(queue_family_);
        if (!borrowed_queue_) return finish_unsubmitted(MGC_INTERNAL_ERROR);
        if (measurement_) measurement_->submitted();
        // Publish possible submission before the external Vulkan call. Any
        // non-OOM unknown failure or exceptional return retains this state.
        submitted_once_ = true; may_be_pending_ = true;
        { std::lock_guard<std::mutex> lock(resource_->mutex); resource_->ownership_uncertain = true; }
        status = ncnn::vkQueueSubmit(borrowed_queue_, 1, &submit, fence_);
        device_->reclaim_queue(queue_family_, borrowed_queue_); borrowed_queue_ = VK_NULL_HANDLE;
        if (status == VK_ERROR_OUT_OF_HOST_MEMORY || status == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
            may_be_pending_ = false;
            { std::lock_guard<std::mutex> lock(resource_->mutex); resource_->ownership_uncertain = false; }
            return finish_unsubmitted(MGC_OUT_OF_MEMORY);
        }
        if (status != VK_SUCCESS) {
            remember(vk_error(status));
            if (ticket_) { ticket_->gpu_complete = false; ticket_->event_complete = false; }
            return MGC_SUCCESS; // Poll the retained fence; not logical success.
        }
        observed_submit_ = true;
        gpu_batch_observer(ticket_, 0, 1);
        if (signal_semaphore_) export_release_fd();
        return MGC_SUCCESS;
    }
    void finish_retirement() {
        if (!state_applied_) {
            std::lock_guard<std::mutex> lock(resource_->mutex);
            auto *image = resource_->gpu_image.data;
            resource_->externally_owned = !acquire_;
            image->access_flags = acquire_ ? barrier_.dstAccessMask : 0;
            image->image_layout = VK_IMAGE_LAYOUT_GENERAL;
            image->stage_flags = acquire_ ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            if (acquire_ && lease_->acquire_fd == original_acquire_fd_ && resource_->external_access_generation == acquire_generation_) {
                if (lease_->acquire_fd >= 0) close(lease_->acquire_fd);
                lease_->acquire_fd = -1;
            }
            if (signal_semaphore_ && export_succeeded_ && !fd_published_) {
                if (resource_->release_fd >= 0) close(resource_->release_fd);
                resource_->release_fd = exported_fd_; exported_fd_ = -1;
                fd_published_ = true;
            }
            lease_->desc.dependency_kind = MGC_DEPENDENCY_READY;
            if (acquire_ && resource_->external_access_state == 3 && resource_->external_access_lease == lease_)
                released_pin_ = detach_external_ready_locked(resource_);
            // This fresh fence has actually completed the recorded ownership
            // barrier. Logical failure remains in outcome_, independently of
            // the now-confirmed layout/queue state. Clear uncertainty only
            // after every ownership metadata step has completed.
            resource_->ownership_uncertain = false;
            state_applied_ = true;
        }
        if (!observation_finished_) {
            observation_finished_ = true;
            if (measurement_ && observed_submit_) {
                measurement_->retired(); uint64_t ns = 0; uint32_t observed = 0, bits = 0, period = 0;
                measurement_->result(&ns, &observed, &bits, &period); observe_gpu(ticket_, ns, observed, bits, period);
            }
            if (observed_submit_) { gpu_batch_observer(ticket_, 0, 2); gpu_batch_observer(ticket_, 0, 3); }
        }
        done_ = true; cleanup();
    }
};
#endif
