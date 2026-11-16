/* Included in the implementation namespace after OwnedAhbTransition.
 * The registry, not MgcDomain, owns these jobs; each live job pins its domain
 * and code image through NcnnDomainOwner. No GPU object is destroyed until its
 * exact fences or the enclosing device-loss retirement proof permit cleanup. */
#if MGC_HAVE_NCNN && NCNN_VULKAN
void report_device_lost(MgcDomain*);
bool domain_is_lost(MgcDomain*);
bool device_loss_retired(MgcDomain*);
MgcStatus service_device_loss(MgcDomain*);

#if defined(__ANDROID__)
namespace standalone_gpu_detail {
constexpr size_t SlotCount=32;
constexpr uint64_t MaxWaitNs=100000000;
struct ImageKey {
    uint64_t domain=0,generation=0,allocation=0,allocation_generation=0,content=0;
    bool acquire=false;
    bool operator==(const ImageKey& other)const {
        return domain==other.domain&&generation==other.generation&&allocation==other.allocation&&
            allocation_generation==other.allocation_generation&&content==other.content&&acquire==other.acquire;
    }
};
ImageKey image_key(MgcLease* lease,bool acquire) {
    auto* resource=lease->resource;
    std::lock_guard<std::mutex> lock(resource->mutex);
    return {resource->domain->id,resource->domain->generation,resource->desc.allocation_id,
        resource->desc.allocation_generation,lease->desc.content_id,acquire};
}
MgcStatus status(VkResult result) {
    if(result==VK_ERROR_OUT_OF_HOST_MEMORY||result==VK_ERROR_OUT_OF_DEVICE_MEMORY)return MGC_OUT_OF_MEMORY;
    if(result==VK_ERROR_DEVICE_LOST)return MGC_DEVICE_LOST;
    if(result==VK_ERROR_FEATURE_NOT_PRESENT||result==VK_ERROR_EXTENSION_NOT_PRESENT||
        result==VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR)return MGC_UNSUPPORTED_CAPABILITY;
    return MGC_INTERNAL_ERROR;
}
struct Probe {
    const ncnn::VulkanDevice* device=nullptr;
    MgcDomain* domain=nullptr;
    VkSemaphore signal=VK_NULL_HANDLE,imported=VK_NULL_HANDLE;
    VkFence fences[2]{VK_NULL_HANDLE,VK_NULL_HANDLE};
    VkQueue borrowed=VK_NULL_HANDLE;
    uint32_t family=0;
    int fd=-1;
    bool possible[2]{false,false};
    bool initialized=false,complete=false;
    enum Stage { CREATE,FIRST,EXPORT,IMPORT,SECOND,RETIRE } stage=CREATE;
    MgcStatus outcome=MGC_SUCCESS;
    void remember(MgcStatus value) {
        if(value==MGC_SUCCESS||value==MGC_BUSY||value==MGC_TIMEOUT)return;
        if(outcome==MGC_SUCCESS||value==MGC_DEVICE_LOST)outcome=value;
        if(value==MGC_DEVICE_LOST)report_device_lost(domain);
    }
    bool pending()const{return possible[0]||possible[1];}
    void reclaim() {
        if(borrowed){device->reclaim_queue(family,borrowed);borrowed=VK_NULL_HANDLE;}
    }
    void cleanup() {
        if(pending()||!device)return;
        reclaim();
        if(fd>=0)close(fd);fd=-1;
        const auto vk=device->vkdevice();
        if(signal)ncnn::vkDestroySemaphore(vk,signal,nullptr);
        if(imported)ncnn::vkDestroySemaphore(vk,imported,nullptr);
        for(auto& fence:fences){if(fence)ncnn::vkDestroyFence(vk,fence,nullptr);fence=VK_NULL_HANDLE;}
        signal=imported=VK_NULL_HANDLE;
    }
    void retire_lost() {
        remember(MGC_DEVICE_LOST);possible[0]=possible[1]=false;
        cleanup();complete=true;
    }
    VkResult submit(size_t index,VkSubmitInfo& info) {
        if(domain_is_lost(domain))return VK_ERROR_DEVICE_LOST;
        borrowed=device->acquire_queue(family);
        if(!borrowed)return VK_NOT_READY;
        if(domain_is_lost(domain)){reclaim();return VK_ERROR_DEVICE_LOST;}
        // A thrown/unknown API result cannot be assumed to have rejected work.
        possible[index]=true;
        const auto result=ncnn::vkQueueSubmit(borrowed,1,&info,fences[index]);
        reclaim();
        if(result==VK_ERROR_OUT_OF_HOST_MEMORY||result==VK_ERROR_OUT_OF_DEVICE_MEMORY)possible[index]=false;
        return result;
    }
    MgcStatus advance(uint64_t wait_ns) {
        if(complete)return outcome;
        if(domain_is_lost(domain)) {
            service_device_loss(domain);
            if(!device_loss_retired(domain))return MGC_BUSY;
            retire_lost();return outcome;
        }
        const auto vk=device->vkdevice();
        if(stage==CREATE) {
            VkExportSemaphoreCreateInfo external{};external.sType=VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
            external.handleTypes=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
            VkSemaphoreCreateInfo desc{};desc.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;desc.pNext=&external;
            auto result=ncnn::vkCreateSemaphore(vk,&desc,nullptr,&signal);
            if(result!=VK_SUCCESS){remember(status(result));stage=RETIRE;}
            else {desc.pNext=nullptr;result=ncnn::vkCreateSemaphore(vk,&desc,nullptr,&imported);
                if(result!=VK_SUCCESS){remember(status(result));stage=RETIRE;}}
            if(stage==CREATE) {
                VkFenceCreateInfo fence{};fence.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                for(auto& item:fences){result=ncnn::vkCreateFence(vk,&fence,nullptr,&item);if(result!=VK_SUCCESS){remember(status(result));stage=RETIRE;break;}}
            }
            if(stage==CREATE)stage=FIRST;
        }
        if(stage==FIRST) {
            VkSubmitInfo first{};first.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;first.signalSemaphoreCount=1;first.pSignalSemaphores=&signal;
            const auto result=submit(0,first);
            if(result!=VK_SUCCESS){remember(result==VK_NOT_READY?MGC_INTERNAL_ERROR:status(result));stage=RETIRE;}
            else stage=EXPORT;
        }
        if(stage==EXPORT) {
            VkSemaphoreGetFdInfoKHR get{};get.sType=VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
            get.semaphore=signal;get.handleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
            int exported=-1;const auto result=device->vkGetSemaphoreFdKHR(vk,&get,&exported);
            if(result!=VK_SUCCESS){remember(status(result));stage=RETIRE;}
            else if(exported < -1){remember(MGC_INTERNAL_ERROR);stage=RETIRE;}
            else {fd=exported;stage=IMPORT;}
        }
        if(stage==IMPORT) {
            VkImportSemaphoreFdInfoKHR desc{};desc.sType=VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
            desc.semaphore=imported;desc.flags=VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
            desc.handleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;desc.fd=fd;
            const auto result=device->vkImportSemaphoreFdKHR(vk,&desc);
            if(result!=VK_SUCCESS){remember(status(result));stage=RETIRE;}
            else {fd=-1;stage=SECOND;}
        }
        if(stage==SECOND) {
            VkPipelineStageFlags pipeline=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            VkSubmitInfo second{};second.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;second.waitSemaphoreCount=1;
            second.pWaitSemaphores=&imported;second.pWaitDstStageMask=&pipeline;
            const auto result=submit(1,second);
            if(result!=VK_SUCCESS)remember(result==VK_NOT_READY?MGC_INTERNAL_ERROR:status(result));
            stage=RETIRE;
        }
        if(stage==RETIRE) {
            reclaim();
            for(size_t i=0;i<2;++i)if(possible[i]) {
                const auto result=ncnn::vkWaitForFences(vk,1,&fences[i],VK_TRUE,wait_ns);
                if(result==VK_SUCCESS)possible[i]=false;
                else if(result!=VK_TIMEOUT&&result!=VK_NOT_READY)remember(status(result));
            }
            if(pending())return MGC_BUSY;
            cleanup();complete=true;
        }
        return complete?outcome:MGC_BUSY;
    }
};
struct Job {
    MgcDomain* domain=nullptr;
    meguviz::NcnnDomainOwner* owner=nullptr;
    bool probe=false,acquire=false,own_flag=false,reserved=false,gpu_cleaned=false,cleaned=false;
    MgcLease* lease=nullptr;MgcLease* released_pin=nullptr;
    ImageKey key{};
    Probe capability;
    std::unique_ptr<OwnedAhbTransition> transition;
    MgcStatus outcome=MGC_SUCCESS;
    uint64_t wait_ns=0;
    void remember(MgcStatus value) {
        if(value==MGC_SUCCESS||value==MGC_BUSY||value==MGC_TIMEOUT)return;
        if(outcome==MGC_SUCCESS||value==MGC_DEVICE_LOST)outcome=value;
        if(value==MGC_DEVICE_LOST)report_device_lost(domain);
    }
    bool pending()const{return probe?capability.pending():transition&&transition->may_be_pending();}
    static MgcStatus advance_scope(void* pointer,const ncnn::VulkanDevice*) {
        auto& job=*static_cast<Job*>(pointer);
        try {
            if(job.gpu_cleaned) {
                const auto cleared=meguviz::ncnn_domain_owner_pending(job.owner,false);
                if(!cleared)job.cleaned=true;
                return cleared;
            }
            bool done=false;
            if(job.probe) {
                const auto value=job.capability.advance(job.wait_ns);job.remember(value);done=job.capability.complete;
            } else {
                if(!job.transition)job.transition=std::make_unique<OwnedAhbTransition>(job.lease,job.acquire,true);
                if(domain_is_lost(job.domain)) {
                    service_device_loss(job.domain);
                    if(!device_loss_retired(job.domain))return MGC_BUSY;
                    job.transition->retire_device_lost();
                }
                const auto value=job.transition->advance(job.wait_ns);job.remember(value);done=job.transition->retired();
            }
            if(!done){if(!job.pending())meguviz::ncnn_domain_owner_pending(job.owner,false);return MGC_BUSY;}
            if(!job.probe) {
                if(auto* pin=job.transition->take_released_pin())job.released_pin=pin;
                job.transition.reset();
                auto* resource=job.lease->resource;
                std::lock_guard<std::mutex> lock(resource->mutex);
                if(job.reserved){--resource->in_flight;job.reserved=false;}
                if(job.own_flag)resource->external_transition=false;
            } else {
                // Every non-success is an actual result, not unsupported. A
                // later explicit capability query can retry transient errors.
                job.domain->sync_fd_probe=job.outcome;
            }
            job.gpu_cleaned=true;
            const auto cleared=meguviz::ncnn_domain_owner_pending(job.owner,false);
            if(cleared)return cleared;
            job.cleaned=true;return MGC_SUCCESS;
        } catch(const std::bad_alloc&) {job.remember(MGC_OUT_OF_MEMORY);}
        catch(...) {job.remember(MGC_INTERNAL_ERROR);}
        job.capability.reclaim();
        if(job.probe)job.capability.stage=Probe::RETIRE;
        else if(!job.transition&&!job.gpu_cleaned) {
            // No transition has yet submitted. A later call performs known
            // CPU-only cleanup before retiring this failed preparation.
            if(job.reserved){std::lock_guard<std::mutex> lock(job.lease->resource->mutex);--job.lease->resource->in_flight;job.reserved=false;}
            if(job.own_flag){std::lock_guard<std::mutex> lock(job.lease->resource->mutex);job.lease->resource->external_transition=false;}
            job.gpu_cleaned=true;
            if(meguviz::ncnn_domain_owner_pending(job.owner,false)==MGC_SUCCESS)job.cleaned=true;
        }
        if(!job.pending())meguviz::ncnn_domain_owner_pending(job.owner,false);
        return MGC_BUSY;
    }
    MgcStatus service(uint64_t wait) {
        wait_ns=wait;
        if(!cleaned) {
            const auto result=domain_is_lost(domain)
                ? meguviz::try_with_ncnn_domain_owner_retire(owner,advance_scope,this)
                : meguviz::try_with_ncnn_domain_owner(owner,advance_scope,this);
            if(result!=MGC_SUCCESS)return MGC_BUSY;
        }
        // Metadata-only release must not re-enter record admission: a second
        // owner may be pending and waiting for this job's final notification.
        const auto result=meguviz::ncnn_domain_owner_release(owner);
        if(result!=MGC_SUCCESS)return MGC_BUSY;
        owner=nullptr;
        if(released_pin){mgc_lease_release(released_pin);released_pin=nullptr;}
        if(lease){mgc_lease_release(lease);lease=nullptr;}
        return MGC_SUCCESS;
    }
};
struct Slot {std::shared_ptr<Job> job;bool polling=false;};
struct Result {ImageKey key{};MgcStatus status=MGC_SUCCESS;bool valid=false;};
struct Registry {std::mutex mutex;std::array<Slot,SlotCount> slots;std::array<Result,SlotCount> results;size_t next_result=0;};
Registry& registry(){static Registry value;return value;}

MgcStatus finish_claim(std::shared_ptr<Job> job,size_t index,uint64_t wait,bool delivered) {
    MgcStatus serviced=MGC_BUSY;
    try{serviced=job->service(wait);}
    catch(...){job->remember(MGC_INTERNAL_ERROR);}
    auto& state=registry();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto& slot=state.slots[index];
        if(serviced==MGC_SUCCESS) {
            if(!job->probe&&!delivered&&job->outcome!=MGC_SUCCESS) {
                auto& result=state.results[state.next_result++%state.results.size()];result={job->key,job->outcome,true};
            }
            slot.job.reset();slot.polling=false;
        } else slot.polling=false;
    }
    return serviced==MGC_SUCCESS?job->outcome:MGC_BUSY;
}
MgcStatus start(MgcDomain* domain,MgcLease* lease,bool acquire,bool own_transition,bool probe) {
    if(active_ticket)return MGC_BUSY;
    auto key=lease?image_key(lease,acquire):ImageKey{};
    const bool lost=domain_is_lost(domain);
    auto& state=registry();std::shared_ptr<Job> job;size_t index=SlotCount;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        for(size_t i=0;i<state.slots.size();++i)if(state.slots[i].job) {
            auto& slot=state.slots[i];auto& item=*slot.job;
            if((probe&&item.probe&&item.domain==domain)||(!probe&&!item.probe&&item.key==key)) {
                if(slot.polling)return MGC_BUSY;slot.polling=true;job=slot.job;index=i;break;
            }
        }
        if(!job) {
            if(lost)return MGC_DEVICE_LOST;
            for(size_t i=0;i<state.slots.size();++i)if(!state.slots[i].job){index=i;break;}
            if(index==SlotCount)return MGC_BACKPRESSURE;
            job=std::make_shared<Job>();job->domain=domain;job->probe=probe;job->acquire=acquire;job->key=key;
            state.slots[index]={job,true};
        }
    }
    if(!job->owner) {
        MgcStatus result=MGC_INTERNAL_ERROR;
        {
            // Initial admission is a try operation; owner creation then reuses
            // the same recursive slot rather than queueing behind NN work.
            DomainScope slot(domain,true);result=slot.status;
            if(!result)result=meguviz::ncnn_domain_owner_create(domain,&job->owner);
            if(!result) {
                job->capability.device=domain->device;job->capability.domain=domain;
                job->capability.family=domain->device->info.compute_queue_family_index();
                if(lease) {
                    auto* resource=lease->resource;
                    std::lock_guard<std::mutex> lock(resource->mutex);
                    if(resource->maps||resource->in_flight||resource->ownership_uncertain||
                        (resource->external_transition&&!own_transition)||resource->external_access_state==1||resource->external_access_state==2)
                        result=MGC_BUSY;
                    else {
                        ++resource->in_flight;job->reserved=true;
                        job->own_flag=!acquire;
                        if(job->own_flag)resource->external_transition=true;
                        mgc_lease_retain(lease);job->lease=lease;
                    }
                }
            }
        }
        if(result) {
            if(job->owner) {
                job->outcome=result;job->cleaned=true;
                return finish_claim(job,index,0,true);
            }
            std::lock_guard<std::mutex> lock(state.mutex);state.slots[index]={};return result;
        }
    }
    return finish_claim(job,index,MaxWaitNs,true);
}
}

MgcStatus probe_sync_fd(MgcDomain* domain) {
    if(!domain||!domain->device)return MGC_UNSUPPORTED_CAPABILITY;
    if(domain_is_lost(domain))return MGC_DEVICE_LOST;
    const auto prior=domain->sync_fd_probe.load();
    if(prior!=MGC_BUSY) {
        if(prior!=MGC_SUCCESS&&prior!=MGC_UNSUPPORTED_CAPABILITY&&prior!=MGC_DEVICE_LOST) {
            auto expected=prior;domain->sync_fd_probe.compare_exchange_strong(expected,MGC_BUSY);
        }
        return prior;
    }
    if(!domain->device->meguviz_sync_fd_enabled||!domain->device->vkImportSemaphoreFdKHR||!domain->device->vkGetSemaphoreFdKHR){domain->sync_fd_probe=MGC_UNSUPPORTED_CAPABILITY;return MGC_UNSUPPORTED_CAPABILITY;}
    try{return standalone_gpu_detail::start(domain,nullptr,false,false,true);}
    catch(const std::bad_alloc&){return MGC_OUT_OF_MEMORY;}catch(...){return MGC_INTERNAL_ERROR;}
}
MgcStatus service_standalone_image(MgcLease* lease,bool acquire) {
    if(!lease)return MGC_INVALID_ARGUMENT;
    const auto key=standalone_gpu_detail::image_key(lease,acquire);
    auto& state=standalone_gpu_detail::registry();std::shared_ptr<standalone_gpu_detail::Job> job;size_t index=standalone_gpu_detail::SlotCount;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        for(auto& result:state.results)if(result.valid&&result.key==key){result.valid=false;return result.status;}
        for(size_t i=0;i<state.slots.size();++i){auto& slot=state.slots[i];if(slot.job&&!slot.job->probe&&slot.job->key==key){if(slot.polling)return MGC_BUSY;slot.polling=true;job=slot.job;index=i;break;}}
    }
    return job?standalone_gpu_detail::finish_claim(job,index,0,true):MGC_SUCCESS;
}
MgcStatus image_ownership(MgcLease* lease,bool acquire,bool own_transition) {
    if(!lease)return MGC_INVALID_ARGUMENT;
    if(!lease->resource->ahb)return MGC_SUCCESS;
    const auto serviced=service_standalone_image(lease,acquire);if(serviced)return serviced;
    if(domain_is_lost(lease->resource->domain))return MGC_DEVICE_LOST;
    {
        auto* resource=lease->resource;
        std::lock_guard<std::mutex> lock(resource->mutex);
        if(resource->ownership_uncertain||(resource->external_transition&&!own_transition)||resource->external_access_state==1||
            resource->external_access_state==2||(resource->external_access_state==3&&(!acquire||resource->external_access_lease!=lease)))return MGC_BUSY;
        if(acquire!=resource->externally_owned)return MGC_SUCCESS;
    }
    try{return standalone_gpu_detail::start(lease->resource->domain,lease,acquire,own_transition,false);}
    catch(const std::bad_alloc&){return MGC_OUT_OF_MEMORY;}catch(...){return MGC_INTERNAL_ERROR;}
}
#endif

MgcStatus service_standalone_gpu(MgcDomain* domain,uint64_t wait_ns) {
    if(wait_ns>100000000)return MGC_INVALID_ARGUMENT;
#if defined(__ANDROID__)
    auto& state=standalone_gpu_detail::registry();MgcStatus outcome=MGC_SUCCESS;bool waiting=false;
    for(size_t i=0;i<state.slots.size();++i) {
        std::shared_ptr<standalone_gpu_detail::Job> job;
        {
            std::lock_guard<std::mutex> lock(state.mutex);auto& slot=state.slots[i];
            if(!slot.job||(domain&&slot.job->domain!=domain))continue;
            if(slot.polling){waiting=true;continue;}slot.polling=true;job=slot.job;
        }
        const auto result=standalone_gpu_detail::finish_claim(job,i,wait_ns,false);
        if(result==MGC_BUSY)waiting=true;
        else if(result!=MGC_SUCCESS&&(outcome==MGC_SUCCESS||result==MGC_DEVICE_LOST))outcome=result;
    }
    return waiting?MGC_BUSY:outcome;
#else
    (void)domain;return MGC_SUCCESS;
#endif
}
#endif
