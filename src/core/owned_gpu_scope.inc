// Included in namespace meguviz after the internal owner implementation.
namespace {
thread_local bool last_owned_scope_deferred = false;
struct OwnedGpuBatch {
    MgcDomain *domain = nullptr;
    MgcTicket *ticket = nullptr;
    NcnnDomainOwner *owner = nullptr;
    NcnnGpuFn record = nullptr;
    MgcStatus (*retire_aux)(void*,uint64_t) = nullptr;
    void* aux_context = nullptr;
    bool aux_pending = false;
    void *context = nullptr;
    std::shared_ptr<void> keepalive;
    std::unique_ptr<ncnn::VkCompute> command;
    std::vector<MgcLease*> requested, pinned, detached;
    std::vector<MgcResource*> idle;
    std::vector<ncnn::VkMat> buffers;
    std::vector<NcnnGpuView> views;
#if defined(__ANDROID__)
    std::vector<std::unique_ptr<OwnedAhbTransition>> acquires, releases;
#endif
    enum Stage { PREPARE, ACQUIRE, RECORD, COMMAND, RELEASE, CLEANUP, DONE } stage = PREPARE;
    size_t acquire_index = 0, release_index = 0;
    bool recorded = false, callback_ok = false, failed = false, cleaned = false;
    bool final_submission_accepted = false;
    uint64_t wait_budget = 0;
    MgcStatus failure = MGC_SUCCESS;
    void fail(MgcStatus status) {
        if(status==MGC_SUCCESS)return;
        if(!failed||status==MGC_DEVICE_LOST)failure=status;
        failed=true; callback_ok=false;
        if(ticket->deferred){ticket->resume_allowed=false;ticket->deferred_failure=failure;}
    }
    static MgcStatus vk_status(VkResult result) {
        return result==VK_ERROR_OUT_OF_HOST_MEMORY||result==VK_ERROR_OUT_OF_DEVICE_MEMORY?MGC_OUT_OF_MEMORY:
            result==VK_ERROR_DEVICE_LOST?MGC_DEVICE_LOST:MGC_INTERNAL_ERROR;
    }
    MgcStatus prepare() {
        const auto count=requested.size();
        buffers.resize(count);views.resize(count);pinned.reserve(count);idle.reserve(count);detached.reserve(count*2);
#if defined(__ANDROID__)
        acquires.resize(count);releases.resize(count);
#endif
        for(size_t i=0;i<count;++i){
            auto *lease=requested[i];if(!lease)return MGC_INVALID_ARGUMENT;
            auto *resource=lease->resource;
            if(resource->domain!=domain)return MGC_INCOMPATIBLE_DOMAIN;
            if(!resource->desc.usage)return MGC_UNSUPPORTED_CAPABILITY;
            {
                std::lock_guard<std::mutex> lock(resource->mutex);
#if defined(__ANDROID__)
                if(resource->ownership_uncertain||resource->external_transition||resource->external_access_state==1||resource->external_access_state==2||
                    (resource->external_access_state==3&&resource->external_access_lease!=lease))return MGC_BUSY;
#endif
                if(resource->maps||(resource->in_flight&&!ticket_owns(lease)&&std::find(idle.begin(),idle.end(),resource)==idle.end()))return MGC_BUSY;
                if(!resource->in_flight){++resource->in_flight;idle.push_back(resource);}
            }
            mgc_lease_retain(lease);pinned.push_back(lease);
            views[i].lease=lease;views[i].info.header=MGC_HEADER_INIT(MgcLeaseInfo);
            auto status=mgc_lease_info(lease,&views[i].info);if(status)return status;
            if(resource->desc.usage==MGC_RESOURCE_GPU_BUFFER){
                if(lease->desc.byte_offset%4||lease->desc.byte_length>uint64_t(INT32_MAX)*4)return MGC_INVALID_ARGUMENT;
                buffers[i]=resource->gpu_buffer;views[i].buffer=&buffers[i];
            }
#if defined(__ANDROID__)
            else if(resource->ahb){
                views[i].image=&resource->gpu_image;
                acquires[i]=std::make_unique<OwnedAhbTransition>(lease,true);
                releases[i]=std::make_unique<OwnedAhbTransition>(lease,false);
            }
#endif
            else return MGC_UNSUPPORTED_CAPABILITY;
        }
        command=std::make_unique<ncnn::VkCompute>(domain->device);
        if(!command->is_valid())return vk_status(command->last_operation_result());
        if(command->set_retirement_managed(100000000)!=0)return MGC_INTERNAL_ERROR;
        command->set_submit_observer(gpu_batch_observer,ticket);
        configure_gpu_measurement(*command,domain,ticket);
        return MGC_SUCCESS;
    }
    MgcStatus advance_command(uint64_t wait_ns) {
        if(!command)return MGC_SUCCESS;
        auto state=command->execution_state();
        if((state==0||state==1)&&failed){
            if(command->discard_unsubmitted()!=0)return MGC_BUSY;
            return MGC_SUCCESS;
        }
        if(state==0||state==1){
            ticket->final_submitting=true;
            const auto result=command->submit_async();
            ticket->final_submitting=false;
            if(!result){ticket->used_final_async=true;final_submission_accepted=true;}
            else fail(vk_status(command->last_submit_result()));
            state=command->execution_state();
            if(command->last_submit_result()==VK_ERROR_DEVICE_LOST){report_device_lost(domain);return MGC_BUSY;}
            if(result&&(state==0||state==1)){command->discard_unsubmitted();return MGC_SUCCESS;}
        }
        if(state==2||state==5){
            const auto waited=command->wait_compute_retirement(wait_ns);
            if(command->submission_uncertain()){ticket->gpu_complete=false;ticket->event_complete=false;fail(vk_status(command->last_submit_result()));}
            if(waited<0){fail(vk_status(command->last_wait_result()));if(command->last_wait_result()==VK_ERROR_DEVICE_LOST){report_device_lost(domain);return MGC_BUSY;}}
            if(!command->compute_batches_retired())return MGC_BUSY;
            state=command->execution_state();
        }
        if(state==3||state==4){
            if(command->host_finalize()!=0)fail(MGC_INTERNAL_ERROR);
            if(!failed&&callback_ok&&final_submission_accepted)domain->vulkan_submission_probe=MGC_SUCCESS;
            return MGC_SUCCESS;
        }
        if(!command->compute_batches_retired())return MGC_BUSY;
        // An initialized command with a failed/unsubmitted tail has no host
        // writes to finalize. Never roll back a possibly accepted submission.
        return MGC_SUCCESS;
    }
    MgcStatus advance(uint64_t wait_ns) {
        if(stage==DONE)return ncnn_domain_owner_pending(owner,false);
        if(domain_is_lost(domain)) {
            fail(MGC_DEVICE_LOST);
            service_device_loss(domain);
            if(!device_loss_retired(domain))return MGC_BUSY;
            if(command)command->retire_device_lost();
            if(retire_aux&&retire_aux(aux_context,0)!=MGC_SUCCESS)return MGC_BUSY;
            aux_pending=false;
#if defined(__ANDROID__)
            for(auto* list:{&acquires,&releases})for(auto& transition:*list)if(transition)transition->retire_device_lost();
#endif
            stage=CLEANUP;
        }
        if(stage==PREPARE){
            auto status=prepare();
            if(status==MGC_DEVICE_LOST)report_device_lost(domain);
            if(status){fail(status);stage=CLEANUP;}
            else stage=ACQUIRE;
        }
        if(stage!=CLEANUP&&domain_is_lost(domain))return MGC_BUSY;
        if(stage==ACQUIRE){
#if defined(__ANDROID__)
            while(acquire_index<acquires.size()){
                auto& transition=acquires[acquire_index];
                if(!transition){++acquire_index;continue;}
                if(ticket->cancelled.load()&&transition->discard_unsubmitted()){fail(MGC_CANCELLED);stage=RELEASE;break;}
                auto status=transition->advance(wait_ns);
                if(domain_is_lost(domain))return MGC_BUSY;
                if(!transition->retired())return MGC_BUSY;
                ++acquire_index;
                if(status){fail(status);stage=RELEASE;break;}
            }
#endif
            if(stage==ACQUIRE)stage=RECORD;
        }
        if(stage!=CLEANUP&&domain_is_lost(domain))return MGC_BUSY;
        if(stage==RECORD){
            if(ticket->cancelled.load()){fail(MGC_CANCELLED);stage=RELEASE;}
            else {
                recorded=true;
                NcnnGpuScope scope{domain->device,domain->blob_allocator,domain->staging_allocator,command.get(),views.data(),uint32_t(views.size())};
                MgcStatus status;
                try{status=record(context,scope);}catch(const std::bad_alloc&){status=MGC_OUT_OF_MEMORY;}catch(...){status=MGC_INTERNAL_ERROR;}
                retire_aux=scope.retire_aux;aux_context=scope.aux_context;
                if(retire_aux){ticket->gpu_complete=false;ticket->event_complete=false;aux_pending=true;}
                callback_ok=status==MGC_SUCCESS;
                if(status){
                    auto cause=status;
                    if(command->last_submit_result()==VK_ERROR_DEVICE_LOST||command->last_wait_result()==VK_ERROR_DEVICE_LOST)cause=MGC_DEVICE_LOST;
                    else if(status==MGC_INTERNAL_ERROR&&!command->compute_batches_retired()&&
                        (command->last_wait_result()==VK_TIMEOUT||command->last_wait_result()==VK_NOT_READY))cause=MGC_TIMEOUT;
                    fail(cause);
                    if(cause==MGC_DEVICE_LOST)report_device_lost(domain);
                }
                stage=COMMAND;
            }
        }
        if(stage!=CLEANUP&&domain_is_lost(domain))return MGC_BUSY;
        if(stage==COMMAND){
            if(retire_aux){
                const auto extra=retire_aux(aux_context,wait_ns);
                if(extra==MGC_DEVICE_LOST){report_device_lost(domain);fail(extra);return MGC_BUSY;}
                if(extra!=MGC_SUCCESS)return MGC_BUSY;
                aux_pending=false;retire_aux=nullptr;aux_context=nullptr;
            }
            if(domain_is_lost(domain))return MGC_BUSY;
            if(advance_command(wait_ns)!=MGC_SUCCESS)return MGC_BUSY;
            stage=RELEASE;
        }
        if(stage!=CLEANUP&&domain_is_lost(domain))return MGC_BUSY;
        if(stage==RELEASE){
#if defined(__ANDROID__)
            while(release_index<releases.size()){
                auto& transition=releases[release_index];
                if(!transition||release_index>=acquire_index){++release_index;continue;}
                auto *resource=pinned[release_index]->resource;
                auto status=transition->advance(wait_ns);
                if(domain_is_lost(domain))return MGC_BUSY;
                if(!transition->retired())return MGC_BUSY;
                if(status)fail(status);
                bool externally_ready;
                {std::lock_guard<std::mutex> lock(resource->mutex);externally_ready=resource->externally_owned&&!resource->ownership_uncertain;}
                if(!externally_ready){transition->retry_unsubmitted();return MGC_BUSY;}
                ++release_index;
            }
#endif
            stage=CLEANUP;
        }
        if(stage==CLEANUP){
            if(command&&!command->compute_batches_retired())return MGC_BUSY;
#if defined(__ANDROID__)
            for(auto *list:{&acquires,&releases})for(auto& transition:*list)if(transition){
                if(transition->may_be_pending())return MGC_BUSY;
                if(auto* pin=transition->take_released_pin())detached.push_back(pin);
                transition.reset();
            }
#endif
            command.reset();buffers.clear();views.clear();
            for(auto *resource:idle){std::lock_guard<std::mutex> lock(resource->mutex);--resource->in_flight;}
            idle.clear();
            const auto status=ncnn_domain_owner_pending(owner,false);
            if(status)return status;
            cleaned=true;stage=DONE;
            return MGC_SUCCESS;
        }
        return MGC_SUCCESS;
    }
    static MgcStatus run(void* pointer,const ncnn::VulkanDevice*) {
        auto& self=*static_cast<OwnedGpuBatch*>(pointer);
        try{
            const auto status=self.advance(self.wait_budget);
            // Producer-readiness waits and retryable unsubmitted bookkeeping
            // retain this task's leases, but must not quarantine an otherwise
            // idle device and prevent its producer from making progress.
            if(status!=MGC_SUCCESS&&!self.gpu_pending())ncnn_domain_owner_pending(self.owner,false);
            return status;
        }
        catch(const std::bad_alloc&){self.fail(MGC_OUT_OF_MEMORY);}
        catch(...){self.fail(MGC_INTERNAL_ERROR);}
        // Before recording starts no work was submitted; after it starts the
        // command's retained state decides whether the tail can be discarded.
        if(self.stage==PREPARE)self.stage=CLEANUP;
        else if(self.stage==RECORD)self.stage=COMMAND;
        return MGC_BUSY;
    }
    bool gpu_pending()const {
        if(aux_pending)return true;
        if(command&&!command->compute_batches_retired())return true;
#if defined(__ANDROID__)
        for(auto* list:{&acquires,&releases})for(const auto& transition:*list)
            if(transition&&transition->may_be_pending())return true;
#endif
        return false;
    }
    MgcStatus service(bool initial=false) {
        wait_budget=initial?100000000:0;
        auto* previous=active_ticket;active_ticket=ticket;
        struct ResetTicket { MgcTicket* previous; ~ResetTicket(){active_ticket=previous;} } reset_ticket{previous};
        auto status=stage==DONE&&cleaned?MGC_SUCCESS:
            domain_is_lost(domain)?try_with_ncnn_domain_owner_retire(owner,run,this):try_with_ncnn_domain_owner(owner,run,this);
        if(stage==DONE&&cleaned){
            const auto released=ncnn_domain_owner_release(owner);
            if(released==MGC_SUCCESS)owner=nullptr;
            else status=released;
        }
        active_ticket=previous;
        if(owner)return status==MGC_SUCCESS?MGC_BUSY:status;
        // No device/owner scope is held here. Dropping borrowed content refs
        // may call producer code, so do it only after serialized GPU cleanup.
        for(auto* lease:detached)mgc_lease_release(lease);detached.clear();
        for(auto* lease:pinned)mgc_lease_release(lease);pinned.clear();
        keepalive.reset();
        return MGC_SUCCESS;
    }
    static MgcStatus poll(void* pointer,MgcTicket* ticket) {
        auto state=std::static_pointer_cast<OwnedGpuBatch>(ticket->gpu_batch);
        if(!state||state.get()!=pointer)return MGC_INTERNAL_ERROR;
        const auto status=state->service();
        if(status!=MGC_SUCCESS)return status;
        if(state->failed){ticket->deferred_failure=state->failure;ticket->resume_allowed=false;}
        ticket->gpu_batch.reset();
        return MGC_SUCCESS;
    }
};
}
bool ncnn_gpu_scope_deferred(){return last_owned_scope_deferred;}
MgcStatus with_ncnn_gpu_owned(MgcDomain *domain,MgcLease *const *leases,uint32_t count,
    std::shared_ptr<void> keepalive,NcnnGpuFn fn,void* context) {
    last_owned_scope_deferred=false;
    auto* ticket=active_ticket;
    if(!domain||!fn||!keepalive||!ticket||count>64||(count&&!leases))return MGC_INVALID_ARGUMENT;
    if(domain!=ticket->domain)return MGC_INCOMPATIBLE_DOMAIN;
    if(!domain->device)return MGC_UNSUPPORTED_CAPABILITY;
    if(ticket->deferred||ticket->gpu_batch)return MGC_BUSY;
    std::shared_ptr<OwnedGpuBatch> state;
    try{
        state=std::make_shared<OwnedGpuBatch>();
        state->domain=domain;state->ticket=ticket;state->record=fn;state->context=context;state->keepalive=std::move(keepalive);
        if(count)state->requested.assign(leases,leases+count);
        auto status=create_ncnn_domain_owner(domain,&state->owner,ticket);
        if(status)return status;
        ticket->gpu_batch=state;
        status=state->service(true);
        if(status==MGC_SUCCESS){ticket->gpu_batch.reset();return state->failed?state->failure:MGC_SUCCESS;}
        const auto cause=state->failed?state->failure:MGC_TIMEOUT;
        auto deferred=mgc_private_ticket_defer(ticket,OwnedGpuBatch::poll,state.get(),state->failed?0u:1u,cause);
        last_owned_scope_deferred=deferred==MGC_SUCCESS;
        return deferred==MGC_SUCCESS?MGC_BUSY:deferred;
    }catch(const std::bad_alloc&){
        if(state&&state->owner){state->fail(MGC_OUT_OF_MEMORY);last_owned_scope_deferred=mgc_private_ticket_defer(ticket,OwnedGpuBatch::poll,state.get(),0,MGC_OUT_OF_MEMORY)==MGC_SUCCESS;return last_owned_scope_deferred?MGC_BUSY:MGC_INTERNAL_ERROR;}
        return MGC_OUT_OF_MEMORY;
    }catch(...){
        if(state&&state->owner){state->fail(MGC_INTERNAL_ERROR);last_owned_scope_deferred=mgc_private_ticket_defer(ticket,OwnedGpuBatch::poll,state.get(),0,MGC_INTERNAL_ERROR)==MGC_SUCCESS;return last_owned_scope_deferred?MGC_BUSY:MGC_INTERNAL_ERROR;}
        return MGC_INTERNAL_ERROR;
    }
}
