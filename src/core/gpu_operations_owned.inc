// Included in the implementation namespace after the common GPU shaders.
// One accepted public operation owns this state until finalization. Stages are
// advanced before recording so a retired deferred batch is never recorded twice.
struct GpuOperationWork {
    MgcDomain *domain;
    MgcSubmitDesc desc;
    int stage=0;
    bool callback_entered=false;
    MgcStatus terminal=MGC_SUCCESS;
    uint64_t input_bytes=0,output_bytes=0;
    uint32_t count=0,groups=0;
    MgcLease* gpu_leases[2]{};
    uint32_t gpu_lease_count=0;
    int input_view=-1,output_view=-1;
    ncnn::VkMat input,output,partial,calculated;
    ncnn::Mat upload,download,output_seed,summaries,statistics_upload;
    MgcStatisticsF32 statistics{};
    std::unique_ptr<ncnn::Net> network;
    std::unique_ptr<ncnn::VkTransfer> transfer;
    std::unique_ptr<ncnn::Extractor> extractor;
    alignas(4) unsigned char empty_model[4]{};

    explicit GpuOperationWork(MgcTicket *ticket):domain(ticket->domain),desc(ticket->desc) {
        mgc_domain_retain(domain);
        if(desc.input->resource->desc.usage){input_view=int(gpu_lease_count);gpu_leases[gpu_lease_count++]=desc.input;}
        if(desc.output->resource->desc.usage){output_view=int(gpu_lease_count);gpu_leases[gpu_lease_count++]=desc.output;}
    }
    ~GpuOperationWork() {
        // The ticket's owned scope must have retired its main and auxiliary
        // commands before dropping this context. No lease callbacks run here.
        meguviz::with_ncnn_domain_cleanup(domain,[](void* p,const ncnn::VulkanDevice*) -> MgcStatus {
            auto& w=*static_cast<GpuOperationWork*>(p);
            w.extractor.reset();w.transfer.reset();w.network.reset();
            w.input.release();w.output.release();w.partial.release();w.calculated.release();
            w.upload.release();w.download.release();w.output_seed.release();w.summaries.release();w.statistics_upload.release();
            return MGC_SUCCESS;
        },this);
        mgc_domain_release(domain);
    }
    MgcStatus initialize() {
        auto* a=desc.input->resource;auto* b=desc.output->resource;
        if(desc.operation!=MGC_OPERATION_COPY&&desc.operation!=MGC_OPERATION_RELU_F32&&
            desc.operation!=MGC_OPERATION_NORMALIZE_F32&&desc.operation!=MGC_OPERATION_STATISTICS_F32&&
            desc.operation!=MGC_OPERATION_RESIZE_RGB8)return MGC_UNSUPPORTED_CAPABILITY;
        const bool kernel=desc.operation==MGC_OPERATION_NORMALIZE_F32||desc.operation==MGC_OPERATION_STATISTICS_F32||desc.operation==MGC_OPERATION_RESIZE_RGB8;
        if(a->desc.usage==MGC_RESOURCE_ANDROID_AHB||b->desc.usage==MGC_RESOURCE_ANDROID_AHB)
            return MGC_UNSUPPORTED_CAPABILITY;
        if(desc.input->desc.byte_offset%4||desc.output->desc.byte_offset%4)
            return kernel?MGC_UNSUPPORTED_CAPABILITY:MGC_INVALID_ARGUMENT;
        const auto alignment=domain->device->info.buffer_offset_alignment();
        if((a->desc.usage&&desc.input->desc.byte_offset%alignment)||(b->desc.usage&&desc.output->desc.byte_offset%alignment))
            return MGC_UNSUPPORTED_CAPABILITY;
        input_bytes=desc.operation==MGC_OPERATION_RESIZE_RGB8?a->desc.row_stride_bytes*a->desc.height:desc.byte_count;
        output_bytes=desc.operation==MGC_OPERATION_RESIZE_RGB8?b->desc.row_stride_bytes*b->desc.height:
            desc.operation==MGC_OPERATION_STATISTICS_F32?sizeof(MgcStatisticsF32):desc.byte_count;
        if(!input_bytes||input_bytes>uint64_t(INT32_MAX)*4||output_bytes>uint64_t(INT32_MAX)*4)return MGC_INVALID_ARGUMENT;
        if(desc.operation!=MGC_OPERATION_RESIZE_RGB8&&(desc.byte_count%4||desc.byte_count/4>INT32_MAX))return MGC_INVALID_ARGUMENT;
        if(desc.operation==MGC_OPERATION_RESIZE_RGB8&&
            (input_bytes>UINT32_MAX||output_bytes>UINT32_MAX||a->desc.width>INT32_MAX||a->desc.height>INT32_MAX||
             b->desc.width>INT32_MAX||b->desc.height>INT32_MAX))return MGC_UNSUPPORTED_CAPABILITY;
        count=uint32_t(input_bytes/4);groups=(count+255u)/256u;
        return MGC_SUCCESS;
    }
    static MgcStatus retire_model(void* p,uint64_t wait_ns) {
        auto& w=*static_cast<GpuOperationWork*>(p);
        if(domain_is_lost(w.domain)){
            service_device_loss(w.domain);
            if(!device_loss_retired(w.domain))return MGC_BUSY;
            if(w.transfer)w.transfer->retire_device_lost();
            return MGC_SUCCESS;
        }
        if(!w.transfer||w.transfer->transfer_batches_retired())return MGC_SUCCESS;
        w.transfer->wait_transfer_retirement(wait_ns);
        if(w.transfer->last_submit_result()==VK_ERROR_DEVICE_LOST||w.transfer->last_wait_result()==VK_ERROR_DEVICE_LOST)
            report_device_lost(w.domain);
        return w.transfer->transfer_batches_retired()?MGC_SUCCESS:MGC_BUSY;
    }
    MgcStatus prepare_input(meguviz::NcnnGpuScope& scope,const ncnn::Option& opt) {
        if(input_view>=0){
            if(!scope.views[size_t(input_view)].buffer)return MGC_UNSUPPORTED_CAPABILITY;
            input=gpu_slice(*scope.views[size_t(input_view)].buffer,int(desc.input->desc.byte_offset/4),int((input_bytes+3)/4));
        } else {
            upload.create(int((input_bytes+3)/4),size_t(4));if(upload.empty())return MGC_OUT_OF_MEMORY;
            std::memset(upload.data,0,upload.total()*4);
            std::memcpy(upload.data,desc.input->resource->host_data()+desc.input->desc.byte_offset,size_t(input_bytes));
            scope.command->record_clone(upload,input,opt);if(input.empty())return MGC_OUT_OF_MEMORY;
        }
        return MGC_SUCCESS;
    }
    MgcStatus select_output(meguviz::NcnnGpuScope& scope,uint64_t bytes) {
        if(output_view<0)return MGC_SUCCESS;
        if(!scope.views[size_t(output_view)].buffer)return MGC_UNSUPPORTED_CAPABILITY;
        output=gpu_slice(*scope.views[size_t(output_view)].buffer,int(desc.output->desc.byte_offset/4),int((bytes+3)/4));
        return MGC_SUCCESS;
    }
    MgcStatus load_network(meguviz::NcnnGpuScope& scope,const ncnn::Option& opt) {
        // Register the auxiliary owner BEFORE the model loader may submit.
        scope.retire_aux=retire_model;scope.aux_context=this;
        network=std::make_unique<ncnn::Net>();network->set_vulkan_device(scope.device);network->opt=opt;
        if(network->load_param_mem("7767517\n2 2\nInput input 0 1 in0\nReLU relu 1 1 in0 out0\n"))return MGC_INTERNAL_ERROR;
        transfer=std::make_unique<ncnn::VkTransfer>(scope.device);
        if(!transfer->is_valid())return MGC_INTERNAL_ERROR;
        const unsigned char* cursor=empty_model;
        ncnn::DataReaderFromMemory reader(cursor,sizeof(empty_model));
        if(network->load_model_with_transfer(reader,transfer.get(),100000000)){
            if(transfer->last_submit_result()==VK_ERROR_DEVICE_LOST||transfer->last_wait_result()==VK_ERROR_DEVICE_LOST){report_device_lost(domain);return MGC_DEVICE_LOST;}
            if(transfer->last_submit_result()==VK_ERROR_OUT_OF_HOST_MEMORY||transfer->last_submit_result()==VK_ERROR_OUT_OF_DEVICE_MEMORY||
                transfer->last_wait_result()==VK_ERROR_OUT_OF_HOST_MEMORY||transfer->last_wait_result()==VK_ERROR_OUT_OF_DEVICE_MEMORY)return MGC_OUT_OF_MEMORY;
            return MGC_INTERNAL_ERROR;
        }
        if(!network->opt.use_vulkan_compute||network->layers().size()!=2||!network->layers()[1]->support_vulkan)
            return MGC_UNSUPPORTED_CAPABILITY;
        return MGC_SUCCESS;
    }
    static MgcStatus first_batch(void* p,meguviz::NcnnGpuScope& scope) {
        auto& w=*static_cast<GpuOperationWork*>(p);w.callback_entered=true;
        if(domain_is_lost(w.domain))return MGC_DEVICE_LOST;
        const auto opt=gpu_options(w.domain);auto& command=*scope.command;
        auto status=w.prepare_input(scope,opt);if(status)return status;
        if(w.desc.operation==MGC_OPERATION_COPY){
            status=w.select_output(scope,w.output_bytes);if(status)return status;
            if(w.output_view>=0)command.record_clone(w.input,w.output,opt);
            else command.record_clone(w.input,w.download,opt);
            return MGC_SUCCESS;
        }
        if(w.desc.operation==MGC_OPERATION_RESIZE_RGB8){
            auto* a=w.desc.input->resource;auto* b=w.desc.output->resource;
            status=w.select_output(scope,w.output_bytes);if(status)return status;
            if(w.output_view<0){
                w.output_seed.create(int((w.output_bytes+3)/4),size_t(4));if(w.output_seed.empty())return MGC_OUT_OF_MEMORY;
                std::memset(w.output_seed.data,0,w.output_seed.total()*4);
                std::memcpy(w.output_seed.data,b->host_data()+w.desc.output->desc.byte_offset,size_t(w.output_bytes));
                command.record_clone(w.output_seed,w.output,opt);if(w.output.empty())return MGC_OUT_OF_MEMORY;
            }
            auto* pipeline=kernel_pipeline(w.domain,resize_rgb_shader);if(!pipeline)return MGC_INTERNAL_ERROR;
            const auto words=uint32_t((w.output_bytes+3)/4);
            const uint32_t values[7]={a->desc.width,a->desc.height,uint32_t(a->desc.row_stride_bytes),b->desc.width,b->desc.height,uint32_t(b->desc.row_stride_bytes),words};
            std::vector<ncnn::vk_constant_type> constants(7);for(size_t i=0;i<7;++i)constants[i].i=int(values[i]);
            command.record_pipeline(pipeline,{w.input,w.output},{},constants,ncnn::Mat(int(std::min(words,w.domain->device->info.max_workgroup_count_x())),1,1,static_cast<void*>(nullptr)));
            if(w.output_view<0)command.record_clone(w.output,w.download,opt);
            return MGC_SUCCESS;
        }
        w.partial.create(int(w.groups*8),size_t(4),1,scope.blob_allocator);
        auto* pipeline=kernel_pipeline(w.domain,reduce_f32_shader);if(!pipeline||w.partial.empty())return MGC_OUT_OF_MEMORY;
        std::vector<ncnn::vk_constant_type> constants(2);constants[0].i=int(w.count);constants[1].i=int(w.groups);
        command.record_pipeline(pipeline,{w.input,w.partial},{},constants,ncnn::Mat(int(std::min(w.groups,w.domain->device->info.max_workgroup_count_x())),1,1,static_cast<void*>(nullptr)));
        command.record_clone(w.partial,w.summaries,opt);return MGC_SUCCESS;
    }
    MgcStatus summarize() {
        if(summaries.empty())return MGC_OUT_OF_MEMORY;
        statistics={};statistics.header=MGC_HEADER_INIT(MgcStatisticsF32);statistics.count=count;
        if(desc.operation==MGC_OPERATION_RELU_F32){
            for(uint32_t group=0;group<groups;++group)statistics.finite_count+=uint64_t(static_cast<const float*>(summaries.data)[group*8+3]);
            return MGC_SUCCESS;
        }
        for(uint32_t group=0;group<groups;++group){
            const auto* v=static_cast<const float*>(summaries.data)+group*8;
            if(v[3]>0){
                statistics.minimum=statistics.finite_count?std::min(statistics.minimum,v[0]):v[0];
                statistics.maximum=statistics.finite_count?std::max(statistics.maximum,v[1]):v[1];
                const auto magnitude=std::max(std::fabs(double(v[0])),std::fabs(double(v[1])));
                statistics.sum+=(double(v[2])+double(v[4]))*(magnitude>double(1e36f)?magnitude:1.0);
                statistics.finite_count+=uint64_t(v[3]);
            }
        }
        return MGC_SUCCESS;
    }
    static MgcStatus final_batch(void* p,meguviz::NcnnGpuScope& scope) {
        auto& w=*static_cast<GpuOperationWork*>(p);w.callback_entered=true;
        if(domain_is_lost(w.domain))return MGC_DEVICE_LOST;
        const auto opt=gpu_options(w.domain);auto& command=*scope.command;
        auto status=w.select_output(scope,w.output_bytes);if(status)return status;
        if(w.desc.operation==MGC_OPERATION_STATISTICS_F32){
            w.statistics_upload.create(int(sizeof(w.statistics)/4),size_t(4));if(w.statistics_upload.empty())return MGC_OUT_OF_MEMORY;
            std::memset(w.statistics_upload.data,0,w.statistics_upload.total()*w.statistics_upload.elemsize);
            std::memcpy(w.statistics_upload.data,&w.statistics,sizeof(w.statistics));
            command.record_clone(w.statistics_upload,w.output,opt);return MGC_SUCCESS;
        }
        if(w.desc.operation==MGC_OPERATION_RELU_F32){
            if(w.output_view<0)w.output.create(int(w.count),size_t(4),1,scope.blob_allocator);
            if(w.output.empty())return MGC_OUT_OF_MEMORY;
            auto* pipeline=kernel_pipeline(w.domain,relu_f32_shader);if(!pipeline)return MGC_INTERNAL_ERROR;
            std::vector<ncnn::vk_constant_type> constants(1);constants[0].i=int(w.count);
            command.record_pipeline(pipeline,{w.input,w.output},{},constants,ncnn::Mat(int(std::min(w.count,w.domain->device->info.max_workgroup_count_x())),1,1,static_cast<void*>(nullptr)));
            if(w.output_view<0)command.record_clone(w.output,w.download,opt);
            return MGC_SUCCESS;
        }
        if(w.output_view<0)w.output.create(int(w.count),size_t(4),1,scope.blob_allocator);
        if(w.output.empty())return MGC_OUT_OF_MEMORY;
        auto* pipeline=kernel_pipeline(w.domain,normalize_f32_shader);if(!pipeline)return MGC_INTERNAL_ERROR;
        std::vector<ncnn::vk_constant_type> constants(3);constants[0].i=int(w.count);constants[1].f=w.statistics.minimum;constants[2].f=w.statistics.maximum;
        command.record_pipeline(pipeline,{w.input,w.output},{},constants,ncnn::Mat(int(std::min(w.count,w.domain->device->info.max_workgroup_count_x())),1,1,static_cast<void*>(nullptr)));
        if(w.output_view<0)command.record_clone(w.output,w.download,opt);
        return MGC_SUCCESS;
    }
    MgcStatus copy_download() {
        if(output_view>=0)return MGC_SUCCESS;
        if(download.empty())return MGC_OUT_OF_MEMORY;
        std::memcpy(desc.output->resource->host_data()+desc.output->desc.byte_offset,download.data,size_t(output_bytes));
        return MGC_SUCCESS;
    }
};

MgcStatus run_gpu_operation_owned(MgcTicket *ticket) {
    if(domain_is_lost(ticket->domain))return MGC_DEVICE_LOST;
    auto state=ticket->operation_state?std::static_pointer_cast<GpuOperationWork>(ticket->operation_state):std::make_shared<GpuOperationWork>(ticket);
    auto& work=*state;
    if(!ticket->operation_state){auto status=work.initialize();if(status)return status;ticket->operation_state=state;}
    if(work.terminal)return work.terminal;
    if(work.stage==0){
        work.stage=1;work.callback_entered=false;
        const auto status=meguviz::with_ncnn_gpu_owned(work.domain,work.gpu_leases,work.gpu_lease_count,state,GpuOperationWork::first_batch,&work);
        if(status){if(!meguviz::ncnn_gpu_scope_deferred()){if(!work.callback_entered)work.stage=0;else work.terminal=status;}return status;}
    }
    if(work.stage==1){
        if(work.desc.operation==MGC_OPERATION_COPY||work.desc.operation==MGC_OPERATION_RESIZE_RGB8){auto status=work.copy_download();if(status)return status;work.stage=4;return MGC_SUCCESS;}
        auto status=work.summarize();if(status)return status;
        if(work.desc.operation==MGC_OPERATION_STATISTICS_F32&&work.output_view<0){
            std::memcpy(work.desc.output->resource->host_data()+work.desc.output->desc.byte_offset,&work.statistics,sizeof(work.statistics));work.stage=4;return MGC_SUCCESS;
        }
        if(work.desc.operation!=MGC_OPERATION_STATISTICS_F32&&work.statistics.finite_count!=work.statistics.count)return MGC_INVALID_ARGUMENT;
        work.stage=2;
    }
    if(work.stage==2){
        work.stage=3;work.callback_entered=false;
        const auto status=meguviz::with_ncnn_gpu_owned(work.domain,work.gpu_leases,work.gpu_lease_count,state,GpuOperationWork::final_batch,&work);
        if(status){if(!meguviz::ncnn_gpu_scope_deferred()){if(!work.callback_entered)work.stage=2;else work.terminal=status;}return status;}
    }
    if(work.stage==3){auto status=work.copy_download();if(status)return status;work.stage=4;}
    return MGC_SUCCESS;
}
MgcStatus run_gpu_kernel(MgcTicket *ticket){return run_gpu_operation_owned(ticket);}
MgcStatus run_gpu_operation(MgcTicket *ticket){return run_gpu_operation_owned(ticket);}
