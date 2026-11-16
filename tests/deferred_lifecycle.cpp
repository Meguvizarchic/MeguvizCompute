#include <meguviz/compute_job.h>
#if MGC_DEFERRED_GPU
#include <meguviz/backend_ncnn_sdk.hpp>
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#define REQUIRE(x) do {if(!(x))throw std::runtime_error(#x);}while(0)
struct Context {
    std::atomic<bool> entered{false},ready{false};
    std::atomic<int> refs{0},callbacks{0},resumes{0},finalized{0};
    MgcCompletion completion{};
};
static void retain(void* p){++static_cast<Context*>(p)->refs;}
static void release(void* p){--static_cast<Context*>(p)->refs;}
static void completed(void* p,const MgcCompletion* c){auto& v=*static_cast<Context*>(p);v.completion=*c;++v.callbacks;}
static void finalized(void* p){++static_cast<Context*>(p)->finalized;}
static MgcStatus poll_host(void* p,MgcTicket*){return static_cast<Context*>(p)->ready?MGC_SUCCESS:MGC_BUSY;}
static MgcStatus resume_host(void* p,MgcTicket*){++static_cast<Context*>(p)->resumes;return MGC_SUCCESS;}
static MgcStatus execute_host(void* p,MgcTicket* t){auto& c=*static_cast<Context*>(p);c.entered=true;return mgc_private_ticket_defer(t,poll_host,p,1,MGC_TIMEOUT)==MGC_SUCCESS?MGC_BUSY:MGC_INTERNAL_ERROR;}
static void entered(Context& c){auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);while(!c.entered&&std::chrono::steady_clock::now()<end)std::this_thread::sleep_for(std::chrono::milliseconds(1));REQUIRE(c.entered);}
static MgcRuntime* runtime(uint32_t provider){MgcRuntimeDesc d{};d.header=MGC_HEADER_INIT(MgcRuntimeDesc);d.provider=provider;d.memory_budget_bytes=1024*1024;MgcRuntime* r=nullptr;REQUIRE(mgc_runtime_create(&d,&r)==MGC_SUCCESS);return r;}
static MgcDomain* domain(MgcRuntime* r){MgcDomainDesc d{};d.header=MGC_HEADER_INIT(MgcDomainDesc);MgcDomain* v=nullptr;REQUIRE(mgc_domain_acquire(r,&d,&v)==MGC_SUCCESS);return v;}
static MgcSession* session(MgcDomain* d){MgcSessionDesc v{};v.header=MGC_HEADER_INIT(MgcSessionDesc);v.max_in_flight=1;MgcSession* s=nullptr;REQUIRE(mgc_session_create(d,&v,&s)==MGC_SUCCESS);return s;}
static MgcHostJobDesc job(Context& c){MgcHostJobDesc d{};d.header=MGC_HEADER_INIT(MgcHostJobDesc);d.context=&c;d.retain_context=retain;d.release_context=release;d.completion=completed;d.finalize=finalized;return d;}
static void host_cases(){
    auto* r=runtime(MGC_PROVIDER_CPU_REFERENCE);auto* d=domain(r);
    for(bool cancel:{false,true}){
        auto* s=session(d);Context c;auto desc=job(c);desc.execute=execute_host;desc.resume=resume_host;MgcTicket* t=nullptr;
        REQUIRE(mgc_private_submit_host_job(s,&desc,&t)==MGC_ACCEPTED);entered(c);
        REQUIRE(mgc_session_drain(s,0)==MGC_TIMEOUT&&c.callbacks==0&&c.refs==1);
        MgcCompletion state{};state.header=MGC_HEADER_INIT(MgcCompletion);REQUIRE(mgc_ticket_query(t,&state)==MGC_BUSY&&!state.execution_retired);
        if(cancel)REQUIRE(mgc_ticket_cancel(t)==MGC_SUCCESS);
        REQUIRE(mgc_session_close(s)==MGC_BUSY);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));REQUIRE(c.callbacks==0&&c.finalized==0);
        c.ready=true;REQUIRE(mgc_session_drain(s,2000000000)==MGC_SUCCESS);
        REQUIRE(c.callbacks==1&&c.finalized==1&&c.refs==0&&c.resumes==(cancel?0:1));
        REQUIRE(c.completion.status==(cancel?MGC_CANCELLED:MGC_SUCCESS)&&c.completion.execution_retired&&c.completion.host_finalized);
        mgc_ticket_release(t);REQUIRE(mgc_session_close(s)==MGC_SUCCESS);
    }
    auto* s=session(d);Context c;auto desc=job(c);desc.execute=execute_host;
    for(auto field:{std::pair<size_t,size_t>{offsetof(MgcHostJobDesc,resume),sizeof(desc.resume)},
                    {offsetof(MgcHostJobDesc,finalize),sizeof(desc.finalize)}})
        for(size_t cut=1;cut<field.second;++cut){desc.header.struct_size=uint32_t(field.first+cut);MgcTicket* t=nullptr;REQUIRE(mgc_private_submit_host_job(s,&desc,&t)==MGC_UNSUPPORTED_ABI&&!t);}
    REQUIRE(mgc_session_close(s)==MGC_SUCCESS);mgc_domain_release(d);REQUIRE(mgc_runtime_close(r)==MGC_SUCCESS);mgc_runtime_release(r);
    std::puts("PASS controlled host defer/resume/cancel, no early callback, exact finalization and old-prefix guards");
}
#if MGC_DEFERRED_GPU
static PFN_vkWaitForFences real_wait;
static PFN_vkQueueSubmit real_submit;
static std::atomic<int> fault{0};
static VKAPI_ATTR VkResult VKAPI_CALL wait_hook(VkDevice d,uint32_t n,const VkFence* f,VkBool32 all,uint64_t timeout){return fault.load()==1?VK_TIMEOUT:real_wait(d,n,f,all,timeout);}
static VKAPI_ATTR VkResult VKAPI_CALL submit_hook(VkQueue q,uint32_t n,const VkSubmitInfo* info,VkFence f){return fault.load()==2?VK_ERROR_OUT_OF_DEVICE_MEMORY:real_submit(q,n,info,f);}
struct GpuWork {ncnn::Mat upload,download;};
struct GpuContext:Context {
    MgcDomain* domain=nullptr;MgcLease* output=nullptr;bool hidden=false;
    std::shared_ptr<GpuWork> work;std::array<float,4> actual{};
    ncnn::VkBufferMemory* metadata=nullptr;VkAccessFlags before_access=0;VkPipelineStageFlags before_stage=0;
};
static GpuContext& gpu_context(void* p){return *static_cast<GpuContext*>(static_cast<Context*>(p));}
static MgcStatus gpu_record(void* p,meguviz::NcnnGpuScope& scope){
    auto& c=gpu_context(p);auto& w=*c.work;
    c.metadata=scope.views[0].buffer->data;c.before_access=c.metadata->access_flags;c.before_stage=c.metadata->stage_flags;
    ncnn::Option opt;opt.blob_vkallocator=opt.workspace_vkallocator=scope.blob_allocator;opt.staging_vkallocator=scope.staging_allocator;
    scope.command->record_clone(w.upload,*scope.views[0].buffer,opt);
    scope.command->record_clone(*scope.views[0].buffer,w.download,opt);
    if(c.hidden){scope.command->submit_and_wait();return MGC_INTERNAL_ERROR;}
    return MGC_SUCCESS;
}
static MgcStatus gpu_resume(void* p,MgcTicket*){auto& c=gpu_context(p);++c.resumes;if(c.work->download.empty())return MGC_INTERNAL_ERROR;std::memcpy(c.actual.data(),c.work->download.data,sizeof(c.actual));return MGC_SUCCESS;}
static MgcStatus gpu_execute(void* p,MgcTicket* t){auto& c=gpu_context(p);c.entered=true;auto status=meguviz::with_ncnn_gpu_owned(c.domain,&c.output,1,c.work,gpu_record,p);return status?status:gpu_resume(p,t);}
static void gpu_finalized(void* p){auto& c=gpu_context(p);c.work.reset();++c.finalized;}
struct SyntheticContext:Context {MgcDomain* domain=nullptr;MgcSession* session=nullptr;MgcStatus nested=MGC_INTERNAL_ERROR,close=MGC_INTERNAL_ERROR;};
static MgcStatus synthetic_record(void*,meguviz::NcnnGpuScope& scope){return scope.device&&scope.command?MGC_SUCCESS:MGC_INTERNAL_ERROR;}
static MgcStatus synthetic_execute(void*,MgcTicket*){return MGC_SUCCESS;}
static void synthetic_completed(void* p,const MgcCompletion* result){
    auto& c=*static_cast<SyntheticContext*>(static_cast<Context*>(p));c.completion=*result;
    c.close=mgc_session_close(c.session);
    // Completion runs on the original session worker after active_ticket is
    // cleared; the synthetic ticket must not borrow that session's pin.
    c.nested=meguviz::with_ncnn_gpu(c.domain,nullptr,0,synthetic_record,nullptr);++c.callbacks;
}
static void gpu_cases(){
    auto* r=runtime(MGC_PROVIDER_NCNN_VULKAN);auto* d=domain(r);
    real_wait=ncnn::vkWaitForFences;real_submit=ncnn::vkQueueSubmit;ncnn::vkWaitForFences=wait_hook;ncnn::vkQueueSubmit=submit_hook;
    for(int mode=0;mode<5;++mode){
        auto* s=session(d);GpuContext c;c.domain=d;c.hidden=mode==3;c.work=std::make_shared<GpuWork>();
        c.work->upload.create(4,size_t(4));const std::array<float,4> expected{1.25f,-2.5f,3.75f,4.5f};std::memcpy(c.work->upload.data,expected.data(),sizeof(expected));
        std::weak_ptr<GpuWork> lifetime=c.work;
        MgcResourceDesc resource{};resource.header=MGC_HEADER_INIT(MgcResourceDesc);resource.capacity_bytes=16;resource.format=MGC_FORMAT_FLOAT32;resource.width=4;resource.height=1;resource.row_stride_bytes=16;resource.usage=MGC_RESOURCE_GPU_BUFFER;
        MgcResource* storage=nullptr;REQUIRE(mgc_resource_create(d,&resource,&storage)==MGC_SUCCESS);
        MgcLeaseDesc lease{};lease.header=MGC_HEADER_INIT(MgcLeaseDesc);lease.byte_length=16;lease.content_id=uint64_t(mode+1);lease.access=MGC_ACCESS_WRITE;REQUIRE(mgc_lease_create(storage,&lease,&c.output)==MGC_SUCCESS);
        auto desc=job(c);desc.execute=gpu_execute;desc.resume=gpu_resume;desc.finalize=gpu_finalized;desc.required_flags=MGC_HOST_JOB_EXPLICIT_GPU_SCOPES;desc.leases=&c.output;desc.lease_count=1;
        fault=mode==4?2:mode?1:0;mgc_lease_retain(c.output);MgcTicket* t=nullptr;REQUIRE(mgc_private_submit_host_job(s,&desc,&t)==MGC_ACCEPTED);entered(c);
        if(mode>0&&mode<4){
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            REQUIRE(mgc_session_drain(s,0)==MGC_TIMEOUT&&c.callbacks==0&&!lifetime.expired());
            MgcCompletion state{};state.header=MGC_HEADER_INIT(MgcCompletion);REQUIRE(mgc_ticket_query(t,&state)==MGC_BUSY&&!state.execution_retired);
            if(mode==2)REQUIRE(mgc_ticket_cancel(t)==MGC_SUCCESS);
            REQUIRE(mgc_session_close(s)==MGC_BUSY);REQUIRE(c.callbacks==0);fault=0;
        }
        REQUIRE(mgc_session_drain(s,5000000000)==MGC_SUCCESS);
        const auto status=mode==2?MGC_CANCELLED:mode==3?MGC_TIMEOUT:mode==4?MGC_OUT_OF_MEMORY:MGC_SUCCESS;
        REQUIRE(c.callbacks==1&&c.finalized==1&&c.refs==0&&lifetime.expired());
        REQUIRE(c.completion.status==status&&c.completion.execution_retired&&c.completion.host_finalized);
        REQUIRE(c.resumes==((mode==0||mode==1)?1:0));
        if(mode<2)REQUIRE(c.actual==expected);
        if(mode==4)REQUIRE(meguviz::with_ncnn_domain(d,[](void* p,const ncnn::VulkanDevice*)->MgcStatus{auto& c=*static_cast<GpuContext*>(p);return c.metadata->access_flags==c.before_access&&c.metadata->stage_flags==c.before_stage?MGC_SUCCESS:MGC_INTERNAL_ERROR;},&c)==MGC_SUCCESS);
        fault=0;mgc_ticket_release(t);mgc_lease_release(c.output);mgc_resource_release(storage);REQUIRE(mgc_session_close(s)==MGC_SUCCESS);
    }
    REQUIRE(meguviz::with_ncnn_gpu(d,nullptr,0,synthetic_record,nullptr)==MGC_SUCCESS);
    {
        auto* s=session(d);SyntheticContext c;c.domain=d;c.session=s;auto desc=job(c);
        desc.execute=synthetic_execute;desc.completion=synthetic_completed;desc.required_flags=MGC_HOST_JOB_EXPLICIT_GPU_SCOPES;
        MgcTicket* ticket=nullptr;REQUIRE(mgc_private_submit_host_job(s,&desc,&ticket)==MGC_ACCEPTED);
        REQUIRE(mgc_session_drain(s,5000000000)==MGC_SUCCESS);
        REQUIRE(c.callbacks==1&&c.completion.status==MGC_SUCCESS&&c.nested==MGC_SUCCESS&&c.close==MGC_BUSY);
        REQUIRE(c.refs==0&&c.finalized==1);mgc_ticket_release(ticket);REQUIRE(mgc_session_close(s)==MGC_SUCCESS);
    }
    ncnn::vkWaitForFences=real_wait;ncnn::vkQueueSubmit=real_submit;
    mgc_domain_release(d);REQUIRE(mgc_runtime_close(r)==MGC_SUCCESS);mgc_runtime_release(r);
    std::puts("PASS actual Vulkan owned copy, deferred resume/cancel, incomplete callback failure and no-submit metadata rollback");
}
#endif
int main(int argc,char** argv){try{
    if(argc==2&&std::strcmp(argv[1],"--host")==0){host_cases();return 0;}
#if MGC_DEFERRED_GPU
    gpu_cases();return 0;
#else
    return 77;
#endif
}catch(const std::exception& error){std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}}
