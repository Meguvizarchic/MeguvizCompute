#include <meguviz/compute.h>
#include <meguviz/backend_ncnn_sdk.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
static PFN_vkQueueSubmit real_submit=nullptr;
static PFN_vkDeviceWaitIdle real_idle=nullptr;
static std::atomic<bool> inject_loss{false},allow_idle{false};
static std::atomic<unsigned> injected{0},idle_attempts{0},real_idle_calls{0};
static VKAPI_ATTR VkResult VKAPI_CALL submit_hook(VkQueue queue,uint32_t count,const VkSubmitInfo* info,VkFence fence) {
    if(inject_loss.exchange(false)){
        const auto result=real_submit(queue,count,info,fence);
        if(result!=VK_SUCCESS)return result;
        ++injected;return VK_ERROR_DEVICE_LOST;
    }
    return real_submit(queue,count,info,fence);
}
static VKAPI_ATTR VkResult VKAPI_CALL idle_hook(VkDevice device) {
    ++idle_attempts;
    if(!allow_idle.load())return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++real_idle_calls;return real_idle(device);
}
struct Callback {std::atomic<int> calls{0};MgcCompletion value{};};
static void completed(void* context,const MgcCompletion* completion) {
    auto& result=*static_cast<Callback*>(context);result.value=*completion;++result.calls;
}
struct Memory {MgcResource* resource=nullptr;MgcLease* lease=nullptr;};
static Memory memory(MgcDomain* domain,bool gpu,uint64_t id,const std::array<float,4>* initial=nullptr) {
    Memory m;MgcResourceDesc resource{};resource.header=MGC_HEADER_INIT(MgcResourceDesc);
    resource.capacity_bytes=16;resource.format=MGC_FORMAT_FLOAT32;resource.width=4;resource.height=1;resource.row_stride_bytes=16;
    resource.usage=gpu?MGC_RESOURCE_GPU_BUFFER:0;
    CHECK(mgc_resource_create(domain,&resource,&m.resource)==MGC_SUCCESS);
    MgcLeaseDesc desc{};desc.header=MGC_HEADER_INIT(MgcLeaseDesc);desc.content_id=id;desc.byte_length=16;desc.access=MGC_ACCESS_WRITE;
    CHECK(mgc_lease_create(m.resource,&desc,&m.lease)==MGC_SUCCESS);
    if(initial){void* p=nullptr;uint64_t n=0;CHECK(mgc_lease_map(m.lease,&p,&n)==MGC_SUCCESS&&n==16);std::memcpy(p,initial->data(),16);CHECK(mgc_lease_unmap(m.lease)==MGC_SUCCESS);}
    return m;
}
static void make_read(Memory& memory) {
    MgcLease* read=nullptr;CHECK(mgc_lease_publish_read(memory.lease,&read)==MGC_SUCCESS);
    mgc_lease_release(memory.lease);memory.lease=read;
}
static void release(Memory& memory) {mgc_lease_release(memory.lease);memory.lease=nullptr;mgc_resource_release(memory.resource);memory.resource=nullptr;}
static MgcSession* session(MgcDomain* domain) {
    MgcSessionDesc desc{};desc.header=MGC_HEADER_INIT(MgcSessionDesc);desc.max_in_flight=1;
    MgcSession* result=nullptr;CHECK(mgc_session_create(domain,&desc,&result)==MGC_SUCCESS);return result;
}
static MgcStatus submit(MgcSession* session,MgcLease* input,MgcLease* output,Callback& callback,MgcTicket** ticket) {
    MgcSubmitDesc desc{};desc.header=MGC_HEADER_INIT(MgcSubmitDesc);desc.operation=MGC_OPERATION_COPY;
    desc.input=input;desc.output=output;desc.byte_count=16;desc.completion=completed;desc.context=&callback;
    mgc_lease_retain(input);mgc_lease_retain(output);
    const auto result=mgc_submit(session,&desc,ticket);
    if(result!=MGC_ACCEPTED){mgc_lease_release(input);mgc_lease_release(output);}
    return result;
}
static MgcDomainStateInfo state(MgcDomain* domain) {
    MgcDomainStateInfo info{};info.header=MGC_HEADER_INIT(MgcDomainStateInfo);CHECK(mgc_domain_state(domain,&info)==MGC_SUCCESS);return info;
}
int main(){
    try {
        MgcRuntimeDesc rd{};rd.header=MGC_HEADER_INIT(MgcRuntimeDesc);rd.provider=MGC_PROVIDER_NCNN_VULKAN;rd.memory_budget_bytes=16*1024*1024;
        MgcRuntime* runtime=nullptr;CHECK(mgc_runtime_create(&rd,&runtime)==MGC_SUCCESS);
        MgcDomainDesc dd{};dd.header=MGC_HEADER_INIT(MgcDomainDesc);MgcDomain *first=nullptr,*peer=nullptr,*fresh=nullptr;
        auto available=mgc_domain_acquire(runtime,&dd,&first);
        if(available==MGC_UNSUPPORTED_CAPABILITY){mgc_runtime_release(runtime);return 77;}
        CHECK(available==MGC_SUCCESS&&first);CHECK(mgc_domain_acquire(runtime,&dd,&peer)==MGC_SUCCESS);
        const auto initial=state(first),initial_peer=state(peer);CHECK(!initial.lost&&initial.device_epoch&&initial.device_epoch==initial_peer.device_epoch);
        auto* original=session(first);auto* losing=session(first);const std::array<float,4> input_values{1.25f,-2.5f,3.75f,4.5f};
        const std::array<float,4> sentinel{31.f,32.f,33.f,34.f};
        auto input=memory(first,false,1,&input_values);make_read(input);
        auto old_gpu=memory(first,true,1);auto failed_output=memory(first,false,1,&sentinel);
        Callback warm;MgcTicket* ticket=nullptr;CHECK(submit(original,input.lease,old_gpu.lease,warm,&ticket)==MGC_ACCEPTED);
        CHECK(mgc_session_drain(original,5000000000ULL)==MGC_SUCCESS&&warm.calls==1&&warm.value.status==MGC_SUCCESS);
        MgcCapability capability{};capability.header=MGC_HEADER_INIT(MgcCapability);
        CHECK(mgc_domain_capability(first,MGC_CAP_VULKAN_COMPUTE,&capability)==MGC_SUCCESS&&capability.enabled&&capability.probe_result==MGC_SUCCESS&&!capability.workload_qualified);
        mgc_ticket_release(ticket);ticket=nullptr;make_read(old_gpu);
        real_submit=ncnn::vkQueueSubmit;real_idle=ncnn::vkDeviceWaitIdle;ncnn::vkQueueSubmit=submit_hook;ncnn::vkDeviceWaitIdle=idle_hook;
        inject_loss=true;Callback failed;CHECK(submit(losing,input.lease,failed_output.lease,failed,&ticket)==MGC_ACCEPTED);
        auto until=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(!idle_attempts.load()&&std::chrono::steady_clock::now()<until)std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(injected==1&&idle_attempts>0&&failed.calls==0);
        CHECK(mgc_session_drain(losing,0)==MGC_TIMEOUT);
        MgcCompletion pending{};pending.header=MGC_HEADER_INIT(MgcCompletion);
        CHECK(mgc_ticket_query(ticket,&pending)==MGC_BUSY&&!pending.execution_retired);
        auto pending_a=state(first),pending_b=state(peer);
        CHECK(pending_a.lost&&pending_b.lost&&!pending_a.device_retired&&!pending_b.device_retired);
        capability.header=MGC_HEADER_INIT(MgcCapability);
        CHECK(mgc_domain_capability(first,MGC_CAP_VULKAN_COMPUTE,&capability)==MGC_SUCCESS&&capability.advertised&&!capability.enabled&&capability.probe_result==MGC_DEVICE_LOST);
        capability.header=MGC_HEADER_INIT(MgcCapability);
        CHECK(mgc_domain_capability(first,MGC_CAP_CPU_REFERENCE,&capability)==MGC_SUCCESS&&capability.enabled&&capability.probe_result==MGC_SUCCESS);
        CHECK(mgc_session_close(losing)==MGC_BUSY);
        allow_idle=true;
        CHECK(mgc_session_drain(losing,5000000000ULL)==MGC_SUCCESS);
        CHECK(real_idle_calls>0&&failed.calls==1&&failed.value.status==MGC_DEVICE_LOST&&failed.value.terminal==MGC_TERMINAL_DEVICE_LOST);
        CHECK(failed.value.execution_retired&&failed.value.host_finalized);
        auto lost=state(first),lost_peer=state(peer);CHECK(lost.lost&&lost.device_retired&&lost_peer.lost&&lost_peer.device_retired);
        MgcLease* forbidden=nullptr;CHECK(mgc_lease_publish_read(failed_output.lease,&forbidden)==MGC_DEVICE_LOST&&!forbidden);
        void* p=nullptr;uint64_t n=0;CHECK(mgc_lease_map(failed_output.lease,&p,&n)==MGC_DEVICE_LOST&&!p);
        Callback rejected;MgcTicket* no_ticket=nullptr;
        CHECK(submit(original,old_gpu.lease,failed_output.lease,rejected,&no_ticket)==MGC_DEVICE_LOST&&!no_ticket&&rejected.calls==0);
        CHECK(mgc_domain_acquire(runtime,&dd,&fresh)==MGC_SUCCESS&&fresh);
        const auto recovered=state(fresh);CHECK(!recovered.lost&&recovered.device_epoch&&recovered.device_epoch!=initial.device_epoch&&recovered.domain_id!=initial.domain_id);
        CHECK(state(first).lost&&state(peer).lost);
        auto* next=session(fresh);auto next_input=memory(fresh,false,1,&input_values);make_read(next_input);auto next_output=memory(fresh,false,1);
        Callback successful;MgcTicket* next_ticket=nullptr;CHECK(submit(next,next_input.lease,next_output.lease,successful,&next_ticket)==MGC_ACCEPTED);
        CHECK(mgc_session_drain(next,5000000000ULL)==MGC_SUCCESS&&successful.calls==1&&successful.value.status==MGC_SUCCESS);
        CHECK(mgc_lease_map(next_output.lease,&p,&n)==MGC_SUCCESS&&n==16&&std::memcmp(p,input_values.data(),16)==0);CHECK(mgc_lease_unmap(next_output.lease)==MGC_SUCCESS);
        // Explicitly replace the poisoned content version, then inspect the
        // untouched host buffer. This is never published as the failed output.
        mgc_lease_release(failed_output.lease);failed_output.lease=nullptr;
        MgcLeaseDesc replacement{};replacement.header=MGC_HEADER_INIT(MgcLeaseDesc);replacement.content_id=2;replacement.byte_length=16;replacement.access=MGC_ACCESS_WRITE;
        CHECK(mgc_lease_create(failed_output.resource,&replacement,&failed_output.lease)==MGC_SUCCESS);
        CHECK(mgc_lease_map(failed_output.lease,&p,&n)==MGC_SUCCESS&&std::memcmp(p,sentinel.data(),16)==0);CHECK(mgc_lease_unmap(failed_output.lease)==MGC_SUCCESS);
        CHECK(mgc_runtime_close(runtime)==MGC_BUSY); // old GPU read and domains still live
        ncnn::vkQueueSubmit=real_submit;ncnn::vkDeviceWaitIdle=real_idle;
        mgc_ticket_release(next_ticket);mgc_ticket_release(ticket);
        CHECK(mgc_session_close(next)==MGC_SUCCESS);CHECK(mgc_session_close(losing)==MGC_SUCCESS);CHECK(mgc_session_close(original)==MGC_SUCCESS);
        release(next_input);release(next_output);release(input);release(old_gpu);release(failed_output);
        mgc_domain_release(fresh);mgc_domain_release(peer);mgc_domain_release(first);
        CHECK(mgc_runtime_close(runtime)==MGC_SUCCESS);mgc_runtime_release(runtime);
        std::puts("PASS controlled DEVICE_LOST, real idle proof, retained old output, new device generation and actual recovered GPU copy");return 0;
    }catch(const std::exception& error){
        if(real_submit)ncnn::vkQueueSubmit=real_submit;if(real_idle)ncnn::vkDeviceWaitIdle=real_idle;
        std::fprintf(stderr,"FAIL %s\n",error.what());return 1;
    }
}
