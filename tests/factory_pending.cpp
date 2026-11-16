#include <meguviz/backend_ncnn_sdk.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
static PFN_vkWaitForFences real_wait=nullptr;
static std::atomic<bool> hold{false};
static std::atomic<unsigned> timeouts{0};
static VKAPI_ATTR VkResult VKAPI_CALL delayed_wait(VkDevice device,uint32_t count,const VkFence* fences,VkBool32 all,uint64_t timeout){
    if(hold.load()){++timeouts;return VK_TIMEOUT;}
    return real_wait(device,count,fences,all,timeout);
}
int main(){
    MgcRuntime* runtime=nullptr;MgcDomain* domain=nullptr;
    try{
        if(ncnn::create_gpu_instance()!=0||ncnn::get_gpu_count()==0)return 77;
        real_wait=ncnn::vkWaitForFences;CHECK(real_wait);
        ncnn::vkWaitForFences=delayed_wait;hold=true;
        MgcRuntimeDesc rd{};rd.header=MGC_HEADER_INIT(MgcRuntimeDesc);rd.provider=MGC_PROVIDER_NCNN_VULKAN;
        CHECK(mgc_runtime_create(&rd,&runtime)==MGC_SUCCESS);
        MgcDomainDesc dd{};dd.header=MGC_HEADER_INIT(MgcDomainDesc);
        const auto first=mgc_domain_acquire(runtime,&dd,&domain);
        CHECK(first==MGC_BUSY&&!domain&&timeouts.load()>0);
        CHECK(mgc_runtime_close(runtime)==MGC_BUSY);
        CHECK(mgc_domain_service(nullptr,0)==MGC_BUSY);
        // The timeout is only an injected observation. Actual GPU resources
        // stay owned until the original Vulkan wait proves retirement.
        hold=false;ncnn::vkWaitForFences=real_wait;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        MgcStatus serviced=MGC_BUSY;
        while(std::chrono::steady_clock::now()<deadline){serviced=mgc_domain_service(nullptr,0);if(serviced==MGC_SUCCESS)break;CHECK(serviced==MGC_BUSY);std::this_thread::sleep_for(std::chrono::milliseconds(2));}
        CHECK(serviced==MGC_SUCCESS);CHECK(mgc_runtime_close(runtime)==MGC_SUCCESS);mgc_runtime_release(runtime);runtime=nullptr;
        CHECK(mgc_runtime_create(&rd,&runtime)==MGC_SUCCESS);
        CHECK(mgc_domain_acquire(runtime,&dd,&domain)==MGC_SUCCESS&&domain);
        mgc_domain_release(domain);domain=nullptr;CHECK(mgc_runtime_close(runtime)==MGC_SUCCESS);mgc_runtime_release(runtime);runtime=nullptr;
        std::printf("PASS pending factory retains instance through %u injected timeouts, real wait retirement and new domain\n",timeouts.load());
        return 0;
    }catch(const std::exception& error){
        hold=false;if(real_wait)ncnn::vkWaitForFences=real_wait;
        std::fprintf(stderr,"FAIL %s\n",error.what());return 1;
    }
}
