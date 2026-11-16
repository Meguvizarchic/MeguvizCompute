#include <meguviz/backend_ncnn_sdk.hpp>
#include <condition_variable>
#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>
#include <cstdio>
#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (0)
using namespace meguviz;
static MgcStatus nothing(void *, const ncnn::VulkanDevice *) { return MGC_SUCCESS; }
static MgcStatus reject_nested_gpu(void *p, const ncnn::VulkanDevice *) {
    MgcDomainDesc desc{}; desc.header = MGC_HEADER_INIT(MgcDomainDesc);
    MgcDomain *domain = nullptr;
    CHECK(mgc_domain_acquire(static_cast<MgcRuntime *>(p), &desc, &domain) == MGC_INCOMPATIBLE_DOMAIN && !domain);
    return MGC_SUCCESS;
}
static MgcStatus retired(void *p, const ncnn::VulkanDevice *) {
    return ncnn_domain_owner_pending(static_cast<NcnnDomainOwner *>(p), false);
}
static MgcStatus throw_after_clear(void *p, const ncnn::VulkanDevice *) {
    CHECK(retired(p, nullptr) == MGC_SUCCESS);
    throw std::runtime_error("controlled callback failure");
}
struct Hold {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, release = false;
};
static MgcStatus held(void *p, const ncnn::VulkanDevice *) {
    auto &h = *static_cast<Hold *>(p);
    std::unique_lock<std::mutex> lock(h.mutex);
    h.entered = true; h.cv.notify_all();
    return h.cv.wait_for(lock, std::chrono::seconds(10), [&] { return h.release; }) ? MGC_SUCCESS : MGC_TIMEOUT;
}
int main() {
    MgcRuntime *gpu = nullptr, *cpu = nullptr;
    MgcDomain *a = nullptr, *b = nullptr, *c = nullptr;
    NcnnDomainOwner *owner = nullptr;
    try {
        MgcRuntimeDesc rd{}; rd.header = MGC_HEADER_INIT(MgcRuntimeDesc); rd.provider = MGC_PROVIDER_NCNN_VULKAN;
        CHECK(mgc_runtime_create(&rd, &gpu) == MGC_SUCCESS);
        MgcDomainDesc dd{}; dd.header = MGC_HEADER_INIT(MgcDomainDesc);
        auto status = mgc_domain_acquire(gpu, &dd, &a);
        if (status == MGC_UNSUPPORTED_CAPABILITY) { mgc_runtime_release(gpu); return 77; }
        CHECK(status == MGC_SUCCESS);
        CHECK(mgc_domain_acquire(gpu, &dd, &b) == MGC_SUCCESS);
        rd.provider = MGC_PROVIDER_NCNN_CPU;
        CHECK(mgc_runtime_create(&rd, &cpu) == MGC_SUCCESS);
        CHECK(mgc_domain_acquire(cpu, &dd, &c) == MGC_SUCCESS);
        CHECK(ncnn_domain_owner_create(a, &owner) == MGC_SUCCESS);
        // The owner marks pending before even a callback with no GPU commands.
        // This tests admission/retention, not a real submitted GPU fence.
        CHECK(with_ncnn_domain_owner(owner, nothing, nullptr) == MGC_SUCCESS);
        CHECK(ncnn_domain_owner_release(owner) == MGC_BUSY);
        CHECK(with_ncnn_domain(a, nothing, nullptr) == MGC_BUSY);
        CHECK(with_ncnn_domain(b, nothing, nullptr) == MGC_BUSY);
        CHECK(with_ncnn_domain_cleanup(b, nothing, nullptr) == MGC_SUCCESS);
        MgcResourceDesc resource{}; resource.header = MGC_HEADER_INIT(MgcResourceDesc);
        resource.capacity_bytes = 16; resource.format = MGC_FORMAT_FLOAT32; resource.usage = MGC_RESOURCE_GPU_BUFFER;
        MgcResource *rejected = nullptr;
        CHECK(mgc_resource_create(b, &resource, &rejected) == MGC_BUSY && !rejected);
        MgcDomain *rejected_domain = nullptr;
        CHECK(mgc_domain_acquire(gpu, &dd, &rejected_domain) == MGC_BUSY && !rejected_domain);
        CHECK(with_ncnn_domain_owner(owner, retired, owner) == MGC_SUCCESS);
        CHECK(with_ncnn_domain(b, nothing, nullptr) == MGC_SUCCESS);
        CHECK(with_ncnn_domain_owner(owner, throw_after_clear, owner) == MGC_INTERNAL_ERROR);
        CHECK(with_ncnn_domain(b, nothing, nullptr) == MGC_BUSY);
        // Hold the actual GPU execution mutex while unrelated CPU work runs.
        Hold hold;
        auto waiting = std::async(std::launch::async, [&] { return with_ncnn_domain_owner(owner, held, &hold); });
        { std::unique_lock<std::mutex> lock(hold.mutex); CHECK(hold.cv.wait_for(lock, std::chrono::seconds(5), [&] { return hold.entered; })); }
        auto cpu_work = std::async(std::launch::async, [&] { return with_ncnn_domain(c, reject_nested_gpu, gpu); });
        // A healthy global service has nothing to retire. It must not wait
        // for an unrelated running inference merely to drop its temporary pin.
        auto service = std::async(std::launch::async, [] { return mgc_domain_service(nullptr, 0); });
        const bool cpu_completed = cpu_work.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        const bool service_completed = service.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        { std::lock_guard<std::mutex> lock(hold.mutex); hold.release = true; hold.cv.notify_all(); }
        CHECK(waiting.get() == MGC_SUCCESS);
        CHECK(cpu_completed && cpu_work.get() == MGC_SUCCESS);
        CHECK(service_completed && service.get() == MGC_SUCCESS);
        CHECK(with_ncnn_domain_owner(owner, retired, owner) == MGC_SUCCESS);
        CHECK(ncnn_domain_owner_release(owner) == MGC_SUCCESS); owner = nullptr;
        // The owner pin keeps runtime close busy after the caller drops domain.
        CHECK(ncnn_domain_owner_create(a, &owner) == MGC_SUCCESS);
        mgc_domain_release(a); a = nullptr;
        mgc_domain_release(b); b = nullptr;
        CHECK(mgc_runtime_close(gpu) == MGC_BUSY);
        CHECK(ncnn_domain_owner_release(owner) == MGC_SUCCESS); owner = nullptr;
        CHECK(mgc_runtime_close(gpu) == MGC_SUCCESS);
        std::puts("PASS actual shared-device gate, exception retention, domain lifetime and independent CPU progress");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (c) mgc_domain_release(c);
    if (a) mgc_domain_release(a);
    if (b) mgc_domain_release(b);
    mgc_runtime_close(cpu); mgc_runtime_release(cpu); mgc_runtime_release(gpu);
    return 0;
}
