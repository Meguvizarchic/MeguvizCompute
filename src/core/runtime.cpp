#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <meguviz/compute.h>
#include <meguviz/compute_android.h>
#include <meguviz/compute_job.h>
#include <meguviz/kernels.h>
#include <memory>
#include <map>
#include <string>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
#include "producer_guard.hpp"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__ANDROID__)
#include <time.h>
#endif
#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <cerrno>
#include "fence_wait.hpp"
#include "sync_file_wait.hpp"
#endif
#if MGC_HAVE_NCNN
#include <datareader.h>
#include <layer.h>
#include <meguviz/backend_ncnn_sdk.hpp>
#endif
#ifndef MGC_ENABLE_GUARDED_PRODUCER_FENCE
#define MGC_ENABLE_GUARDED_PRODUCER_FENCE 0
#endif
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__) && MGC_ENABLE_GUARDED_PRODUCER_FENCE
struct ProducerWake;
#endif
#include "gpu_storage.hpp"
#include "code_image_lease.hpp"
#ifndef MGC_BUILD_FP
#define MGC_BUILD_FP ""
#endif
#ifndef MGC_PRIVATE_FP
#define MGC_PRIVATE_FP ""
#endif
#if defined(_WIN32)
extern "C" __declspec(dllexport) void meguviz_code_image_anchor_v1() {}
#else
extern "C" __attribute__((visibility("default"))) void meguviz_code_image_anchor_v1() {}
#endif
namespace
{
const unsigned char code_image_identity = 0;
std::atomic<uint64_t> next_domain{1};
std::atomic<uint64_t> next_allocation{1};
std::recursive_mutex execution_slot;
#if MGC_HAVE_NCNN && NCNN_VULKAN
std::mutex instance_mutex;
uint32_t instance_users = 0;
struct DeviceAdmission {
    std::mutex mutex;
    meguviz::NcnnDomainOwner *pending = nullptr;
    uint32_t owners = 0;
    std::atomic<bool> lost{false}, quiesced{false}, detached{false};
    std::atomic<int32_t> last_error{MGC_SUCCESS};
    const ncnn::VulkanDevice *device = nullptr;
    VkDevice identity = VK_NULL_HANDLE;
    uint32_t device_index = 0, domains = 0;
    uint64_t epoch = 0;
    std::atomic<bool> factory_pin{false};
    MgcRuntime *factory_runtime = nullptr;
    meguviz::detail::CodeImageLease factory_code;
};
std::map<VkDevice, std::shared_ptr<DeviceAdmission>> device_admissions;
std::map<uint32_t,std::shared_ptr<DeviceAdmission>> current_device_admissions;
std::atomic<uint64_t> next_device_epoch{1};
#endif
bool header_ok(const MgcHeader *h, size_t minimum)
{
    return h && h->abi_major == MGC_ABI_MAJOR && h->struct_size >= minimum;
}
template <class T> MgcStatus read_desc(const T *src, T &dst, size_t minimum)
{
    if (!src)
        return MGC_INVALID_ARGUMENT;
    if (!header_ok(&src->header, minimum))
        return MGC_UNSUPPORTED_ABI;
    std::memset(&dst, 0, sizeof(dst));
    std::memcpy(&dst, src, std::min<size_t>(src->header.struct_size, sizeof(dst)));
    return MGC_SUCCESS;
}
bool zeros(const void *ptr, size_t bytes)
{
    const auto *p = static_cast<const unsigned char *>(ptr);
    for (size_t i = 0; i < bytes; ++i)
        if (p[i])
            return false;
    return true;
}
void hash_from_hex(const char *s, MgcHash &h)
{
    std::memset(&h, 0, sizeof(h));
    if (std::strlen(s) != 64)
        return;
    for (int i = 0; i < 32; ++i)
    {
        auto n = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c - 'a' + 10; };
        h.bytes[i] = static_cast<uint8_t>((n(s[i * 2]) << 4) | n(s[i * 2 + 1]));
    }
}
MgcStatus flags(uint32_t required, const void *reserved, size_t bytes)
{
    return required                 ? MGC_UNSUPPORTED_ABI
           : zeros(reserved, bytes) ? MGC_SUCCESS
                                    : MGC_INVALID_ARGUMENT;
}
void invoke_context(MgcContextFn fn, void *ctx) noexcept
{
    if (fn)
    {
        try
        {
            fn(ctx);
        }
        catch (...)
        {
        }
    }
}
bool thread_cpu_ns(uint64_t &ns) noexcept
{
    ns = 0;
#if defined(_WIN32)
    FILETIME created{}, exited{}, kernel{}, user{};
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)) return false;
    const uint64_t kernel_ticks = (uint64_t(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime;
    const uint64_t user_ticks = (uint64_t(user.dwHighDateTime) << 32) | user.dwLowDateTime;
    if (kernel_ticks > UINT64_MAX - user_ticks) return false;
    const uint64_t ticks = kernel_ticks + user_ticks;
    if (ticks > UINT64_MAX / 100) return false;
    ns = ticks * 100; // FILETIME units are 100 ns.
    return true;
#elif (defined(__linux__) || defined(__ANDROID__)) && defined(CLOCK_THREAD_CPUTIME_ID)
    timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0 || value.tv_sec < 0 ||
        value.tv_nsec < 0 || value.tv_nsec >= 1000000000) return false;
    const uint64_t seconds = static_cast<uint64_t>(value.tv_sec);
    const uint64_t nanos = static_cast<uint64_t>(value.tv_nsec);
    if (seconds > (UINT64_MAX - nanos) / 1000000000) return false;
    ns = seconds * 1000000000 + nanos;
    return true;
#else
    return false;
#endif
}
} // namespace
struct MgcRuntime
{
    std::atomic<uint32_t> refs{1};
    std::mutex mutex;
    bool closed = false;
    uint32_t provider = 1;
    uint64_t budget = 0, live_bytes = 0;
    uint32_t domains = 0;
    uint32_t preparing_domains = 0, pending_initializations = 0;
    uint32_t measurement_flags = 0;
    MgcAllocationStats allocation_stats{};
};
namespace {
struct DomainPreparation {
    MgcRuntime* runtime=nullptr;
    MgcStatus acquire(MgcRuntime* value){std::lock_guard<std::mutex> lock(value->mutex);if(value->closed)return MGC_BUSY;if(value->preparing_domains==UINT32_MAX)return MGC_BACKPRESSURE;++value->preparing_domains;mgc_runtime_retain(value);runtime=value;return MGC_SUCCESS;}
    ~DomainPreparation(){if(runtime){{std::lock_guard<std::mutex> lock(runtime->mutex);--runtime->preparing_domains;}mgc_runtime_release(runtime);}}
};
}
struct MgcDomain
{
    std::atomic<uint32_t> refs{1};
    MgcRuntime *runtime = nullptr;
    uint64_t id = 0, generation = 1;
    uint32_t device_index = 0;
    std::atomic<uint32_t> measurement_flags{0};
    std::atomic<int32_t> timestamp_probe{MGC_BUSY};
    MgcAllocationStats allocation_stats{};
    std::mutex allocations_mutex;
    std::recursive_mutex cpu_execution_slot;
    std::set<std::pair<uint64_t, uint64_t>> allocations;
#if MGC_HAVE_NCNN && NCNN_VULKAN
    const ncnn::VulkanDevice *device = nullptr;
    std::shared_ptr<DeviceAdmission> admission;
    std::atomic<int32_t> vulkan_submission_probe{MGC_BUSY};
    bool instance_owned = false;
    ncnn::VkAllocator *blob_allocator = nullptr;
    ncnn::VkAllocator *staging_allocator = nullptr;
    std::map<std::string, std::unique_ptr<ncnn::Pipeline>> kernel_pipelines;
#if defined(__ANDROID__)
    std::set<AHardwareBuffer *> ahb_allocations;
    std::atomic<int32_t> sync_fd_probe{MGC_BUSY};
#endif
#endif
};
#if (MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__) && MGC_ENABLE_GUARDED_PRODUCER_FENCE) || defined(MGC_TEST_GUARDED_HOST)
// A weak per-resource reference avoids retaining a session across teardown.
// The Android guarded build waits on an eventfd: a notify is a counter
// increment, so a signal racing with poll() cannot be lost and no mutex or
// condition variable is needed. The host test build keeps the epoch pair
// because it has no sync-file descriptor to poll.
#if defined(__ANDROID__) && MGC_ENABLE_GUARDED_PRODUCER_FENCE
#define MGC_PRODUCER_WAKE_FD 1
struct ProducerWake {
    int fd = -1;
    void notify() noexcept {
        if (fd < 0) return;
        uint64_t one = 1;
        // A saturated counter stays readable, so EAGAIN needs no retry.
        (void)::write(fd, &one, sizeof(one));
    }
};
#else
#define MGC_PRODUCER_WAKE_FD 0
struct ProducerWake {
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<uint64_t> epoch{0};
    void notify() noexcept {
        // Predicate mutation must share the wait mutex, otherwise a notify
        // between predicate evaluation and wait could be lost.
        try { std::lock_guard<std::mutex> lock(mutex); epoch.fetch_add(1, std::memory_order_release); }
        catch (...) { /* The bounded polling backstop remains available. */ }
        changed.notify_all();
    }
};
#endif
#define MGC_HAVE_PRODUCER_WAKE 1
#else
#define MGC_HAVE_PRODUCER_WAKE 0
#endif
struct MgcResource
{
    std::atomic<uint32_t> refs{1};
    MgcDomain *domain = nullptr;
    MgcResourceDesc desc{};
    std::vector<unsigned char> data;
    unsigned char *external_data = nullptr;
    uint32_t external_access = 0;
    void *external_context = nullptr;
    MgcContextFn external_release = nullptr;
    unsigned char *host_data() noexcept { return external_data ? external_data : data.data(); }
    std::mutex mutex;
    uint32_t readers = 0, writers = 0, maps = 0, in_flight = 0;
    uint64_t current_content = 0;
    bool content_invalid = false;
    MgcStatus content_failure = MGC_DEVICE_LOST;
#if defined(MGC_TEST_GUARDED_HOST)
    // Host-only synthetic AHB state; Android stores it in AhbAllocator.
    std::array<std::weak_ptr<ProducerWake>, 2> producer_waiters;
    bool guarded_content = false;
#endif
#if defined(MGC_TEST_GUARDED_HOST)
    uint8_t test_guard = 0;
    uint64_t test_generation = 0;
    MgcStatus test_fence_status = MGC_SUCCESS;
    MgcLease *test_producer_pin = nullptr;
#endif
    uint64_t accounted_bytes = 0;
    uint64_t budget_bytes = 0;
    bool imported_allocation = false;
#if MGC_HAVE_NCNN && NCNN_VULKAN
    ncnn::VkMat gpu_buffer;
#if defined(__ANDROID__)
    std::unique_ptr<AhbAllocator> ahb_allocator;
    ncnn::VkImageMat gpu_image;
    AHardwareBuffer *ahb = nullptr;
    uint32_t external_family = VK_QUEUE_FAMILY_FOREIGN_EXT;
    bool externally_owned = true;
    int release_fd = -1;
    uint32_t ahb_sync_mode = MGC_ANDROID_HOST_WAIT_COMPAT;
    // At most one external read per allocation. Both reserved and fenced
    // states own an independent content-lease reference until physical ready.
    uint32_t external_access_state = 0; // 0 idle, 1 reserved, 2 fenced
    uint64_t external_access_generation = 0;
    MgcLease *external_access_lease = nullptr;
    int external_access_fd = -1;
    // 0: ordinary access; 1: guarded producer cleanup pending;
    // 2: guarded producer cleanup confirmed; 3: cleanup failed/uncertain.
    uint8_t producer_guard = 0;
    bool external_transition = false;
    bool ownership_uncertain = false;
#endif
#endif
};
struct MgcLease
{
    std::atomic<uint32_t> refs{1};
    MgcResource *resource = nullptr;
    MgcLeaseDesc desc{};
    uint32_t maps = 0;
#if defined(__ANDROID__)
    int acquire_fd = -1;
    uint32_t sync_mode = MGC_ANDROID_HOST_WAIT_COMPAT;
#endif
    void *content_context = nullptr;
    MgcContextFn release_content = nullptr;
};
#if defined(MGC_TEST_METADATA_SIZE)
extern "C" {
// Cross-compiled object-symbol sizes can be inspected without executing an
// Android binary. These probes are absent from all normal production builds.
unsigned char mgc_test_size_resource[sizeof(MgcResource)]{};
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
unsigned char mgc_test_size_ahb_allocator[sizeof(AhbAllocator)]{};
#endif
}
#endif
// Tickets can outlive their session. Keep a separate identity alive so a new
// session at the same address cannot acquire an old ticket's cancellation right.
struct SessionIdentity {
#if MGC_HAVE_PRODUCER_WAKE
    // Mutable because existing tickets intentionally hold a const identity.
    // The wake shares this already-allocated identity's control block. The
    // eventfd is owned here rather than by ProducerWake so the aliasing
    // shared_ptr used by resources never closes it early; tickets may notify
    // after the session itself is gone.
    mutable ProducerWake wake;
#if MGC_PRODUCER_WAKE_FD
    SessionIdentity() { wake.fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC); }
    ~SessionIdentity() { if (wake.fd >= 0) ::close(wake.fd); }
#endif
#endif
};
struct MgcTicket
{
    std::shared_ptr<const SessionIdentity> session_identity;
    MgcDomain *domain = nullptr;
    std::atomic<uint32_t> refs{1};
    std::atomic<bool> cancelled{false};
#if MGC_HAVE_PRODUCER_WAKE
    std::atomic<bool> guarded_submission{false}; // may also be discovered by a private batch
    bool guarded_write_published = false; // any private transfer; protected by ticket mutex
#endif
    std::mutex mutex;
    MgcCompletion result{};
    bool done = false;
    MgcSubmitDesc desc{};
    MgcHostJobFn execute = nullptr;
    uint32_t host_flags = 0;
    MgcHostJobFn resume = nullptr;
    MgcContextFn finalize = nullptr;
    MgcHostRetireFn retire = nullptr;
    void *retire_context = nullptr;
    bool deferred = false, resume_allowed = false;
    MgcStatus deferred_failure = MGC_INTERNAL_ERROR;
    std::shared_ptr<void> gpu_batch;
    std::shared_ptr<void> operation_state;
    std::vector<MgcLease *> leases;
    std::atomic<uint64_t> submitted_batches{0}, retired_batches{0}, finalized_batches{0}, recording_batches{0};
    bool final_submitting = false;
    std::atomic<bool> used_final_async{false};
    uint32_t measurement_flags = 0;
    uint64_t gpu_ns = 0, gpu_observed_batches = 0;
    uint32_t gpu_bits = 0, gpu_period = 0, gpu_pending = 0;
    bool gpu_complete = true, event_complete = true;
    std::vector<MgcBatchEvent> events;
};
struct MgcSession
{
    std::shared_ptr<const SessionIdentity> identity = std::make_shared<SessionIdentity>();
    MgcDomain *domain = nullptr;
#if MGC_HAVE_NCNN && NCNN_VULKAN
    // Session close joins the worker only after every accepted ticket retires.
    // Owners created by that worker can therefore borrow this verified image
    // reference for exactly the session's lifetime.
    meguviz::detail::CodeImageLease code_image;
#endif
    std::mutex mutex;
    std::condition_variable work, retired;
    std::deque<MgcTicket *> queue;
    uint32_t max_in_flight = 1, in_flight = 0, preparing = 0;
    bool closing = false;
    std::thread worker;
};
#if defined(MGC_TEST_METADATA_SIZE)
extern "C" {
unsigned char mgc_test_size_session[sizeof(MgcSession)]{};
#if MGC_HAVE_PRODUCER_WAKE
unsigned char mgc_test_size_producer_wake[sizeof(ProducerWake)]{};
#endif
}
#endif
#if MGC_HAVE_PRODUCER_WAKE
std::shared_ptr<ProducerWake> producer_wake_alias(const std::shared_ptr<const SessionIdentity> &identity) noexcept
{
    return identity ? std::shared_ptr<ProducerWake>(identity, &identity->wake) : std::shared_ptr<ProducerWake>{};
}
constexpr size_t producer_waiter_capacity = 2;
std::weak_ptr<ProducerWake> *producer_waiter_slots_locked(MgcResource *resource) noexcept
{
#if defined(MGC_TEST_GUARDED_HOST)
    return resource->producer_waiters.data();
#else
    return resource->ahb_allocator ? resource->ahb_allocator->producer_waiters.data() : nullptr;
#endif
}
bool resource_guarded_content_locked(MgcResource *resource) noexcept
{
#if defined(MGC_TEST_GUARDED_HOST)
    return resource->guarded_content;
#else
    return resource->ahb_allocator && resource->ahb_allocator->guarded_content;
#endif
}
void set_guarded_content_locked(MgcResource *resource, bool value) noexcept
{
#if defined(MGC_TEST_GUARDED_HOST)
    resource->guarded_content = value;
#else
    if (resource->ahb_allocator) resource->ahb_allocator->guarded_content = value;
#endif
}
void register_producer_waiter_locked(MgcResource *resource, const std::shared_ptr<ProducerWake> &wake) noexcept
{
    auto *slots = producer_waiter_slots_locked(resource);
    if (!wake || !slots) return;
    std::weak_ptr<ProducerWake> *empty = nullptr;
    for (size_t i = 0; i < producer_waiter_capacity; ++i) {
        auto &slot = slots[i];
        auto existing = slot.lock();
        if (existing == wake) return;
        if (!existing && !empty) empty = &slot;
    }
    // Bounded capacity: a private batch may register a requested lease that
    // accept() does not count in-flight. If more than two distinct waiters
    // contend, the extras fall back to the bounded watchdog rather than
    // growing per-AHB storage.
    if (empty) *empty = wake;
}
void unregister_producer_waiter_locked(MgcResource *resource, const std::shared_ptr<ProducerWake> &wake) noexcept
{
    auto *slots = producer_waiter_slots_locked(resource);
    if (!wake || !slots) return;
    for (size_t i = 0; i < producer_waiter_capacity; ++i) {
        auto &slot = slots[i];
        auto existing = slot.lock();
        if (!existing || existing == wake) slot.reset();
    }
}
using ProducerWakeList = std::array<std::shared_ptr<ProducerWake>, producer_waiter_capacity>;
ProducerWakeList producer_waiters_locked(MgcResource *resource) noexcept
{
    ProducerWakeList result{};
    auto *slots = producer_waiter_slots_locked(resource);
    if (slots) for (size_t i = 0; i < result.size(); ++i) result[i] = slots[i].lock();
    return result;
}
void notify_producer_waiters(const ProducerWakeList &waiters) noexcept
{
    for (const auto &waiter : waiters) if (waiter) waiter->notify();
}
#endif
#if MGC_HAVE_NCNN && NCNN_VULKAN
namespace meguviz {
struct NcnnDomainOwner {
    MgcDomain *domain = nullptr;
    MgcTicket *ticket = nullptr;
    detail::CodeImageLease code_image;
    bool pending = false;
    bool retirement_only = false;
};
}
#endif
namespace
{
// A caller's context callbacks run outside the session lock. Pin the session
// until submit finishes, so a reentrant/concurrent close cannot free it.
struct SubmissionPreparation
{
    MgcSession *session = nullptr;
    MgcStatus acquire(MgcSession *s)
    {
        std::lock_guard<std::mutex> lock(s->mutex);
        if (s->closing)
            return MGC_BUSY;
        if (s->preparing == UINT32_MAX)
            return MGC_BACKPRESSURE;
        ++s->preparing;
        session = s;
        return MGC_SUCCESS;
    }
    ~SubmissionPreparation()
    {
        if (!session)
            return;
        std::lock_guard<std::mutex> lock(session->mutex);
        --session->preparing;
    }
};
thread_local MgcSession *worker_session = nullptr;
thread_local MgcDomain *active_domain = nullptr;
thread_local MgcTicket *active_ticket = nullptr;
thread_local std::recursive_mutex *active_execution_slot = nullptr;
#if MGC_HAVE_NCNN && NCNN_VULKAN
thread_local meguviz::NcnnDomainOwner *active_owner = nullptr;
#endif
std::recursive_mutex &domain_slot(MgcDomain *domain) {
#if MGC_HAVE_NCNN && NCNN_VULKAN
    if (domain->device) return execution_slot;
#endif
    return domain->cpu_execution_slot;
}
MgcStatus domain_admission(MgcDomain *domain) {
#if MGC_HAVE_NCNN && NCNN_VULKAN
    if(domain->admission&&domain->admission->lost.load()){
        // Owners may enter only to retire their lost-device state; all such
        // implementations check loss before recording any further command.
        if(!active_owner||!active_owner->retirement_only||active_owner->domain->admission!=domain->admission)return MGC_DEVICE_LOST;
        return MGC_SUCCESS;
    }
    if (domain->admission && domain->admission->pending &&
        domain->admission->pending != active_owner) return MGC_BUSY;
#endif
    return MGC_SUCCESS;
}
// GPU serialization retains its established process-wide order. CPU work uses
// its own domain mutex, so finite GPU retirement polls cannot hold up CPU jobs.
// Normal cross-slot nesting is rejected before taking a second mutex.
class DomainScope {
    MgcDomain *domain_;
    MgcDomain *previous_domain_ = nullptr;
    std::recursive_mutex *previous_slot_ = nullptr;
    std::unique_lock<std::recursive_mutex> lock_;
    bool entered_ = false;
public:
    MgcStatus status = MGC_INTERNAL_ERROR;
    explicit DomainScope(MgcDomain *domain, bool try_only = false, bool enabled = true) : domain_(domain) {
        mgc_domain_retain(domain_);
        if (!enabled) { status = MGC_SUCCESS; return; }
        try {
        auto &slot = domain_slot(domain_);
        if (active_execution_slot && active_execution_slot != &slot) {
            status = MGC_INCOMPATIBLE_DOMAIN; return;
        }
        lock_ = std::unique_lock<std::recursive_mutex>(slot, std::defer_lock);
        if (try_only) { if (!lock_.try_lock()) { status = MGC_BUSY; return; } }
        else lock_.lock();
        status = domain_admission(domain_);
        if (status) return;
        previous_domain_ = active_domain;
        previous_slot_ = active_execution_slot;
        active_domain = domain_;
        active_execution_slot = &slot;
        entered_ = true;
        } catch (...) { status = MGC_INTERNAL_ERROR; }
    }
    ~DomainScope() {
        if (entered_) { active_domain = previous_domain_; active_execution_slot = previous_slot_; }
        if (lock_.owns_lock()) lock_.unlock();
        mgc_domain_release(domain_);
    }
    DomainScope(const DomainScope &) = delete;
    DomainScope &operator=(const DomainScope &) = delete;
};
#if MGC_HAVE_NCNN && NCNN_VULKAN
bool domain_is_lost(MgcDomain *domain){return domain&&domain->admission&&domain->admission->lost.load();}
bool device_loss_retired(MgcDomain *domain){return domain_is_lost(domain)&&domain->admission->quiesced.load();}
void report_device_lost(MgcDomain *domain){
    if(!domain||!domain->admission)return;
    std::lock_guard<std::recursive_mutex> device(execution_slot);
    std::lock_guard<std::mutex> metadata(domain->admission->mutex);
    domain->admission->last_error=MGC_DEVICE_LOST;
    domain->admission->lost=true;
}
MgcStatus detach_retired_device(const std::shared_ptr<DeviceAdmission>& state){
    if(!state->quiesced.load()||state->detached.load()||!state->device)return MGC_SUCCESS;
    std::lock_guard<std::mutex> instance(instance_mutex);
    size_t old_generations=0;
    for(const auto& entry:device_admissions)if(entry.second->device_index==state->device_index&&entry.second->detached.load()&&(entry.second->domains||entry.second->factory_pin))++old_generations;
    if(old_generations>=4)return MGC_BACKPRESSURE;
    if(ncnn::detach_gpu_device(int(state->device_index),state->device)!=state->device)return MGC_BUSY;
    state->detached=true;
    auto current=current_device_admissions.find(state->device_index);
    if(current!=current_device_admissions.end()&&current->second==state)current_device_admissions.erase(current);
    if(!state->domains&&!state->factory_pin){delete state->device;state->device=nullptr;device_admissions.erase(state->identity);}
    return MGC_SUCCESS;
}
MgcStatus service_loss_state(const std::shared_ptr<DeviceAdmission>& state){
    if(!state||!state->lost.load())return MGC_SUCCESS;
    std::unique_lock<std::recursive_mutex> device(execution_slot,std::try_to_lock);
    if(!device.owns_lock())return MGC_BUSY;
    if(!state->device)return MGC_SUCCESS;
    if(!state->quiesced.load()){
        if(!state->device->vkdevice()){
            if(state->device->initialization_pending())return MGC_BUSY;
            state->device->retire_initialization_device_lost();state->quiesced=true;
            detach_retired_device(state);return MGC_SUCCESS;
        }
        const auto result=ncnn::vkDeviceWaitIdle(state->device->vkdevice());
        if(result!=VK_SUCCESS&&result!=VK_ERROR_DEVICE_LOST){state->last_error=result==VK_ERROR_OUT_OF_HOST_MEMORY||result==VK_ERROR_OUT_OF_DEVICE_MEMORY?MGC_OUT_OF_MEMORY:MGC_INTERNAL_ERROR;return MGC_BUSY;}
        state->device->retire_initialization_device_lost();
        state->quiesced=true;
    }
    // A retained generation limit can delay new admission, never retirement.
    detach_retired_device(state);
    return MGC_SUCCESS;
}
MgcStatus service_device_loss(MgcDomain *domain){return domain&&domain->admission?service_loss_state(domain->admission):MGC_SUCCESS;}
MgcStatus pin_factory_locked(const std::shared_ptr<DeviceAdmission>& state,MgcRuntime* runtime){
    if(state->factory_pin)return MGC_SUCCESS;
    state->factory_code.acquire(&code_image_identity,"meguviz_code_image_anchor_v1");
    state->factory_runtime=runtime;++instance_users;
    mgc_runtime_retain(runtime);
    {std::lock_guard<std::mutex> lock(runtime->mutex);++runtime->pending_initializations;}
    // Publish only after every retained object and counter is established: a
    // service snapshot can already hold this admission on another thread.
    state->factory_pin.store(true,std::memory_order_release);
    return state->factory_code.available()?MGC_SUCCESS:MGC_UNSUPPORTED_CAPABILITY;
}
void attach_cached_device_locked(MgcDomain* domain,uint32_t index,MgcRuntime* runtime){
    if(domain->admission){auto old=domain->admission;if(old->domains)--old->domains;
        if(!old->domains&&old->detached.load()&&!old->factory_pin&&old->device){delete old->device;old->device=nullptr;device_admissions.erase(old->identity);}}
    domain->device=nullptr;domain->admission.reset();
    auto& current=current_device_admissions[index];
    if(!current){current=std::make_shared<DeviceAdmission>();current->device_index=index;current->epoch=next_device_epoch++;if(!current->epoch)throw std::bad_alloc();}
    domain->admission=current;++current->domains;
    // A new VulkanDevice may submit initialization before its constructor
    // returns. Secure the code image before that first possible submission.
    if(!current->device){
        current->factory_code.acquire(&code_image_identity,"meguviz_code_image_anchor_v1");
        if(!current->factory_code.available())throw std::runtime_error("cannot retain factory code image");
    }
    domain->device=ncnn::get_gpu_device(index);
    if(current->device&&current->device!=domain->device)throw std::runtime_error("device cache identity changed");
    if(!current->device)current->device=domain->device;
    if(domain->device){
        if(current->identity==VK_NULL_HANDLE)current->identity=domain->device->vkdevice();
        if(!domain->device->is_valid()&&domain->device->initialization_pending())pin_factory_locked(current,runtime);
        device_admissions[current->identity]=current;
    }
    if(!current->factory_pin)current->factory_code.reset();
}
MgcStatus service_factory(const std::shared_ptr<DeviceAdmission>& state){
    // Healthy instances have no factory work. Avoid reporting BUSY solely
    // because an unrelated inference currently owns the execution slot.
    if(!state||!state->factory_pin.load())return MGC_SUCCESS;
    MgcRuntime* release_runtime=nullptr;
    {
        std::unique_lock<std::recursive_mutex> slot(execution_slot,std::try_to_lock);
        if(!slot.owns_lock())return MGC_BUSY;
        if(!state->factory_pin)return MGC_SUCCESS;
        if(state->lost.load()){
            service_loss_state(state);if(!state->quiesced.load())return MGC_BUSY;
        }else if(state->device&&state->device->initialization_pending()){
            state->device->poll_initialization(0);
            if(state->device->initialization_result()==VK_ERROR_DEVICE_LOST){state->lost=true;state->last_error=MGC_DEVICE_LOST;service_loss_state(state);}
            if(state->device->initialization_pending()&&!state->quiesced.load())return MGC_BUSY;
        }
        if(state->device&&!state->device->is_valid()&&!state->detached.load()){
            // An unpublished initialization candidate has no other queue users.
            // No pending initialization (or lost proof) permits origin cleanup.
            if(state->device->initialization_pending())return MGC_BUSY;
            state->quiesced=true;
            if(detach_retired_device(state)!=MGC_SUCCESS)return MGC_BUSY;
        }
        if(!state->factory_code.reset())return MGC_BUSY;
        std::lock_guard<std::mutex> instance(instance_mutex);
        state->factory_pin=false;
        release_runtime=state->factory_runtime;state->factory_runtime=nullptr;
        {std::lock_guard<std::mutex> lock(release_runtime->mutex);--release_runtime->pending_initializations;}
        if(state->detached.load()&&!state->domains&&state->device){delete state->device;state->device=nullptr;device_admissions.erase(state->identity);}
        if(!--instance_users){device_admissions.clear();current_device_admissions.clear();ncnn::destroy_gpu_instance();}
    }
    mgc_runtime_release(release_runtime);
    return MGC_SUCCESS;
}
void cleanup_domain_device(MgcDomain *domain) {
    if (!domain->instance_owned) return;
    std::lock_guard<std::recursive_mutex> slot(execution_slot);
    std::lock_guard<std::mutex> gpu(instance_mutex);
    domain->kernel_pipelines.clear();
    if (domain->blob_allocator) domain->device->reclaim_blob_allocator(domain->blob_allocator);
    if (domain->staging_allocator) domain->device->reclaim_staging_allocator(domain->staging_allocator);
    auto state=domain->admission;
    if(state&&state->domains){
        --state->domains;
        if(!state->domains&&state->detached.load()&&!state->factory_pin){
            delete state->device;state->device=nullptr;
            device_admissions.erase(state->identity);
        }
    }
    domain->instance_owned = false;
    if (!--instance_users) {
        device_admissions.clear();current_device_admissions.clear();
        ncnn::destroy_gpu_instance();
    }
}
#endif
bool ticket_owns(MgcLease *l)
{
    return active_ticket && std::find(active_ticket->leases.begin(), active_ticket->leases.end(), l) != active_ticket->leases.end();
}
void record_event(MgcTicket *t, uint32_t kind, uint64_t batch)
{
    if (!t || !t->desc.batch_event) return;
    if (t->events.size() >= 128) { t->event_complete = false; return; }
    MgcBatchEvent e{}; e.header = MGC_HEADER_INIT(MgcBatchEvent);
    e.request_id = t->desc.request_id; e.batch_id = batch; e.kind = kind;
    e.monotonic_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    t->events.push_back(e);
}
void observe_gpu(MgcTicket *t, uint64_t ns, uint32_t observed, uint32_t bits, uint32_t period)
{
    if (!t || !(t->measurement_flags & MGC_MEASURE_GPU_TIMESTAMPS)) return;
    if (bits && period) {
        if (t->gpu_bits && (t->gpu_bits != bits || t->gpu_period != period)) { t->gpu_complete = false; return; }
        t->gpu_bits = bits; t->gpu_period = period;
    }
    if (!observed || !bits || !period || t->gpu_ns > UINT64_MAX - ns) { t->gpu_complete = false; return; }
    if (t->gpu_observed_batches && (t->gpu_bits != bits || t->gpu_period != period)) { t->gpu_complete = false; return; }
    t->gpu_bits = bits; t->gpu_period = period; t->gpu_ns += ns; ++t->gpu_observed_batches;
    if (active_domain) active_domain->timestamp_probe = MGC_SUCCESS;
}
void allocation_change(MgcDomain *domain, uint64_t bytes, bool imported, bool add)
{
    for (auto *stats : {&domain->runtime->allocation_stats, &domain->allocation_stats}) {
        auto &live = imported ? stats->imported_live_bytes : stats->managed_live_bytes;
        auto &count = imported ? stats->imported_live_count : stats->managed_live_count;
        auto &peak = imported ? stats->imported_highwater_bytes : stats->managed_highwater_bytes;
        auto &peak_count = imported ? stats->imported_highwater_count : stats->managed_highwater_count;
        if (add) {
            if (live > UINT64_MAX - bytes || count == UINT64_MAX || stats->total_allocation_count == UINT64_MAX) stats->resource_ranges_observed = 0;
            else { live += bytes; ++count; ++stats->total_allocation_count; peak = std::max(peak, live); peak_count = std::max(peak_count, count); }
        }
        else if (live >= bytes && count) { live -= bytes; --count; }
        else stats->resource_ranges_observed = 0;
    }
}
#include "gpu_runtime.inc"
#include "owned_ahb_transition.inc"
#include "standalone_gpu_jobs.inc"
MgcStatus output_struct(void *out, const void *src, size_t size, size_t minimum)
{
    if (!out)
        return MGC_INVALID_ARGUMENT;
    auto *h = static_cast<MgcHeader *>(out);
    if (!header_ok(h, minimum))
        return MGC_UNSUPPORTED_ABI;
    const uint32_t caller_size = h->struct_size;
    size_t n = std::min<size_t>(caller_size, size);
    std::memcpy(out, src, n);
    h->struct_size = caller_size;
    return MGC_SUCCESS;
}
MgcStatus validate_lease(MgcSession *s, MgcLease *l)
{
    if (!l)
        return MGC_INVALID_ARGUMENT;
    if (l->resource->domain != s->domain)
        return MGC_INCOMPATIBLE_DOMAIN;
    if (l->desc.dependency_kind != MGC_DEPENDENCY_READY &&
        l->desc.dependency_kind != MGC_DEPENDENCY_HOST_WAIT &&
        l->desc.dependency_kind != MGC_DEPENDENCY_SYNC_FD)
        return MGC_UNSUPPORTED_CAPABILITY;
    return MGC_SUCCESS;
}
MgcStatus operation_valid(const MgcSubmitDesc &d)
{
    if (!d.input || !d.output)
        return MGC_INVALID_ARGUMENT;
    auto *a = d.input;
    auto *b = d.output;
    if (a->desc.access != MGC_ACCESS_READ || b->desc.access != MGC_ACCESS_WRITE)
        return MGC_INVALID_ARGUMENT;
    if (a->resource == b->resource)
        return MGC_INVALID_ARGUMENT;
    if (d.operation == MGC_OPERATION_COPY)
        return d.byte_count && d.byte_count <= a->desc.byte_length &&
                       d.byte_count <= b->desc.byte_length
                   ? MGC_SUCCESS
                   : MGC_INVALID_ARGUMENT;
    if (d.operation == MGC_OPERATION_NORMALIZE_F32 || d.operation == MGC_OPERATION_STATISTICS_F32 ||
        d.operation == MGC_OPERATION_RELU_F32)
    {
        if (!d.byte_count || d.byte_count % 4 || a->resource->desc.format != MGC_FORMAT_FLOAT32 ||
            d.byte_count > a->desc.byte_length)
            return MGC_INVALID_ARGUMENT;
        uint64_t needed =
            d.operation == MGC_OPERATION_STATISTICS_F32 ? sizeof(MgcStatisticsF32) : d.byte_count;
        if (needed > b->desc.byte_length)
            return MGC_INVALID_ARGUMENT;
        return MGC_SUCCESS;
    }
    if (d.operation == MGC_OPERATION_RESIZE_RGB8)
    {
        const auto &x = a->resource->desc;
        const auto &y = b->resource->desc;
        if (d.byte_count || x.format != MGC_FORMAT_RGB8 || y.format != MGC_FORMAT_RGB8 ||
            !x.width || !x.height || !y.width || !y.height)
            return MGC_INVALID_ARGUMENT;
        auto valid = [](const MgcResourceDesc &v, uint64_t n)
        {
            return v.row_stride_bytes >= uint64_t(v.width) * 3 &&
                   v.height <= UINT64_MAX / v.row_stride_bytes &&
                   uint64_t(v.height) * v.row_stride_bytes <= n;
        };
        return valid(x, a->desc.byte_length) && valid(y, b->desc.byte_length)
                   ? MGC_SUCCESS
                   : MGC_INVALID_ARGUMENT;
    }
    return MGC_UNSUPPORTED_CAPABILITY;
}
MgcStatus run_operation(MgcTicket *t)
{
    const auto &d = t->desc;
#if MGC_HAVE_NCNN && NCNN_VULKAN
    if (d.input->resource->domain->device)
    {
        const auto *device=d.input->resource->domain->device;
        if(d.operation!=MGC_OPERATION_RESIZE_RGB8||device->enabled_features().shaderFloat64)
            return run_gpu_operation(t);
        // Preserve the CPU reference's binary64 rounding when the device has
        // no shaderFloat64. GPU-only and AHB storage cannot be CPU mapped here.
        if(d.input->resource->desc.usage||d.output->resource->desc.usage)
            return MGC_UNSUPPORTED_CAPABILITY;
    }
#endif
    const unsigned char *src = d.input->resource->host_data() + d.input->desc.byte_offset;
    unsigned char *dst = d.output->resource->host_data() + d.output->desc.byte_offset;
    if (d.operation == MGC_OPERATION_COPY)
    {
        std::memcpy(dst, src, static_cast<size_t>(d.byte_count));
        return MGC_SUCCESS;
    }
    if (d.operation == MGC_OPERATION_NORMALIZE_F32 || d.operation == MGC_OPERATION_STATISTICS_F32 ||
        d.operation == MGC_OPERATION_RELU_F32)
    {
        MgcStatisticsF32 st{};
        st.header = MGC_HEADER_INIT(MgcStatisticsF32);
        st.count = d.byte_count / 4;
        for (uint64_t i = 0; i < st.count; ++i)
        {
            float x;
            std::memcpy(&x, src + i * 4, 4);
            if (std::isfinite(x))
            {
                if (!st.finite_count)
                    st.minimum = st.maximum = x;
                else
                {
                    st.minimum = std::min(st.minimum, x);
                    st.maximum = std::max(st.maximum, x);
                }
                st.sum += x;
                ++st.finite_count;
            }
        }
        if (d.operation == MGC_OPERATION_STATISTICS_F32)
        {
            std::memcpy(dst, &st, sizeof(st));
            return MGC_SUCCESS;
        }
        if (d.operation == MGC_OPERATION_RELU_F32)
        {
            if (st.finite_count != st.count || st.count > INT32_MAX)
                return MGC_INVALID_ARGUMENT;
            auto provider = d.input->resource->domain->runtime->provider;
            if (provider == MGC_PROVIDER_NCNN_VULKAN)
                return MGC_UNSUPPORTED_CAPABILITY;
#if MGC_HAVE_NCNN
            if (provider == MGC_PROVIDER_NCNN_CPU)
            {
                ncnn::Net net;
                net.opt.use_vulkan_compute = false;
                net.opt.num_threads = 1;
                net.opt.use_packing_layout = false;
                net.opt.use_fp16_storage = false;
                net.opt.use_fp16_arithmetic = false;
                if (net.load_param_mem(
                        "7767517\n2 2\nInput input 0 1 in0\nReLU relu 1 1 in0 out0\n"))
                    return MGC_INTERNAL_ERROR;
                alignas(4) unsigned char empty_model[4] = {};
                const unsigned char *model_cursor = empty_model;
                ncnn::DataReaderFromMemory reader(model_cursor, sizeof(empty_model));
                if (net.load_model(reader))
                    return MGC_INTERNAL_ERROR;
                ncnn::Mat input(static_cast<int>(st.count));
                if (input.empty())
                    return MGC_OUT_OF_MEMORY;
                input.fill(0.f);
                std::memcpy(input.data, src, static_cast<size_t>(d.byte_count));
                auto ex = net.create_extractor();
                if (ex.input("in0", input))
                    return MGC_INTERNAL_ERROR;
                ncnn::Mat output;
                if (ex.extract("out0", output))
                    return MGC_INTERNAL_ERROR;
                if (output.dims != 1 || output.w != static_cast<int>(st.count) ||
                    output.elempack != 1 || output.total() < st.count || output.elemsize != 4)
                    return MGC_INTERNAL_ERROR;
                std::memcpy(dst, output.data, static_cast<size_t>(d.byte_count));
                return MGC_SUCCESS;
            }
#endif
            for (uint64_t i = 0; i < st.count; ++i)
            {
                float x;
                std::memcpy(&x, src + i * 4, 4);
                x = std::max(0.f, x);
                std::memcpy(dst + i * 4, &x, 4);
            }
            return MGC_SUCCESS;
        }
        if (st.finite_count != st.count)
            return MGC_INVALID_ARGUMENT;
        double range = double(st.maximum) - st.minimum;
        for (uint64_t i = 0; i < st.count; ++i)
        {
            float x;
            std::memcpy(&x, src + i * 4, 4);
            float y = range ? static_cast<float>((double(x) - st.minimum) / range) : 0.f;
            std::memcpy(dst + i * 4, &y, 4);
        }
        return MGC_SUCCESS;
    }
    const auto &x = d.input->resource->desc;
    const auto &y = d.output->resource->desc;
    for (uint32_t j = 0; j < y.height; ++j)
        for (uint32_t i = 0; i < y.width; ++i)
        {
            double fx = (double(i) + .5) * x.width / y.width - .5,
                   fy = (double(j) + .5) * x.height / y.height - .5;
            int64_t ix = static_cast<int64_t>(std::floor(fx)),
                    iy = static_cast<int64_t>(std::floor(fy));
            double ax = fx - ix, ay = fy - iy;
            auto clamp = [](int64_t v, uint32_t n)
            { return static_cast<uint32_t>(std::max<int64_t>(0, std::min<int64_t>(v, n - 1))); };
            for (uint32_t c = 0; c < 3; ++c)
            {
                auto at = [&](int64_t a, int64_t b)
                {
                    return double(src[uint64_t(clamp(b, x.height)) * x.row_stride_bytes +
                                      uint64_t(clamp(a, x.width)) * 3 + c]);
                };
                double z = (at(ix, iy) * (1 - ax) + at(ix + 1, iy) * ax) * (1 - ay) +
                           (at(ix, iy + 1) * (1 - ax) + at(ix + 1, iy + 1) * ax) * ay;
                dst[uint64_t(j) * y.row_stride_bytes + uint64_t(i) * 3 + c] =
                    static_cast<unsigned char>(std::floor(z + .5));
            }
        }
    return MGC_SUCCESS;
}
void release_submission(MgcTicket *t)
{
    for (auto *l : t->leases)
        mgc_lease_release(l);
    t->leases.clear();
}
#if defined(MGC_TEST_GUARDED_HOST)
// Host-only injection exercises the real accept/worker lifecycle with a
// synthetic producer fence. It does not model Android GPU or fd ownership.
MgcStatus test_guard_probe(MgcLease *lease)
{
    auto *r = lease->resource;
    std::lock_guard<std::mutex> lock(r->mutex);
    if (r->content_invalid) return r->content_failure;
    if (!r->test_guard) return MGC_SUCCESS;
    if (r->test_fence_status == MGC_PRODUCER_FAILED) {
        r->content_invalid = true; r->content_failure = MGC_PRODUCER_FAILED; r->test_guard = 3;
        return MGC_PRODUCER_FAILED;
    }
    if (r->test_fence_status != MGC_SUCCESS) return r->test_fence_status;
    return meguviz_compute_detail::producer_gate_status(r->test_guard, r->content_invalid, r->content_failure);
}
extern "C" void mgc_test_after_guard_ready(MgcTicket *ticket);
#endif
enum class ProducerGate { Ready, Waiting, Failed, Cancelled };
struct ProducerGateResult {
    ProducerGate gate = ProducerGate::Ready;
    MgcStatus status = MGC_SUCCESS;
    bool producer_gate_skipped = false;
    bool inspectable = true; // at least one waiting dependency can be polled
};
// One non-blocking pass over a ticket's producer dependencies. A waiting
// ticket contributes a duplicated fence descriptor so the worker can block on
// the kernel fence instead of a timer. Resource locks are taken only inside
// the probe helpers; the caller must not hold the session lock.
ProducerGateResult probe_producer_gate(MgcTicket *t, std::vector<int> &poll_fds) noexcept
{
    ProducerGateResult result;
    result.status = MGC_CANCELLED;
#if (MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)) || defined(MGC_TEST_GUARDED_HOST)
    // Accepted work may wait in the session without taking the GPU owner or
    // issuing any command. Every AHB dependency (including a sibling lease)
    // must verify the producer and its host cleanup gate first.
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
    if (domain_is_lost(t->domain)) {
        result.status = MGC_DEVICE_LOST;
        result.gate = ProducerGate::Failed;
        return result;
    }
#endif
    if (t->cancelled.load()) {
        for (auto *lease : t->leases) if (lease) {
            std::lock_guard<std::mutex> lock(lease->resource->mutex);
            // A previously confirmed producer failure outranks a concurrent
            // cancellation; cancellation must not hide it.
            if (lease->resource->content_invalid &&
                lease->resource->content_failure == MGC_PRODUCER_FAILED)
                result.status = MGC_PRODUCER_FAILED;
#if defined(MGC_TEST_GUARDED_HOST)
            if (lease->resource->test_guard) result.producer_gate_skipped = true;
#else
            if (lease->resource->ahb) result.producer_gate_skipped = true;
#endif
        }
        result.gate = ProducerGate::Cancelled;
        return result;
    }
    bool waiting = false;
    for (auto *lease : t->leases) if (lease) {
        MgcStatus dependency;
#if defined(MGC_TEST_GUARDED_HOST)
        dependency = test_guard_probe(lease);
#else
        if (!lease->resource->ahb) continue;
        dependency = wait_ready(lease, 0);
#endif
        if (dependency == MGC_SUCCESS) continue;
        if (dependency != MGC_BUSY && dependency != MGC_TIMEOUT) {
            result.status = dependency;
            result.gate = ProducerGate::Failed;
            return result;
        }
        waiting = true;
#if MGC_HAVE_PRODUCER_WAKE
        const int fd = dup_producer_poll_fd(lease);
        if (fd >= 0) poll_fds.push_back(fd);
        else if (fd == -2) result.inspectable = false;
#else
        result.inspectable = false;
#endif
    }
    if (waiting) {
        result.gate = ProducerGate::Waiting;
        return result;
    }
    result.status = MGC_SUCCESS;
    result.gate = ProducerGate::Ready;
#else
    result.status = MGC_SUCCESS;
    result.gate = ProducerGate::Ready;
#endif
    return result;
}
#if MGC_HAVE_PRODUCER_WAKE
// Block until a producer dependency may have changed. External sync-file
// signals wake through the duplicated descriptors; cleanup completion,
// failure, cancellation and close wake through the session eventfd. A long
// watchdog bounds the error paths that have neither an fd nor a notify
// (device loss, an unverifiable fence) without reintroducing a 5 ms loop.
void wait_producer_progress(MgcSession *s, const std::vector<int> &fds, bool needs_backstop,
    uint64_t observed_epoch) noexcept
{
#if MGC_PRODUCER_WAKE_FD
    (void)observed_epoch;
    const int wake_fd = s->identity->wake.fd;
    if (wake_fd >= 0) {
        const int timeout = needs_backstop ? 5 : 1000;
        if (fds.size() + 1 <= 33) {
            // Common case: no heap allocation on the wait path.
            std::array<pollfd, 33> set{};
            size_t count = 0;
            set[count++] = pollfd{wake_fd, POLLIN, 0};
            for (const int fd : fds) if (fd >= 0) set[count++] = pollfd{fd, POLLIN, 0};
            (void)::poll(set.data(), static_cast<nfds_t>(count), timeout);
            if (set[0].revents & (POLLIN | POLLERR | POLLHUP)) {
                uint64_t value = 0;
                (void)::read(wake_fd, &value, sizeof(value));
            }
            return;
        }
        std::vector<pollfd> set;
        set.reserve(fds.size() + 1);
        set.push_back(pollfd{wake_fd, POLLIN, 0});
        for (const int fd : fds) if (fd >= 0) set.push_back(pollfd{fd, POLLIN, 0});
        (void)::poll(set.data(), static_cast<nfds_t>(set.size()), timeout);
        if (set[0].revents & (POLLIN | POLLERR | POLLHUP)) {
            uint64_t value = 0;
            (void)::read(wake_fd, &value, sizeof(value));
        }
        return;
    }
    std::unique_lock<std::mutex> lock(s->mutex);
    // Without an eventfd only accept/close signal s->work; a bounded 5 ms
    // wait keeps cleanup-gate completion from stalling up to the watchdog.
    s->work.wait_for(lock, std::chrono::milliseconds(needs_backstop || wake_fd < 0 ? 5 : 1000));
#else
    (void)fds;
    (void)needs_backstop;
    // The caller snapshotted the epoch before probing. Taking it here would
    // swallow a notification that arrived during the probe and force a full
    // timeout wait.
    std::unique_lock<std::mutex> lock(s->identity->wake.mutex);
    s->identity->wake.changed.wait_for(lock, std::chrono::milliseconds(5), [&] {
        return s->identity->wake.epoch.load(std::memory_order_acquire) != observed_epoch;
    });
#endif
}
#else
void wait_producer_progress(MgcSession *s, const std::vector<int> &, bool, uint64_t) noexcept
{
    std::unique_lock<std::mutex> lock(s->mutex);
    s->work.wait_for(lock, std::chrono::milliseconds(5));
}
#endif
#if defined(__ANDROID__)
inline void close_poll_fd(int fd) noexcept { if (fd >= 0) ::close(fd); }
#else
inline void close_poll_fd(int) noexcept {}
#endif
void worker(MgcSession *s)
{
    worker_session = s;
    // Accepted tickets stay in this worker-local list between probes. Only a
    // new acceptance takes the session lock, so a blocked ticket does not
    // bounce through s->queue on every wake. A blocked ticket remains counted
    // by s->in_flight, which keeps it alive and makes the reorder safe:
    // accept() still rejects a second ticket that references an in-flight
    // resource, so independent tickets are the only ones reordered.
    std::vector<MgcTicket *> pending;
    std::vector<int> poll_fds;
    for (;;)
    {
        {
            std::unique_lock<std::mutex> lock(s->mutex);
            if (s->queue.empty() && pending.empty())
            {
                s->work.wait(lock, [&] { return s->closing || !s->queue.empty(); });
                if (s->queue.empty())
                {
                    if (s->closing)
                        break;
                    continue;
                }
            }
            // Drain every newly accepted ticket before probing, so a newly
            // ready submission is always considered in this cycle.
            while (!s->queue.empty())
            {
                pending.push_back(s->queue.front());
                s->queue.pop_front();
            }
        }
        poll_fds.clear();
        MgcTicket *t = nullptr;
        ProducerGateResult gate;
        bool needs_backstop = false;
#if MGC_HAVE_PRODUCER_WAKE && !MGC_PRODUCER_WAKE_FD
        // Snapshot before probing: a notify during the probe must still be
        // observed by the wait, otherwise it is swallowed and the worker
        // waits a full timeout for an event that already happened.
        const uint64_t observed_epoch = s->identity->wake.epoch.load(std::memory_order_acquire);
#else
        const uint64_t observed_epoch = 0;
#endif
        for (size_t index = 0; index < pending.size(); ++index)
        {
            ProducerGateResult result = probe_producer_gate(pending[index], poll_fds);
            if (result.gate != ProducerGate::Waiting)
            {
                t = pending[index];
                gate = result;
                pending.erase(pending.begin() + index);
                break;
            }
            if (!result.inspectable) needs_backstop = true;
        }
        if (!t)
        {
            wait_producer_progress(s, poll_fds, needs_backstop, observed_epoch);
            for (const int fd : poll_fds) close_poll_fd(fd);
            continue;
        }
        for (const int fd : poll_fds) close_poll_fd(fd);
        uint64_t cpu_begin = 0, cpu_end = 0;
        const bool cpu_begin_observed = thread_cpu_ns(cpu_begin);
        MgcStatus status = gate.status;
        const bool producer_gate_skipped = gate.producer_gate_skipped;
        const bool dependency_failed = status != MGC_SUCCESS;
#if defined(MGC_TEST_GUARDED_HOST)
        if (!dependency_failed && t->guarded_submission) mgc_test_after_guard_ready(t);
#endif
        bool resuming = false;
        for (;;) {
            if (!t->cancelled.load() && !dependency_failed) {
                bool explicit_gpu = false;
#if MGC_HAVE_NCNN && NCNN_VULKAN
                explicit_gpu = s->domain->device && (t->host_flags & MGC_HOST_JOB_EXPLICIT_GPU_SCOPES);
#endif
                DomainScope scope(s->domain,false,!explicit_gpu);
                active_ticket = t;
                try {
                    const auto fn = resuming ? t->resume : t->execute;
                    auto admitted = scope.status;
#if MGC_HAVE_NCNN && NCNN_VULKAN
                    if(explicit_gpu) {
                        std::lock_guard<std::mutex> gate(s->domain->admission->mutex);
                        admitted=domain_admission(s->domain);
                    }
#endif
                    status = admitted ? admitted : fn ? fn(t->desc.context,t) : run_operation(t);
                } catch (const std::bad_alloc &) { status = MGC_OUT_OF_MEMORY; }
                catch (...) { status = MGC_INTERNAL_ERROR; }
                active_ticket = nullptr;
            }
            if (!t->deferred) break;
            // No device slot or session lock is held during retirement polls.
            // Cancellation cannot skip a poll or make this backoff spin.
            for (;;) {
#if MGC_HAVE_PRODUCER_WAKE && !MGC_PRODUCER_WAKE_FD
                const uint64_t retire_epoch = s->identity->wake.epoch.load(std::memory_order_acquire);
#else
                const uint64_t retire_epoch = 0;
#endif
                MgcStatus retired = MGC_INTERNAL_ERROR;
                try { retired = t->retire(t->retire_context,t); } catch (...) {}
                if (retired == MGC_SUCCESS) break;
                if (retired == MGC_DEVICE_LOST) { t->deferred_failure = retired; t->resume_allowed = false; }
                // Retirement is a bounded poll; a wake (completion, cancel,
                // close) or the 5 ms backstop decides when to retry.
                wait_producer_progress(s, {}, true, retire_epoch);
            }
            t->deferred = false;
            t->retire = nullptr; t->retire_context = nullptr;
            if (t->cancelled.load() || !t->resume_allowed || !t->resume) {
                status = t->deferred_failure;
                break;
            }
            resuming = true;
        }
        t->operation_state.reset();
        invoke_context(t->finalize,t->desc.context);
        // For guarded work, serialize the cancellation decision with ticket
        // completion before dropping any resource's in-flight protection.
        // No caller or producer callback executes while this lock is held.
        std::unique_lock<std::mutex> completion_lock(t->mutex, std::defer_lock);
        bool guarded_cancelled = false;
#if MGC_HAVE_PRODUCER_WAKE
        if (t->guarded_submission) {
            completion_lock.lock();
            guarded_cancelled = t->cancelled.load();
        }
#endif
        if(status==MGC_DEVICE_LOST||status==MGC_PRODUCER_FAILED||
            (dependency_failed && (status != MGC_CANCELLED || producer_gate_skipped)) || guarded_cancelled) {
            for (auto *lease : t->leases) {
                std::lock_guard<std::mutex> lock(lease->resource->mutex);
                if (lease->desc.access == MGC_ACCESS_WRITE) {
                    lease->resource->content_invalid = true;
                    lease->resource->content_failure = guarded_cancelled && status != MGC_DEVICE_LOST &&
                        status != MGC_PRODUCER_FAILED ? MGC_CANCELLED : status;
                }
            }
        }
        const bool cpu_end_observed = thread_cpu_ns(cpu_end);
        MgcCompletion result{};
        if (cpu_begin_observed && cpu_end_observed && cpu_end >= cpu_begin)
        {
            result.worker_cpu_ns = cpu_end - cpu_begin;
            result.worker_cpu_observed = 1;
        }
        // Physical work is retired. Keep submission references through callback,
        // while allowing a completion consumer to read/publish the finished data.
        for (auto *l : t->leases)
        {
            std::lock_guard<std::mutex> lock(l->resource->mutex);
            --l->resource->in_flight;
#if MGC_HAVE_PRODUCER_WAKE
            unregister_producer_waiter_locked(l->resource, producer_wake_alias(t->session_identity));
#endif
        }
        result.header = MGC_HEADER_INIT(MgcCompletion);
        result.request_id = t->desc.request_id;
        result.execution_retired = 1;
        result.host_finalized = 1;
        result.gpu_batch_count = t->submitted_batches.load();
        result.gpu_observed_batch_count = t->gpu_observed_batches;
        result.gpu_timestamp_valid_bits = t->gpu_bits;
        result.gpu_timestamp_period_bits = t->gpu_period;
        result.gpu_elapsed_observed = (t->measurement_flags & MGC_MEASURE_GPU_TIMESTAMPS) && t->gpu_complete &&
            !t->gpu_pending && result.gpu_batch_count && result.gpu_batch_count == t->gpu_observed_batches;
        if (result.gpu_elapsed_observed) result.gpu_elapsed_ns = t->gpu_ns;
        record_event(t, MGC_EVENT_EXECUTION_RETIRED, 0);
        result.batch_event_coverage_observed = t->desc.batch_event && t->event_complete && !t->gpu_pending;
        {
            if (!completion_lock.owns_lock()) completion_lock.lock();
            const bool cancelled = t->cancelled.load();
            if (cancelled)
                result.result_discarded = 1;
            // Cancellation suppresses publication, but must not hide a device
            // failure which occurred while the accepted job was executing.
            if (cancelled && status != MGC_DEVICE_LOST && status != MGC_PRODUCER_FAILED)
            {
                result.terminal = MGC_TERMINAL_CANCELLED;
                result.status = MGC_CANCELLED;
            }
            else
            {
                result.status = status;
                result.terminal = status == MGC_SUCCESS       ? MGC_TERMINAL_SUCCESS
                                  : status == MGC_DEVICE_LOST ? MGC_TERMINAL_DEVICE_LOST
                                                              : MGC_TERMINAL_FAILED;
            }
            t->result = result;
            t->done = true;
        }
        completion_lock.unlock();
        if (t->desc.batch_event) for (const auto &event : t->events) {
            try { t->desc.batch_event(t->desc.context, &event); } catch (...) {}
        }
        if (t->desc.completion)
        {
            try
            {
                t->desc.completion(t->desc.context, &result);
            }
            catch (...)
            {
            }
        }
        invoke_context(t->desc.release_context, t->desc.context);
        release_submission(t);
        {
            std::lock_guard<std::mutex> lock(s->mutex);
            --s->in_flight;
        }
        s->retired.notify_all();
        mgc_ticket_release(t);
    }
    worker_session = nullptr;
}
MgcStatus accept(MgcSession *s, std::unique_ptr<MgcTicket> &t, MgcTicket **out)
{
#if MGC_HAVE_NCNN && NCNN_VULKAN
    if (s->domain->admission) {
        std::lock_guard<std::mutex> gate(s->domain->admission->mutex);
        const auto status = domain_admission(s->domain);
        if (status) return status;
    }
#endif
    for (auto *l : t->leases)
    {
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
        reap_external_ready(l->resource);
#endif
        auto rc = validate_lease(s, l);
        if (rc != MGC_SUCCESS)
            return rc;
    }
    std::vector<MgcResource *> resources;
    for (auto *l : t->leases)
        resources.push_back(l->resource);
    std::sort(resources.begin(), resources.end());
    resources.erase(std::unique(resources.begin(), resources.end()), resources.end());
    std::vector<std::unique_lock<std::mutex>> locks;
    for (auto *r : resources)
    {
        locks.emplace_back(r->mutex, std::try_to_lock);
        if (!locks.back().owns_lock()) return MGC_BUSY;
    }
    for (auto *r : resources)
        if (r->maps || r->in_flight)
            return MGC_BUSY;
        else if (r->content_invalid) return r->content_failure;
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
    for (auto *l : t->leases)
    {
        auto *r = l->resource;
        if (r->ownership_uncertain || r->external_transition || r->external_access_state == 1 || r->external_access_state == 2 ||
            (!meguviz_compute_detail::same_producer_content(r->external_access_state, l->desc.content_id, r->current_content))) return MGC_BUSY;
    }
#endif
#if defined(MGC_TEST_GUARDED_HOST)
    for (auto *l : t->leases) {
        auto *r = l->resource;
        if (r->test_guard && !meguviz_compute_detail::same_producer_content(3, l->desc.content_id, r->current_content))
            return MGC_BUSY;
    }
#endif
    std::unique_lock<std::mutex> lock(s->mutex);
    if (s->closing)
        return MGC_BUSY;
    if (s->in_flight >= s->max_in_flight)
        return MGC_BACKPRESSURE;
    t->session_identity = s->identity;
    t->domain = s->domain;
    s->queue.push_back(t.get());
#if MGC_HAVE_PRODUCER_WAKE
    const auto producer_wake = producer_wake_alias(s->identity);
    for (auto *resource : resources) {
#if defined(MGC_TEST_GUARDED_HOST)
        if (resource_guarded_content_locked(resource)) t->guarded_submission = true;
        if (resource->test_guard) register_producer_waiter_locked(resource, producer_wake);
#else
        if (resource->ahb) {
            if (resource_guarded_content_locked(resource)) t->guarded_submission = true;
            register_producer_waiter_locked(resource, producer_wake);
        }
#endif
    }
#endif
    t->measurement_flags = s->domain->measurement_flags.load();
    record_event(t.get(), MGC_EVENT_ACCEPTED, 0);
    ++s->in_flight;
    for (auto *l : t->leases)
        ++l->resource->in_flight;
    t->refs.store(2);
    *out = t.release();
    lock.unlock();
    locks.clear();
    s->work.notify_one();
#if MGC_HAVE_PRODUCER_WAKE
    // The worker may be blocked on a dependency poll instead of s->work.
    s->identity->wake.notify();
#endif
    return MGC_ACCEPTED;
}
} // namespace
extern "C"
{
#if defined(MGC_TEST_GUARDED_HOST)
    // Test-only hooks: exercise real host accept/worker semantics without an
    // Android device. No symbols or fields exist in production builds.
    MgcStatus mgc_test_guard_arm(MgcLease *lease, uint64_t generation)
    {
        if (!lease || !generation) return MGC_INVALID_ARGUMENT;
        auto *r = lease->resource;
        std::lock_guard<std::mutex> lock(r->mutex);
        if (r->test_guard || r->test_producer_pin) return MGC_BUSY;
        r->test_generation = generation; r->test_guard = 1; r->test_fence_status = MGC_BUSY;
        set_guarded_content_locked(r, true);
        r->test_producer_pin = lease; mgc_lease_retain(lease);
        return MGC_SUCCESS;
    }
    MgcStatus mgc_test_guard_fence(MgcLease *lease, uint64_t generation, MgcStatus status)
    {
        if (!lease || (status != MGC_SUCCESS && status != MGC_BUSY && status != MGC_PRODUCER_FAILED && status != MGC_INTERNAL_ERROR))
            return MGC_INVALID_ARGUMENT;
        auto *r = lease->resource; MgcLease *pin = nullptr; ProducerWakeList wake{};
        {
            std::lock_guard<std::mutex> lock(r->mutex);
            if (!r->test_producer_pin || r->test_generation != generation || r->test_producer_pin != lease) return MGC_INVALID_ARGUMENT;
            r->test_fence_status = status;
            if (status == MGC_PRODUCER_FAILED) { r->content_invalid = true; r->content_failure = status; r->test_guard = 3; }
            else if (status == MGC_SUCCESS && r->test_guard == 2) {
                pin = r->test_producer_pin; r->test_producer_pin = nullptr; r->test_guard = 0;
            }
            wake = producer_waiters_locked(r);
        }
        notify_producer_waiters(wake);
        if (pin) mgc_lease_release(pin);
        return MGC_SUCCESS;
    }
    MgcStatus mgc_test_guard_finish(MgcLease *lease, uint64_t generation, MgcStatus status)
    {
        if (!lease) return MGC_INVALID_ARGUMENT;
        auto *r = lease->resource; MgcLease *pin = nullptr; ProducerWakeList wake{};
        {
            std::lock_guard<std::mutex> lock(r->mutex);
            if (!r->test_producer_pin || r->test_producer_pin != lease) return MGC_INVALID_ARGUMENT;
            const auto rc = meguviz_compute_detail::finish_producer_guard(r->test_guard, r->test_generation,
                generation, status, r->content_invalid, r->content_failure);
            if (rc != MGC_SUCCESS) return rc;
            if (r->test_guard == 2 && r->test_fence_status == MGC_SUCCESS) {
                pin = r->test_producer_pin; r->test_producer_pin = nullptr; r->test_guard = 0;
            }
            wake = producer_waiters_locked(r);
        }
        notify_producer_waiters(wake);
        if (pin) mgc_lease_release(pin);
        return MGC_SUCCESS;
    }
    uint32_t mgc_test_guard_pin(MgcLease *lease)
    {
        if (!lease) return 0;
        std::lock_guard<std::mutex> lock(lease->resource->mutex);
        return lease->resource->test_producer_pin ? 1u : 0u;
    }
    uint32_t mgc_test_guard_waiters(MgcLease *lease)
    {
        if (!lease) return 0;
        std::lock_guard<std::mutex> lock(lease->resource->mutex);
        uint32_t count = 0;
        for (const auto &waiter : lease->resource->producer_waiters)
            if (!waiter.expired()) ++count;
        return count;
    }
#endif
    MgcStatus mgc_query_build_info(MgcBuildInfo *info)
    {
        MgcBuildInfo v{};
        v.header = MGC_HEADER_INIT(MgcBuildInfo);
        v.provider = MGC_HAVE_NCNN ? MGC_PROVIDER_NCNN_CPU : MGC_PROVIDER_CPU_REFERENCE;
        hash_from_hex(MGC_PRIVATE_FP, v.private_sdk_fingerprint);
        hash_from_hex(MGC_BUILD_FP, v.build_fingerprint);
        return output_struct(info, &v, sizeof(v), sizeof(MgcHeader));
    }
    MgcStatus mgc_runtime_create(const MgcRuntimeDesc *input, MgcRuntime **out)
    {
        if (!out)
            return MGC_INVALID_ARGUMENT;
        *out = nullptr;
        MgcRuntimeDesc d{};
        auto rc = read_desc(input, d, offsetof(MgcRuntimeDesc, reserved));
        if (rc)
            return rc;
        // Optional tail scalars are either entirely absent or fully present.
        // A byte-prefix must not silently enable a partial measurement flag.
        const auto size = input->header.struct_size;
        if ((size > offsetof(MgcRuntimeDesc, measurement_flags) &&
             size < offsetof(MgcRuntimeDesc, measurement_flags) + sizeof(d.measurement_flags)) ||
            (size > offsetof(MgcRuntimeDesc, measurement_reserved) &&
             size < offsetof(MgcRuntimeDesc, measurement_reserved) + sizeof(d.measurement_reserved)))
            return MGC_UNSUPPORTED_ABI;
        rc = flags(d.required_flags, d.reserved, sizeof(d.reserved));
        if (rc)
            return rc;
        if (d.measurement_reserved || (d.measurement_flags & ~MGC_MEASURE_GPU_TIMESTAMPS)) return MGC_UNSUPPORTED_ABI;
        if (d.provider != MGC_PROVIDER_CPU_REFERENCE && d.provider != MGC_PROVIDER_NCNN_CPU &&
            d.provider != MGC_PROVIDER_NCNN_VULKAN)
            return MGC_UNSUPPORTED_CAPABILITY;
#if !MGC_HAVE_NCNN
        if (d.provider != MGC_PROVIDER_CPU_REFERENCE)
            return MGC_UNSUPPORTED_CAPABILITY;
#elif !NCNN_VULKAN
        if (d.provider == MGC_PROVIDER_NCNN_VULKAN)
            return MGC_UNSUPPORTED_CAPABILITY;
#endif
        try
        {
            auto *r = new MgcRuntime;
            r->provider = d.provider;
            r->budget = d.memory_budget_bytes ? d.memory_budget_bytes : 64 * 1024 * 1024;
            r->measurement_flags = d.measurement_flags;
            r->allocation_stats.resource_ranges_observed = 1;
            *out = r;
            return MGC_SUCCESS;
        }
        catch (...)
        {
            return MGC_OUT_OF_MEMORY;
        }
    }
    void mgc_runtime_retain(MgcRuntime *r)
    {
        if (r)
            ++r->refs;
    }
    void mgc_runtime_release(MgcRuntime *r)
    {
        if (r && --r->refs == 0)
            delete r;
    }
    MgcStatus mgc_runtime_close(MgcRuntime *r)
    {
        if (!r)
            return MGC_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> lock(r->mutex);
        r->closed = true;
        return r->domains||r->preparing_domains||r->pending_initializations ? MGC_BUSY : MGC_SUCCESS;
    }
    MgcStatus mgc_domain_acquire(MgcRuntime *r, const MgcDomainDesc *input, MgcDomain **out)
    {
        if (!r || !out)
            return MGC_INVALID_ARGUMENT;
        *out = nullptr;
        MgcDomainDesc d{};
        auto rc = read_desc(input, d, offsetof(MgcDomainDesc, reserved));
        if (rc)
            return rc;
        rc = flags(d.required_flags, d.reserved, sizeof(d.reserved));
        if (rc)
            return rc;
        MgcHash fp{};
        hash_from_hex(MGC_PRIVATE_FP, fp);
        if (!zeros(&d.private_sdk_fingerprint, sizeof(fp)) &&
            std::memcmp(&fp, &d.private_sdk_fingerprint, sizeof(fp)))
            return MGC_ARTIFACT_MISMATCH;
        DomainPreparation preparation;
        rc=preparation.acquire(r);if(rc)return rc;
        try
        {
            auto discard = [](MgcDomain *domain) {
#if MGC_HAVE_NCNN && NCNN_VULKAN
                cleanup_domain_device(domain);
#endif
                delete domain;
            };
            std::unique_ptr<MgcDomain, decltype(discard)> dom(new MgcDomain, discard);
            { std::lock_guard<std::mutex> lock(r->mutex); if (r->closed) return MGC_BUSY; }
#if MGC_HAVE_NCNN && NCNN_VULKAN
            if (r->provider == MGC_PROVIDER_NCNN_VULKAN)
            {
                // Reject cross-slot nesting before attaching an instance; its
                // rollback would otherwise have to wait for the GPU slot.
                if (active_execution_slot && active_execution_slot != &execution_slot)
                    return MGC_INCOMPATIBLE_DOMAIN;
                // Device construction and initialization polling both touch
                // ncnn's mutable initialization command. Serialize them under
                // the execution slot, then take instance_mutex in that order.
                std::lock_guard<std::recursive_mutex> slot(execution_slot);
                {
                    std::lock_guard<std::mutex> gpu(instance_mutex);
                    if (!instance_users && ncnn::create_gpu_instance() != 0)
                        return MGC_UNSUPPORTED_CAPABILITY;
                    if (d.device_index >= static_cast<uint32_t>(ncnn::get_gpu_count())) {
                        if (!instance_users) ncnn::destroy_gpu_instance();
                        return MGC_UNSUPPORTED_CAPABILITY;
                    }
                    ++instance_users;
                    dom->instance_owned = true;
                    attach_cached_device_locked(dom.get(),d.device_index,r);
                }
                for(unsigned attempt=0;attempt<2;++attempt){
                    if(!dom->device)return MGC_UNSUPPORTED_CAPABILITY;
                    if(!dom->device->is_valid()){
                        const auto failure=dom->device->initialization_result();
                        if(failure==VK_ERROR_DEVICE_LOST)report_device_lost(dom.get());
                        else if(dom->device->initialization_pending())return MGC_BUSY;
                        else {
                            dom->admission->quiesced=true;
                            detach_retired_device(dom->admission);
                            return failure==VK_ERROR_OUT_OF_HOST_MEMORY||failure==VK_ERROR_OUT_OF_DEVICE_MEMORY?MGC_OUT_OF_MEMORY:MGC_INTERNAL_ERROR;
                        }
                    }
                    if(!domain_is_lost(dom.get()))break;
                    service_device_loss(dom.get());
                    if(!dom->admission->detached.load())return dom->admission->quiesced.load()?MGC_BACKPRESSURE:MGC_BUSY;
                    if(dom->admission->factory_pin)service_factory(dom->admission);
                    if(attempt==1)return MGC_BUSY;
                    {std::lock_guard<std::mutex> instance(instance_mutex);attach_cached_device_locked(dom.get(),d.device_index,r);}
                }
                const auto admitted=domain_admission(dom.get());if(admitted)return admitted;
                dom->generation=dom->admission->epoch;
                dom->blob_allocator = dom->device->acquire_blob_allocator();
                dom->staging_allocator = dom->device->acquire_staging_allocator();
                if (!dom->blob_allocator || !dom->staging_allocator) return MGC_OUT_OF_MEMORY;
            }
#endif
            if (r->provider != MGC_PROVIDER_NCNN_VULKAN && d.device_index)
                return MGC_INVALID_ARGUMENT;
            dom->runtime = r;
            dom->id = next_domain++;
            dom->device_index = d.device_index;
            dom->measurement_flags = r->measurement_flags;
            dom->allocation_stats.resource_ranges_observed = 1;
            std::lock_guard<std::mutex> lock(r->mutex);
            if (r->closed) return MGC_BUSY;
            mgc_runtime_retain(r);
            ++r->domains;
            *out = dom.release();
            return MGC_SUCCESS;
        }
        catch (const std::bad_alloc&) { return MGC_OUT_OF_MEMORY; }
        catch (...) { return MGC_INTERNAL_ERROR; }
    }
    MgcStatus mgc_domain_identity(MgcDomain *d, uint64_t *id, uint64_t *gen)
    {
        if (!d || !id || !gen)
            return MGC_INVALID_ARGUMENT;
        *id = d->id;
        *gen = d->generation;
        return MGC_SUCCESS;
    }
    MgcStatus mgc_domain_state(MgcDomain *d,MgcDomainStateInfo *out)
    {
        if(!d||!out)return MGC_INVALID_ARGUMENT;
        MgcDomainStateInfo value{};value.header=MGC_HEADER_INIT(MgcDomainStateInfo);
        value.domain_id=d->id;value.domain_generation=d->generation;
#if MGC_HAVE_NCNN && NCNN_VULKAN
        if(d->admission){value.device_epoch=d->admission->epoch;value.lost=d->admission->lost.load();value.device_retired=d->admission->quiesced.load()&&value.lost;value.detached=d->admission->detached.load();value.last_error=d->admission->last_error.load();}
#endif
        constexpr size_t minimum=offsetof(MgcDomainStateInfo,reserved);
        if(!header_ok(&out->header,minimum)||(out->header.struct_size>minimum&&out->header.struct_size<sizeof(value)&&
            (out->header.struct_size-minimum)%sizeof(uint32_t)))return MGC_UNSUPPORTED_ABI;
        return output_struct(out,&value,sizeof(value),minimum);
    }
    MgcStatus mgc_domain_service(MgcDomain *d,uint64_t wait_ns)
    {
        if(wait_ns>100000000)return MGC_INVALID_ARGUMENT;
#if MGC_HAVE_NCNN && NCNN_VULKAN
        try{
            if(d){auto status=service_device_loss(d);if(status!=MGC_SUCCESS)return status;return service_standalone_gpu(d,wait_ns);}
            if(active_execution_slot&&active_execution_slot!=&execution_slot)return MGC_INCOMPATIBLE_DOMAIN;
            std::vector<std::shared_ptr<DeviceAdmission>> states;
            struct Pin {bool held=false;~Pin(){if(held){
                // A different owner keeps the instance alive: return this
                // temporary reference without waiting for a busy GPU slot.
                {std::lock_guard<std::mutex> lock(instance_mutex);if(instance_users>1){--instance_users;return;}}
                // The final reference may destroy shared Vulkan state, which
                // must still use the established execution-slot lock order.
                std::lock_guard<std::recursive_mutex> slot(execution_slot);std::lock_guard<std::mutex> lock(instance_mutex);
                if(!--instance_users){device_admissions.clear();current_device_admissions.clear();ncnn::destroy_gpu_instance();}
            }}} pin;
            {std::lock_guard<std::mutex> lock(instance_mutex);if(!instance_users)return MGC_SUCCESS;
                for(const auto& entry:current_device_admissions)if(entry.second)states.push_back(entry.second);
                for(const auto& entry:device_admissions)if(entry.second&&std::find(states.begin(),states.end(),entry.second)==states.end())states.push_back(entry.second);
                ++instance_users;pin.held=true;}
            MgcStatus result=MGC_SUCCESS;
            for(auto& state:states){if(service_loss_state(state)==MGC_BUSY||service_factory(state)==MGC_BUSY)result=MGC_BUSY;}
            const auto standalone=service_standalone_gpu(nullptr,wait_ns);
            if(standalone!=MGC_SUCCESS)result=standalone;
            return result;
        }catch(const std::bad_alloc&){return MGC_OUT_OF_MEMORY;}catch(...){return MGC_INTERNAL_ERROR;}
#else
        (void)d;return MGC_SUCCESS;
#endif
    }
    void mgc_domain_retain(MgcDomain *d)
    {
        if (d)
            ++d->refs;
    }
    MgcStatus mgc_domain_set_measurement_flags(MgcDomain *d, uint32_t measurement_flags)
    { if (!d) return MGC_INVALID_ARGUMENT; if (measurement_flags & ~MGC_MEASURE_GPU_TIMESTAMPS) return MGC_UNSUPPORTED_ABI; d->measurement_flags = measurement_flags; return MGC_SUCCESS; }
    MgcStatus mgc_runtime_allocation_stats(MgcRuntime *r, MgcAllocationStats *out)
    {
        if (!r) return MGC_INVALID_ARGUMENT; std::lock_guard<std::mutex> lock(r->mutex);
        auto stats = r->allocation_stats; stats.header = MGC_HEADER_INIT(MgcAllocationStats);
        return output_struct(out, &stats, sizeof(stats), sizeof(MgcHeader));
    }
    MgcStatus mgc_domain_allocation_stats(MgcDomain *d, MgcAllocationStats *out)
    {
        if (!d) return MGC_INVALID_ARGUMENT; std::lock_guard<std::mutex> lock(d->runtime->mutex);
        auto stats = d->allocation_stats; stats.header = MGC_HEADER_INIT(MgcAllocationStats); stats.domain_id = d->id;
        return output_struct(out, &stats, sizeof(stats), sizeof(MgcHeader));
    }
    void mgc_domain_release(MgcDomain *d)
    {
        if (!d || --d->refs)
            return;
#if MGC_HAVE_NCNN && NCNN_VULKAN
        cleanup_domain_device(d);
#endif
        auto *r = d->runtime;
        {
            std::lock_guard<std::mutex> lock(r->mutex);
            --r->domains;
        }
        delete d;
        mgc_runtime_release(r);
    }
    MgcStatus mgc_domain_capability(MgcDomain *d, uint32_t feature, MgcCapability *out)
    {
        if (!d || feature < 1 || feature > MGC_CAP_GUARDED_PRODUCER_FENCE)
            return MGC_INVALID_ARGUMENT;
        MgcCapability v{};
        v.header = MGC_HEADER_INIT(MgcCapability);
        v.feature = feature;
        v.probe_result = MGC_UNSUPPORTED_CAPABILITY;
        v.reason = MGC_UNSUPPORTED_CAPABILITY;
        if (feature == MGC_CAP_CPU_REFERENCE || feature == MGC_CAP_HOST_JOB_ASYNC || feature == MGC_CAP_EXTERNAL_HOST_MEMORY)
        {
            v.advertised = v.enabled = v.workload_qualified = 1;
            v.probe_result = v.reason = MGC_SUCCESS;
        }
#if MGC_HAVE_NCNN && NCNN_VULKAN
        if (feature == MGC_CAP_GPU_TIMESTAMP && d->device) {
            try {
                uint32_t count = 0; ncnn::vkGetPhysicalDeviceQueueFamilyProperties(d->device->info.physicalDevice(), &count, nullptr);
                std::vector<VkQueueFamilyProperties> queues(count);
                if (count) ncnn::vkGetPhysicalDeviceQueueFamilyProperties(d->device->info.physicalDevice(), &count, queues.data());
                uint32_t family = d->device->info.compute_queue_family_index();
                v.advertised = family < count && queues[family].timestampValidBits > 0 && queues[family].timestampValidBits <= 64;
                v.enabled = v.advertised && (d->measurement_flags.load() & MGC_MEASURE_GPU_TIMESTAMPS);
                if (v.enabled) v.probe_result = v.reason = d->timestamp_probe.load();
            } catch (...) { return MGC_OUT_OF_MEMORY; }
        }
        if (feature == MGC_CAP_VULKAN_COMPUTE && d->device)
        {
            v.advertised = v.enabled = 1;
            v.probe_result = v.reason = d->vulkan_submission_probe.load();
        }
        if (d->device && feature == MGC_CAP_FINAL_SUBMIT_ASYNC)
        { v.advertised = v.enabled = 1; v.probe_result = v.reason = d->vulkan_submission_probe.load(); }
        if (d->device && (feature == MGC_CAP_FP16_STORAGE || feature == MGC_CAP_FP16_ARITHMETIC)) {
            v.advertised = feature == MGC_CAP_FP16_STORAGE ? d->device->info.support_fp16_storage() : d->device->info.support_fp16_arithmetic();
            struct FeatureHeader { VkStructureType sType; const void *pNext; };
            for (auto *item = static_cast<const FeatureHeader *>(d->device->enabled_extension_features()); item; item = static_cast<const FeatureHeader *>(item->pNext)) {
                if (feature == MGC_CAP_FP16_STORAGE && item->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR)
                    v.enabled = reinterpret_cast<const VkPhysicalDevice16BitStorageFeaturesKHR *>(item)->storageBuffer16BitAccess;
                if (feature == MGC_CAP_FP16_ARITHMETIC && item->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR)
                    v.enabled = reinterpret_cast<const VkPhysicalDeviceFloat16Int8FeaturesKHR *>(item)->shaderFloat16;
            }
            if (v.enabled && v.advertised) v.probe_result = v.reason = MGC_BUSY;
            else v.enabled = 0;
        }
#if defined(__ANDROID__)
        if (d->device && feature == MGC_CAP_GUARDED_PRODUCER_FENCE && MGC_ENABLE_GUARDED_PRODUCER_FENCE) {
            v.advertised = v.enabled = d->device->info.support_VK_ANDROID_external_memory_android_hardware_buffer() &&
                d->device->info.support_VK_EXT_queue_family_foreign();
            if (v.enabled) v.probe_result = v.reason = MGC_SUCCESS;
        }
        if (d->device && (feature == MGC_CAP_AHB_IMPORT || feature == MGC_CAP_VULKAN_AHB_SAMPLEABLE || feature == MGC_CAP_VULKAN_AHB_STORAGE))
        {
            v.advertised = v.enabled = d->device->info.support_VK_ANDROID_external_memory_android_hardware_buffer() && d->device->info.support_VK_EXT_queue_family_foreign();
            if (v.enabled) v.probe_result = v.reason = MGC_BUSY;
        }
        if (d->device && (feature == MGC_CAP_SYNC_FD_IMPORT || feature == MGC_CAP_SYNC_FD_EXPORT))
        {
            v.advertised = v.enabled = d->device->meguviz_sync_fd_enabled && d->device->vkImportSemaphoreFdKHR && d->device->vkGetSemaphoreFdKHR;
            if (v.enabled && !domain_is_lost(d)) v.probe_result = v.reason = probe_sync_fd(d);
        }
#endif
        if (d->device && domain_is_lost(d) && feature != MGC_CAP_CPU_REFERENCE && feature != MGC_CAP_HOST_JOB_ASYNC &&
            feature != MGC_CAP_EGL_AHB_RENDERABLE && feature != MGC_CAP_EGL_AHB_SAMPLEABLE) {
            v.enabled = v.workload_qualified = 0;
            v.probe_result = v.reason = MGC_DEVICE_LOST;
        }
#endif
        return output_struct(out, &v, sizeof(v), offsetof(MgcCapability, reserved));
    }
    static MgcStatus resource_create_impl(MgcDomain *d, const MgcResourceDesc *input,
                                         const MgcHostMemoryDesc *external, MgcResource **out)
    {
        if (!d || !out)
            return MGC_INVALID_ARGUMENT;
        *out = nullptr;
        MgcResourceDesc v{};
        auto rc = read_desc(input, v, offsetof(MgcResourceDesc, reserved));
        if (rc)
            return rc;
        rc = flags(v.required_flags, v.reserved, sizeof(v.reserved));
        if (rc)
            return rc;
        if (!v.capacity_bytes || v.capacity_bytes > SIZE_MAX || v.format < 1 || v.format > 4 ||
            v.usage > MGC_RESOURCE_GPU_BUFFER)
            return MGC_INVALID_ARGUMENT;
        if (external && (v.usage != 0 || external->capacity_bytes != v.capacity_bytes))
            return MGC_INVALID_ARGUMENT;
        if (v.usage == MGC_RESOURCE_GPU_BUFFER &&
            (d->runtime->provider != MGC_PROVIDER_NCNN_VULKAN || v.capacity_bytes % 4 ||
             v.capacity_bytes / 4 > INT32_MAX))
            return MGC_UNSUPPORTED_CAPABILITY;
        try
        {
            // Workers hold the execution slot before creating derived resources.
            // Match that order before taking the domain allocation identity lock.
            DomainScope slot(d, false, v.usage == MGC_RESOURCE_GPU_BUFFER);
            if (slot.status) return slot.status;
            auto res = std::make_unique<MgcResource>();
            if (!v.allocation_generation)
                v.allocation_generation = 1;
            std::unique_lock<std::mutex> identity(d->allocations_mutex);
            if (!v.allocation_id)
            {
                do
                {
                    v.allocation_id = next_allocation++;
                    if (!v.allocation_id)
                        return MGC_OUT_OF_MEMORY;
                } while (d->allocations.count({v.allocation_id, v.allocation_generation}));
            }
            else if (d->allocations.count({v.allocation_id, v.allocation_generation}))
                return MGC_INVALID_ARGUMENT;
            auto *r = d->runtime;
            {
                std::lock_guard<std::mutex> lock(r->mutex);
                if (r->closed)
                    return MGC_BUSY;
                if (v.capacity_bytes > r->budget - r->live_bytes)
                    return MGC_OUT_OF_MEMORY;
                r->live_bytes += v.capacity_bytes;
            }
            try
            {
                if (v.usage == MGC_RESOURCE_GPU_BUFFER)
                {
#if MGC_HAVE_NCNN && NCNN_VULKAN
                    res->gpu_buffer.create(static_cast<int>(v.capacity_bytes / 4), size_t(4), 1, d->blob_allocator);
                    if (res->gpu_buffer.empty()) throw std::bad_alloc();
#else
                    throw std::bad_alloc();
#endif
                }
                else if (!external) res->data.resize(static_cast<size_t>(v.capacity_bytes));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(r->mutex);
                r->live_bytes -= v.capacity_bytes;
                throw;
            }
            res->accounted_bytes = external ? v.capacity_bytes : res->data.capacity();
#if MGC_HAVE_NCNN && NCNN_VULKAN
            if (v.usage == MGC_RESOURCE_GPU_BUFFER) res->accounted_bytes = res->gpu_buffer.data->capacity;
#endif
            // The allocator may align a requested range. Only publish the
            // resource after that actual range fits the registered budget.
            res->budget_bytes = res->accounted_bytes;
            if (res->accounted_bytes != v.capacity_bytes) {
                std::lock_guard<std::mutex> lock(r->mutex);
                if (res->accounted_bytes > v.capacity_bytes &&
                    res->accounted_bytes - v.capacity_bytes > r->budget - r->live_bytes)
                {
                    r->live_bytes -= v.capacity_bytes;
                    return MGC_OUT_OF_MEMORY;
                }
                r->live_bytes -= v.capacity_bytes;
                r->live_bytes += res->budget_bytes;
            }
            try
            {
                d->allocations.insert({v.allocation_id, v.allocation_generation});
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(r->mutex);
                r->live_bytes -= res->budget_bytes;
                throw;
            }
            res->domain = d;
            res->desc = v;
            if (external) {
                res->external_data = static_cast<unsigned char *>(external->data);
                res->external_access = external->access;
                res->external_context = external->context;
                res->external_release = external->release;
                res->imported_allocation = true;
            }
            { std::lock_guard<std::mutex> lock(r->mutex); allocation_change(d, res->accounted_bytes, res->imported_allocation, true); }
            mgc_domain_retain(d);
            *out = res.release();
            return MGC_SUCCESS;
        }
        catch (...)
        {
            return MGC_OUT_OF_MEMORY;
        }
    }
    MgcStatus mgc_resource_create(MgcDomain *d, const MgcResourceDesc *input, MgcResource **out)
    {
        return resource_create_impl(d, input, nullptr, out);
    }
    MgcStatus mgc_resource_import_host_v1(MgcDomain *d, const MgcResourceDesc *input,
                                          const MgcHostMemoryDesc *external, MgcResource **out)
    {
        if (!out) return MGC_INVALID_ARGUMENT;
        *out = nullptr;
        MgcHostMemoryDesc v{};
        MgcStatus status = read_desc(external, v, sizeof(MgcHostMemoryDesc));
        if (status != MGC_SUCCESS) return status;
        status = flags(v.required_flags, v.reserved, sizeof(v.reserved));
        if (status != MGC_SUCCESS) return status;
        if (!v.data || !v.capacity_bytes || !v.retain || !v.release ||
            (v.access != MGC_ACCESS_READ && v.access != MGC_ACCESS_WRITE)) return MGC_INVALID_ARGUMENT;
        try { v.retain(v.context); }
        catch (...) { return MGC_INTERNAL_ERROR; }
        status = resource_create_impl(d, input, &v, out);
        if (status != MGC_SUCCESS) {
            try { v.release(v.context); }
            catch (...) { return MGC_INTERNAL_ERROR; }
        }
        return status;
    }
    void mgc_resource_retain(MgcResource *r)
    {
        if (r)
            ++r->refs;
    }
    void mgc_resource_release(MgcResource *r)
    {
        if (!r || --r->refs)
            return;
        auto *d = r->domain;
#if MGC_HAVE_NCNN && NCNN_VULKAN
        if (r->desc.usage == MGC_RESOURCE_GPU_BUFFER) {
            std::lock_guard<std::recursive_mutex> slot(execution_slot);
            r->gpu_buffer.release();
        }
#if defined(__ANDROID__)
        // AHB owns an independent allocator/import, and a final reference is
        // impossible while a task or external reservation still pins it.
        // Vulkan object destruction therefore does not need an unrelated NN's
        // shared VkMat allocator slot.
        if (r->ahb) {
            r->gpu_image.release();
            r->ahb_allocator.reset();
            if (r->release_fd >= 0) close(r->release_fd);
            if (r->ahb) AHardwareBuffer_release(r->ahb);
        }
#endif
#endif
        {
            std::lock_guard<std::mutex> identity(d->allocations_mutex);
            d->allocations.erase({r->desc.allocation_id, r->desc.allocation_generation});
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
            if (r->ahb) d->ahb_allocations.erase(r->ahb);
#endif
        }
        {
            std::lock_guard<std::mutex> lock(d->runtime->mutex);
            d->runtime->live_bytes -= r->budget_bytes;
            allocation_change(d, r->accounted_bytes, r->imported_allocation, false);
        }
        const auto external_release = r->external_release;
        void *external_context = r->external_context;
        delete r;
        invoke_context(external_release, external_context);
        mgc_domain_release(d);
    }
    MgcStatus mgc_lease_create(MgcResource *r, const MgcLeaseDesc *input, MgcLease **out)
    {
        if (!r || !out)
            return MGC_INVALID_ARGUMENT;
        *out = nullptr;
        MgcLeaseDesc d{};
        auto rc = read_desc(input, d, offsetof(MgcLeaseDesc, reserved));
        if (rc)
            return rc;
        rc = flags(d.required_flags, d.reserved, sizeof(d.reserved));
        if (rc)
            return rc;
        if (input->header.struct_size > offsetof(MgcLeaseDesc, content_context) &&
            input->header.struct_size < offsetof(MgcLeaseDesc, release_content) + sizeof(d.release_content))
            return MGC_UNSUPPORTED_ABI;
        if (bool(d.retain_content) != bool(d.release_content)) return MGC_INVALID_ARGUMENT;
        if (!d.content_id || !d.byte_length || d.byte_offset > r->desc.capacity_bytes ||
            d.byte_length > r->desc.capacity_bytes - d.byte_offset ||
            (d.access != MGC_ACCESS_READ && d.access != MGC_ACCESS_WRITE))
            return MGC_INVALID_ARGUMENT;
        if (r->external_data && r->external_access == MGC_ACCESS_READ && d.access == MGC_ACCESS_WRITE)
            return MGC_UNSUPPORTED_CAPABILITY;
        if (d.dependency_kind != MGC_DEPENDENCY_READY || d.dependency_id)
            return MGC_UNSUPPORTED_CAPABILITY;
#if MGC_HAVE_NCNN && NCNN_VULKAN
        if(r->desc.usage&&domain_is_lost(r->domain))return MGC_DEVICE_LOST;
#endif
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
        reap_external_ready(r);
#endif
        try
        {
            auto lease = std::make_unique<MgcLease>();
            {
            std::unique_lock<std::mutex> lock(r->mutex, std::try_to_lock);
            if (!lock.owns_lock()) return MGC_BUSY;
            if(d.access==MGC_ACCESS_READ&&r->content_invalid)return r->content_failure;
            if (r->writers || (d.access == MGC_ACCESS_WRITE && r->readers))
                return MGC_BUSY;
            if (d.access == MGC_ACCESS_WRITE && (r->maps || r->in_flight)) return MGC_BUSY;
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
            if (d.access == MGC_ACCESS_WRITE && (r->ownership_uncertain || r->external_transition || r->external_access_state)) return MGC_BUSY;
#endif
            if (d.access == MGC_ACCESS_READ && r->current_content &&
                d.content_id != r->current_content)
                return MGC_ARTIFACT_MISMATCH;
            if (d.access == MGC_ACCESS_WRITE)
            {
                if (d.content_id <= r->current_content)
                    return MGC_ARTIFACT_MISMATCH;
                ++r->writers;
                r->current_content = d.content_id;
                r->content_invalid = false;
                r->content_failure = MGC_DEVICE_LOST;
#if MGC_HAVE_PRODUCER_WAKE
                set_guarded_content_locked(r, false);
#endif
            }
            else
            {
                ++r->readers;
                if (!r->current_content)
                    r->current_content = d.content_id;
            }
            lease->resource = r;
            lease->desc = d;
            lease->content_context = d.content_context;
            lease->release_content = d.release_content;
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
            if (r->ahb) lease->sync_mode = r->ahb_sync_mode;
#endif
            mgc_resource_retain(r);
            }
            invoke_context(d.retain_content, d.content_context);
            *out = lease.release();
            return MGC_SUCCESS;
        }
        catch (...)
        {
            return MGC_OUT_OF_MEMORY;
        }
    }
    void mgc_lease_retain(MgcLease *l)
    {
        if (l)
            ++l->refs;
    }
    void mgc_lease_release(MgcLease *l)
    {
        if (!l) return;
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
        reap_external_ready(l->resource);
#endif
        if (--l->refs)
            return;
        auto *r = l->resource;
        {
            std::lock_guard<std::mutex> lock(r->mutex);
            if (l->desc.access == MGC_ACCESS_WRITE)
                --r->writers;
            else
                --r->readers;
        }
        auto release_content = l->release_content;
        void *content_context = l->content_context;
#if defined(__ANDROID__)
        if (l->acquire_fd >= 0) close(l->acquire_fd);
#endif
        delete l;
        mgc_resource_release(r);
        invoke_context(release_content, content_context);
    }
    MgcStatus mgc_lease_info(MgcLease *l, MgcLeaseInfo *out)
    {
        if (!l)
            return MGC_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> lock(l->resource->mutex);
        if(l->resource->content_invalid)return l->resource->content_failure;
        MgcLeaseInfo v{};
        v.header = MGC_HEADER_INIT(MgcLeaseInfo);
        v.domain_id = l->resource->domain->id;
        v.domain_generation = l->resource->domain->generation;
        v.allocation_id = l->resource->desc.allocation_id;
        v.allocation_generation = l->resource->desc.allocation_generation;
        v.content_id = l->desc.content_id;
        v.capacity_bytes = l->resource->desc.capacity_bytes;
        v.access = l->desc.access;
        v.dependency_kind = l->desc.dependency_kind;
#if MGC_HAVE_NCNN && NCNN_VULKAN && defined(__ANDROID__)
        if (l->resource->external_access_state)
            v.dependency_kind = l->resource->external_access_state != 1 && l->sync_mode ? MGC_DEPENDENCY_SYNC_FD : MGC_DEPENDENCY_HOST_WAIT;
#endif
        v.format = l->resource->desc.format;
        v.width = l->resource->desc.width;
        v.height = l->resource->desc.height;
        v.row_stride_bytes = l->resource->desc.row_stride_bytes;
        v.byte_offset = l->desc.byte_offset;
        v.byte_length = l->desc.byte_length;
        return output_struct(out, &v, sizeof(v), offsetof(MgcLeaseInfo, reserved));
    }
    MgcStatus mgc_lease_resource(MgcLease *lease, MgcResource **out)
    {
        if (!lease || !out) return MGC_INVALID_ARGUMENT;
        mgc_resource_retain(lease->resource); *out = lease->resource;
        return MGC_SUCCESS;
    }
    MgcStatus mgc_lease_map(MgcLease *l, void **data, uint64_t *bytes)
    {
        if (!l || !data || !bytes)
            return MGC_INVALID_ARGUMENT;
        *data = nullptr;
        *bytes = 0;
        if (l->resource->desc.usage) return MGC_UNSUPPORTED_CAPABILITY;
        std::lock_guard<std::mutex> lock(l->resource->mutex);
        if (l->resource->content_invalid) return l->resource->content_failure;
        bool own = active_ticket &&
                   std::find(active_ticket->leases.begin(), active_ticket->leases.end(), l) !=
                       active_ticket->leases.end();
        if ((l->resource->in_flight && !own) || l->maps)
            return MGC_BUSY;
        ++l->maps;
        ++l->resource->maps;
        mgc_lease_retain(l);
        *data = l->resource->host_data() + l->desc.byte_offset;
        *bytes = l->desc.byte_length;
        return MGC_SUCCESS;
    }
    MgcStatus mgc_lease_unmap(MgcLease *l)
    {
        if (!l)
            return MGC_INVALID_ARGUMENT;
        {
            std::lock_guard<std::mutex> lock(l->resource->mutex);
            if (!l->maps)
                return MGC_INVALID_ARGUMENT;
            --l->maps;
            --l->resource->maps;
        }
        mgc_lease_release(l);
        return MGC_SUCCESS;
    }
    MgcStatus mgc_session_create(MgcDomain *d, const MgcSessionDesc *input, MgcSession **out)
    {
        if (!d || !out)
            return MGC_INVALID_ARGUMENT;
        *out = nullptr;
        MgcSessionDesc v{};
        auto rc = read_desc(input, v, offsetof(MgcSessionDesc, reserved));
        if (rc)
            return rc;
        rc = flags(v.required_flags, v.reserved, sizeof(v.reserved));
        if (rc)
            return rc;
        if (!v.max_in_flight || v.max_in_flight > 1024)
            return MGC_INVALID_ARGUMENT;
#if MGC_HAVE_NCNN && NCNN_VULKAN
        if(domain_is_lost(d))return MGC_DEVICE_LOST;
#endif
        {
            std::lock_guard<std::mutex> lock(d->runtime->mutex);
            if (d->runtime->closed)
                return MGC_BUSY;
        }
        try
        {
            auto s = std::make_unique<MgcSession>();
            s->domain = d;
            s->max_in_flight = v.max_in_flight;
#if MGC_HAVE_NCNN && NCNN_VULKAN
            if (d->device) {
                s->code_image.acquire(&code_image_identity, "meguviz_code_image_anchor_v1");
                if (!s->code_image.available()) return MGC_UNSUPPORTED_CAPABILITY;
            }
#endif
            mgc_domain_retain(d);
            try
            {
                s->worker = std::thread(worker, s.get());
            }
            catch (...)
            {
                mgc_domain_release(d);
                throw;
            }
            *out = s.release();
            return MGC_SUCCESS;
        }
        catch (...)
        {
            return MGC_OUT_OF_MEMORY;
        }
    }
    MgcStatus mgc_session_close(MgcSession *s)
    {
        if (!s)
            return MGC_INVALID_ARGUMENT;
        if (worker_session == s)
            return MGC_BUSY;
        {
            std::lock_guard<std::mutex> lock(s->mutex);
            s->closing = true;
            s->work.notify_all();
#if MGC_HAVE_PRODUCER_WAKE
            s->identity->wake.notify();
#endif
            if (s->in_flight || s->preparing)
                return MGC_BUSY;
        }
        if (s->worker.joinable())
            s->worker.join();
#if MGC_HAVE_NCNN && NCNN_VULKAN
        // The caller retains its own loader reference through the close call.
        // On release failure retain the closed session for an explicit retry.
        if (!s->code_image.reset()) return MGC_INTERNAL_ERROR;
#endif
        auto *d = s->domain;
        delete s;
        mgc_domain_release(d);
        return MGC_SUCCESS;
    }
    MgcStatus mgc_submit(MgcSession *s, const MgcSubmitDesc *input, MgcTicket **out)
    {
        if (!s || !out)
            return MGC_INVALID_ARGUMENT;
        *out = nullptr;
        MgcSubmitDesc d{};
        auto rc = read_desc(input, d, offsetof(MgcSubmitDesc, reserved));
        if (rc)
            return rc;
        if (input->header.struct_size > offsetof(MgcSubmitDesc, batch_event) &&
            input->header.struct_size < offsetof(MgcSubmitDesc, batch_event) + sizeof(d.batch_event)) return MGC_UNSUPPORTED_ABI;
        rc = flags(d.required_flags, d.reserved, sizeof(d.reserved));
        if (rc)
            return rc;
        if (bool(d.retain_context) != bool(d.release_context))
            return MGC_INVALID_ARGUMENT;
        rc = operation_valid(d);
        if (rc)
            return rc;
        SubmissionPreparation preparation;
        bool retained = false;
        try
        {
            rc = preparation.acquire(s);
            if (rc != MGC_SUCCESS) return rc;
            auto t = std::make_unique<MgcTicket>();
            t->desc = d;
#if MGC_HAVE_NCNN && NCNN_VULKAN
            if(s->domain->device){t->host_flags=MGC_HOST_JOB_EXPLICIT_GPU_SCOPES;t->resume=[](void*,MgcTicket* ticket){return run_operation(ticket);};}
#endif
            if (d.batch_event) t->events.reserve(128);
            t->result.header = MGC_HEADER_INIT(MgcCompletion);
            t->result.request_id = d.request_id;
            t->leases = {d.input, d.output};
            if (d.retain_context)
            {
                d.retain_context(d.context);
                retained = true;
            }
            rc = accept(s, t, out);
            if (rc != MGC_ACCEPTED && retained)
                invoke_context(d.release_context, d.context);
            return rc;
        }
        catch (...)
        {
            if (retained)
                invoke_context(d.release_context, d.context);
            return MGC_OUT_OF_MEMORY;
        }
    }
    void mgc_ticket_retain(MgcTicket *t)
    {
        if (t)
            ++t->refs;
    }
    void mgc_ticket_release(MgcTicket *t)
    {
        if (t && --t->refs == 0)
            delete t;
    }
    MgcStatus mgc_ticket_cancel(MgcTicket *t)
    {
        if (!t)
            return MGC_INVALID_ARGUMENT;
        {
            std::lock_guard<std::mutex> lock(t->mutex);
            if (!t->done)
            {
#if MGC_HAVE_PRODUCER_WAKE
                // Guarded private output transfer is an irrevocable publish
                // point: cancellation cannot promise to retract that result.
                if (t->guarded_submission.load() && t->guarded_write_published) return MGC_BUSY;
#endif
                t->cancelled.store(true);
                t->result.result_discarded = 1;
            }
        }
#if MGC_HAVE_PRODUCER_WAKE
        if (t->session_identity) t->session_identity->wake.notify();
#endif
        return MGC_SUCCESS;
    }
    MgcStatus mgc_private_session_cancel(MgcSession *s, MgcTicket *t)
    {
        if (!s || !t || s->identity != t->session_identity)
            return MGC_INVALID_ARGUMENT;
        return mgc_ticket_cancel(t);
    }
    MgcStatus mgc_ticket_query(MgcTicket *t, MgcCompletion *out)
    {
        if (!t)
            return MGC_INVALID_ARGUMENT;
        std::lock_guard<std::mutex> lock(t->mutex);
        auto rc =
            output_struct(out, &t->result, sizeof(t->result), offsetof(MgcCompletion, reserved));
        return rc ? rc : t->done ? MGC_SUCCESS : MGC_BUSY;
    }
    MgcStatus mgc_ticket_execution_info(MgcTicket *t, MgcExecutionInfo *out)
    {
        if (!t) return MGC_INVALID_ARGUMENT;
        MgcExecutionInfo v{}; v.header = MGC_HEADER_INIT(MgcExecutionInfo);
        v.submitted_batches = t->submitted_batches.load(); v.retired_batches = t->retired_batches.load();
        v.finalized_batches = t->finalized_batches.load(); v.recording_submitted_batches = t->recording_batches.load();
        v.final_submit_async = t->used_final_async.load() ? 1 : 0;
        return output_struct(out, &v, sizeof(v), offsetof(MgcExecutionInfo, reserved));
    }
    MgcStatus mgc_session_drain(MgcSession *s, uint64_t ns)
    {
        if (!s)
            return MGC_INVALID_ARGUMENT;
        if (worker_session == s)
            return MGC_BUSY;
        std::unique_lock<std::mutex> lock(s->mutex);
        if (!s->in_flight)
            return MGC_SUCCESS;
        if (!ns)
            return MGC_TIMEOUT;
        auto done = [&] { return !s->in_flight; };
        if (ns == UINT64_MAX) {
            s->retired.wait(lock, done);
            return MGC_SUCCESS;
        }
        // libc++'s predicate wait_for adds the duration to steady_clock::now().
        // Large finite durations can overflow that addition; bounded slices
        // preserve their finite deadline without overflowing the clock.
        constexpr uint64_t slice_ns = 3600ULL * 1000000000ULL;
        uint64_t remaining = ns;
        for (;;) {
            const auto start = std::chrono::steady_clock::now();
            const auto slice = std::chrono::nanoseconds(static_cast<int64_t>(std::min(remaining, slice_ns)));
            if (s->retired.wait_for(lock, slice, done)) return MGC_SUCCESS;
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed > 0 && static_cast<uint64_t>(elapsed) >= remaining) return MGC_TIMEOUT;
            if (elapsed > 0) remaining -= static_cast<uint64_t>(elapsed);
        }
    }
    MgcStatus mgc_private_submit_host_job(MgcSession *s, const MgcHostJobDesc *input,
                                          MgcTicket **out)
    {
        if (!s || !out)
            return MGC_INVALID_ARGUMENT;
        *out = nullptr;
        MgcHostJobDesc d{};
        auto rc = read_desc(input, d, offsetof(MgcHostJobDesc, batch_event));
        if (rc)
            return rc;
        for (auto field : {std::pair<size_t,size_t>{offsetof(MgcHostJobDesc,batch_event),sizeof(d.batch_event)},
                           {offsetof(MgcHostJobDesc,resume),sizeof(d.resume)},
                           {offsetof(MgcHostJobDesc,finalize),sizeof(d.finalize)}})
            if(input->header.struct_size>field.first&&input->header.struct_size<field.first+field.second)return MGC_UNSUPPORTED_ABI;
        if (d.required_flags & ~MGC_HOST_JOB_EXPLICIT_GPU_SCOPES)
            return MGC_UNSUPPORTED_ABI;
        if (!d.execute || d.lease_count > 64 || (d.lease_count && !d.leases) ||
            bool(d.retain_context) != bool(d.release_context))
            return MGC_INVALID_ARGUMENT;
        SubmissionPreparation preparation;
        bool retained = false;
        try
        {
            rc = preparation.acquire(s);
            if (rc != MGC_SUCCESS) return rc;
            auto t = std::make_unique<MgcTicket>();
            t->desc.header = MGC_HEADER_INIT(MgcSubmitDesc);
            t->desc.request_id = d.request_id;
            t->desc.completion = d.completion;
            t->desc.context = d.context;
            t->desc.retain_context = d.retain_context;
            t->desc.release_context = d.release_context;
            t->desc.batch_event = d.batch_event;
            if (d.batch_event) t->events.reserve(128);
            t->execute = d.execute;
            t->host_flags = d.required_flags;
            t->resume = d.resume;
            t->finalize = d.finalize;
            t->result.header = MGC_HEADER_INIT(MgcCompletion);
            t->result.request_id = d.request_id;
            if (d.lease_count)
                t->leases.assign(d.leases, d.leases + d.lease_count);
            if (d.retain_context)
            {
                d.retain_context(d.context);
                retained = true;
            }
            rc = accept(s, t, out);
            if (rc != MGC_ACCEPTED && retained)
                invoke_context(d.release_context, d.context);
            return rc;
        }
        catch (...)
        {
            if (retained)
                invoke_context(d.release_context, d.context);
            return MGC_OUT_OF_MEMORY;
        }
    }
    uint32_t mgc_private_ticket_cancelled(MgcTicket *t)
    {
        return t && t->cancelled.load() ? 1u : 0u;
    }
    MgcStatus mgc_private_ticket_defer(MgcTicket *ticket, MgcHostRetireFn poll,
        void *context, uint32_t resume_allowed, MgcStatus failure_status)
    {
        if (!ticket || ticket != active_ticket || !poll || ticket->deferred || resume_allowed > 1 ||
            (ticket->gpu_batch && context != ticket->gpu_batch.get()))
            return MGC_INVALID_ARGUMENT;
        ticket->retire = poll; ticket->retire_context = context;
        ticket->resume_allowed = resume_allowed != 0;
        ticket->deferred_failure = failure_status == MGC_SUCCESS ? MGC_INTERNAL_ERROR : failure_status;
        ticket->deferred = true;
        return MGC_SUCCESS;
    }
    uint32_t mgc_private_ticket_is_deferred(MgcTicket *ticket)
    { return ticket && ticket == active_ticket && ticket->deferred ? 1u : 0u; }
    MgcStatus mgc_private_ticket_release_retired_write(MgcTicket *ticket, MgcLease *lease)
    {
        if(!ticket||ticket!=active_ticket||!lease||lease->resource->domain!=ticket->domain)return MGC_INVALID_ARGUMENT;
        if(ticket->deferred||ticket->gpu_batch)return MGC_BUSY;
        if(std::count(ticket->leases.begin(),ticket->leases.end(),lease)!=1)return MGC_INVALID_ARGUMENT;
        auto found=std::find(ticket->leases.begin(),ticket->leases.end(),lease);
#if MGC_HAVE_PRODUCER_WAKE
        std::unique_lock<std::mutex> ticket_lock(ticket->mutex);
        const bool guarded = ticket->guarded_submission.load();
        if (guarded && ticket->cancelled.load()) return MGC_CANCELLED;
#endif
        {
            std::lock_guard<std::mutex> lock(lease->resource->mutex);
            if (lease->desc.access != MGC_ACCESS_WRITE) return MGC_INVALID_ARGUMENT;
            if(lease->resource->in_flight!=1||lease->resource->maps)return MGC_BUSY;
            --lease->resource->in_flight;
#if MGC_HAVE_PRODUCER_WAKE
            unregister_producer_waiter_locked(lease->resource, producer_wake_alias(ticket->session_identity));
#endif
        }
        ticket->leases.erase(found);
#if MGC_HAVE_PRODUCER_WAKE
        ticket->guarded_write_published = true;
        ticket_lock.unlock();
#endif
        mgc_lease_release(lease);
        return MGC_SUCCESS;
    }
    MgcStatus mgc_private_session_domain(MgcSession *s, MgcDomain **out)
    {
        if (!s || !out)
            return MGC_INVALID_ARGUMENT;
        mgc_domain_retain(s->domain);
        *out = s->domain;
        return MGC_SUCCESS;
    }
#include "android_api.inc"
}
#if MGC_HAVE_NCNN
namespace meguviz
{
MgcStatus with_ncnn_domain(MgcDomain *d, NcnnDomainFn fn, void *ctx)
{
    if (!d || !fn)
        return MGC_INVALID_ARGUMENT;
    if (d->runtime->provider == MGC_PROVIDER_CPU_REFERENCE)
        return MGC_UNSUPPORTED_CAPABILITY;
    DomainScope scope(d);
    if (scope.status) return scope.status;
    MgcStatus rc;
    try
    {
        rc = fn(ctx,
#if NCNN_VULKAN
                d->device
#else
                nullptr
#endif
        );
    }
    catch (const std::bad_alloc &)
    {
        rc = MGC_OUT_OF_MEMORY;
    }
    catch (...)
    {
        rc = MGC_INTERNAL_ERROR;
    }
    return rc;
}
MgcStatus with_ncnn_domain_cleanup(MgcDomain *d, NcnnDomainFn fn, void *ctx)
{
    if (!d || !fn) return MGC_INVALID_ARGUMENT;
    // Cleanup owns exclusive object references and never submits. CPU objects
    // have no shared Vulkan allocator, so cleanup takes no CPU execution mutex.
    mgc_domain_retain(d);
    MgcStatus status;
    {
        std::unique_lock<std::recursive_mutex> lock(execution_slot, std::defer_lock);
#if NCNN_VULKAN
        if (d->device) lock.lock();
#endif
        auto *previous = active_domain;
        auto *previous_slot = active_execution_slot;
        active_domain = d;
#if NCNN_VULKAN
        if (d->device) active_execution_slot = &execution_slot;
#endif
        try { status = fn(ctx,
#if NCNN_VULKAN
            d->device
#else
            nullptr
#endif
        ); }
        catch (const std::bad_alloc &) { status = MGC_OUT_OF_MEMORY; }
        catch (...) { status = MGC_INTERNAL_ERROR; }
        active_domain = previous;
        active_execution_slot = previous_slot;
    }
    mgc_domain_release(d);
    return status;
}
const char *private_sdk_fingerprint()
{
    return MGC_PRIVATE_FP;
}
bool ncnn_domain_is_lost(MgcDomain* d){
#if NCNN_VULKAN
    return domain_is_lost(d);
#else
    (void)d;return false;
#endif
}
bool ncnn_domain_loss_retired(MgcDomain* d){
#if NCNN_VULKAN
    return device_loss_retired(d);
#else
    (void)d;return false;
#endif
}
void ncnn_report_device_lost(MgcDomain* d){
#if NCNN_VULKAN
    report_device_lost(d);
#else
    (void)d;
#endif
}
#if NCNN_VULKAN
static MgcStatus create_ncnn_domain_owner(MgcDomain *domain, NcnnDomainOwner **out, MgcTicket *ticket)
{
    if (!out) return MGC_INVALID_ARGUMENT;
    *out = nullptr;
    if (!domain) return MGC_INVALID_ARGUMENT;
    if (!domain->device) return MGC_UNSUPPORTED_CAPABILITY;
    if (active_ticket && active_ticket != ticket) return MGC_BUSY;
    try {
        auto owner = std::make_unique<NcnnDomainOwner>();
        const bool session_pin = ticket && active_ticket == ticket && worker_session &&
            worker_session->domain == domain && worker_session->identity == ticket->session_identity &&
            worker_session->code_image.available();
        if (!session_pin) {
            owner->code_image.acquire(&code_image_identity, "meguviz_code_image_anchor_v1");
            if (!owner->code_image.available()) return MGC_UNSUPPORTED_CAPABILITY;
        }
        DomainScope scope(domain);
        if (scope.status) return scope.status;
        if (domain->admission->owners >= 16) return MGC_BACKPRESSURE;
        owner->domain = domain;
        owner->ticket = ticket;
        mgc_domain_retain(domain);
        ++domain->admission->owners;
        *out = owner.release();
        return MGC_SUCCESS;
    } catch (const std::bad_alloc &) { return MGC_OUT_OF_MEMORY; }
    catch (...) { return MGC_INTERNAL_ERROR; }
}
MgcStatus ncnn_domain_owner_create(MgcDomain *domain, NcnnDomainOwner **out)
{ return create_ncnn_domain_owner(domain,out,nullptr); }
MgcStatus ncnn_domain_owner_pending(NcnnDomainOwner *owner, bool pending)
{
    if (!owner || active_owner != owner || active_execution_slot != &execution_slot ||
        (active_ticket && owner->ticket != active_ticket))
        return MGC_INVALID_ARGUMENT;
    auto &gate = *owner->domain->admission;
    std::lock_guard<std::mutex> guard(gate.mutex);
    if(gate.lost.load()){
        owner->pending=pending;
        if(pending&&!gate.pending)gate.pending=owner;
        else if(!pending&&gate.pending==owner)gate.pending=nullptr;
        return MGC_SUCCESS;
    }
    if (gate.pending && gate.pending != owner) return MGC_BUSY;
    owner->pending = pending;
    gate.pending = pending ? owner : nullptr;
    return MGC_SUCCESS;
}
static MgcStatus call_ncnn_domain_owner(NcnnDomainOwner *owner, NcnnDomainFn fn, void *ctx, bool try_only)
{
    if (!owner || !fn) return MGC_INVALID_ARGUMENT;
    if ((active_ticket && owner->ticket != active_ticket) || active_owner) return MGC_BUSY;
    // Publish the identity before gate checking; only this owner may re-enter
    // its quarantined device. The callback is conservatively pending on entry.
    active_owner = owner;
    struct ResetOwner { ~ResetOwner(){active_owner=nullptr;} } reset_owner;
    MgcStatus status;
    {
        DomainScope scope(owner->domain, try_only);
        status = scope.status;
        if (!status) {
            ncnn_domain_owner_pending(owner, true);
            try { status = fn(ctx, owner->domain->device); }
            catch (const std::bad_alloc &) { ncnn_domain_owner_pending(owner, true); status = MGC_OUT_OF_MEMORY; }
            catch (...) { ncnn_domain_owner_pending(owner, true); status = MGC_INTERNAL_ERROR; }
        }
    }
    return status;
}
MgcStatus with_ncnn_domain_owner(NcnnDomainOwner *owner, NcnnDomainFn fn, void *ctx)
{ return call_ncnn_domain_owner(owner, fn, ctx, false); }
MgcStatus try_with_ncnn_domain_owner(NcnnDomainOwner *owner, NcnnDomainFn fn, void *ctx)
{ return call_ncnn_domain_owner(owner, fn, ctx, true); }
MgcStatus try_with_ncnn_domain_owner_retire(NcnnDomainOwner *owner, NcnnDomainFn fn, void *ctx)
{
    if(!owner)return MGC_INVALID_ARGUMENT;
    owner->retirement_only=true;
    struct Reset {NcnnDomainOwner* owner;~Reset(){owner->retirement_only=false;}} reset{owner};
    return call_ncnn_domain_owner(owner,fn,ctx,true);
}
MgcStatus ncnn_domain_owner_release(NcnnDomainOwner *owner)
{
    if (!owner) return MGC_INVALID_ARGUMENT;
    if (active_owner || (active_ticket && owner->ticket != active_ticket) ||
        (active_execution_slot && active_execution_slot != &execution_slot)) return MGC_BUSY;
    MgcDomain *domain = owner->domain;
    {
        std::unique_lock<std::recursive_mutex> lock(execution_slot, std::try_to_lock);
        if (!lock.owns_lock()) return MGC_BUSY;
        if (owner->pending) return MGC_BUSY;
        if (!owner->code_image.reset()) return MGC_INTERNAL_ERROR;
        --domain->admission->owners;
        delete owner;
        // Keep the recursive GPU slot through the last domain release: its
        // origin cleanup must not reacquire behind a newly started NN job.
        mgc_domain_release(domain);
    }
    return MGC_SUCCESS;
}
#include "owned_gpu_scope.inc"
MgcStatus with_ncnn_gpu(MgcDomain *domain, MgcLease *const *leases, uint32_t count,
                        NcnnGpuFn fn, void *context)
{
    if(!domain||!fn||count>64||(count&&!leases))return MGC_INVALID_ARGUMENT;
    MgcTicket local;auto* previous=active_ticket;
    auto* ticket=previous?previous:&local;
    if(!previous){local.domain=domain;local.measurement_flags=domain->measurement_flags.load();active_ticket=&local;}
    struct Reset {MgcTicket* previous;~Reset(){active_ticket=previous;}} reset{previous};
    // This legacy API borrows its callback context. It must remain on the
    // caller stack until retirement rather than returning an asynchronous loan.
    std::shared_ptr<void> anchor;
    try{anchor=std::make_shared<int>(0);}catch(...){return MGC_OUT_OF_MEMORY;}
    auto status=with_ncnn_gpu_owned(domain,leases,count,std::move(anchor),fn,context);
    if(status!=MGC_BUSY||!ncnn_gpu_scope_deferred()||!ticket->deferred||!ticket->gpu_batch)return status;
    auto state=std::static_pointer_cast<OwnedGpuBatch>(ticket->gpu_batch);
    for(;;){
        MgcStatus retired=MGC_INTERNAL_ERROR;
        try{retired=ticket->retire(ticket->retire_context,ticket);}catch(...){}
        if(retired==MGC_SUCCESS)break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ticket->deferred=false;ticket->retire=nullptr;ticket->retire_context=nullptr;
    return state->failed?state->failure:MGC_SUCCESS;
}
#endif
} // namespace meguviz
#endif
