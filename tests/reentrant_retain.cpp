#include <meguviz/compute.h>
#include <meguviz/compute_job.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>

// A successful retain arms the next allocation on the submitter thread.
// The failure is therefore in submission acceptance, not context retention.
static thread_local bool fail_next_allocation = false;

void *operator new(std::size_t size)
{
    if (fail_next_allocation)
    {
        fail_next_allocation = false;
        throw std::bad_alloc();
    }
    if (auto *memory = std::malloc(size ? size : 1))
        return memory;
    throw std::bad_alloc();
}

void operator delete(void *memory) noexcept
{
    std::free(memory);
}

void operator delete(void *memory, std::size_t) noexcept
{
    std::free(memory);
}

template<class T>T init(){T x{};x.header=MGC_HEADER_INIT(T);return x;}
struct Fixture {MgcRuntime *runtime=nullptr;MgcDomain *domain=nullptr;MgcSession *session=nullptr;
 bool create(){auto rd=init<MgcRuntimeDesc>();rd.provider=MGC_PROVIDER_CPU_REFERENCE;rd.memory_budget_bytes=4096;auto dd=init<MgcDomainDesc>();auto sd=init<MgcSessionDesc>();sd.max_in_flight=1;return !mgc_runtime_create(&rd,&runtime)&&!mgc_domain_acquire(runtime,&dd,&domain)&&!mgc_session_create(domain,&sd,&session);}
 bool finish(){auto s=mgc_session_close(session);mgc_domain_release(domain);auto r=mgc_runtime_close(runtime);mgc_runtime_release(runtime);return !s&&!r;}};
struct Context {MgcSession *session=nullptr;std::mutex mutex;std::condition_variable cv;bool entered=false,proceed=false;std::atomic<int> retained{0},released{0},completed{0};MgcStatus nested=MGC_INTERNAL_ERROR,drain=MGC_INTERNAL_ERROR,close=MGC_INTERNAL_ERROR;MgcTicket *inner=nullptr;};
MgcStatus immediate(void*,MgcTicket*){return MGC_SUCCESS;}
MgcStatus wait_gate(void *p,MgcTicket*){auto &c=*static_cast<Context*>(p);std::unique_lock<std::mutex> l(c.mutex);c.cv.wait(l,[&]{return c.proceed;});return MGC_SUCCESS;}
void release(void *p){++static_cast<Context*>(p)->released;}
void complete(void *p,const MgcCompletion*){++static_cast<Context*>(p)->completed;}
void retain_close(void *p){auto &c=*static_cast<Context*>(p);++c.retained;c.drain=mgc_session_drain(c.session,0);c.close=mgc_session_close(c.session);}
void retain_wait(void *p){auto &c=*static_cast<Context*>(p);++c.retained;std::unique_lock<std::mutex> l(c.mutex);c.entered=true;c.cv.notify_all();c.cv.wait(l,[&]{return c.proceed;});}
void retain_throw(void*){throw std::runtime_error("before_retain");}
void retain_nested(void *p){auto &c=*static_cast<Context*>(p);++c.retained;c.drain=mgc_session_drain(c.session,0);auto inner=init<MgcHostJobDesc>();inner.execute=wait_gate;inner.context=&c;c.nested=mgc_private_submit_host_job(c.session,&inner,&c.inner);}
bool case_close(){Fixture f;if(!f.create())return false;Context c;c.session=f.session;auto job=init<MgcHostJobDesc>();job.execute=immediate;job.context=&c;job.retain_context=retain_close;job.release_context=release;job.completion=complete;MgcTicket *t=nullptr;auto rc=mgc_private_submit_host_job(f.session,&job,&t);bool ok=rc==MGC_BUSY&&!t&&c.drain==MGC_SUCCESS&&c.close==MGC_BUSY&&c.retained==1&&c.released==1&&c.completed==0;return f.finish()&&ok;}
bool case_nested(){Fixture f;if(!f.create())return false;Context c;c.session=f.session;auto job=init<MgcHostJobDesc>();job.execute=immediate;job.context=&c;job.retain_context=retain_nested;job.release_context=release;job.completion=complete;MgcTicket *t=nullptr;auto rc=mgc_private_submit_host_job(f.session,&job,&t);bool ok=rc==MGC_BACKPRESSURE&&!t&&c.drain==MGC_SUCCESS&&c.nested==MGC_ACCEPTED&&c.inner&&c.retained==1&&c.released==1&&c.completed==0;{std::lock_guard<std::mutex> l(c.mutex);c.proceed=true;}c.cv.notify_all();ok=mgc_session_drain(f.session,5000000000ULL)==MGC_SUCCESS&&ok;mgc_ticket_release(c.inner);return f.finish()&&ok;}
bool case_concurrent(){Fixture f;if(!f.create())return false;Context c;c.session=f.session;auto job=init<MgcHostJobDesc>();job.execute=immediate;job.context=&c;job.retain_context=retain_wait;job.release_context=release;job.completion=complete;MgcTicket *t=nullptr;MgcStatus rc=MGC_INTERNAL_ERROR;std::thread caller([&]{rc=mgc_private_submit_host_job(f.session,&job,&t);});
 {std::unique_lock<std::mutex> l(c.mutex);if(!c.cv.wait_for(l,std::chrono::seconds(2),[&]{return c.entered;})){c.proceed=true;l.unlock();c.cv.notify_all();caller.join();return false;}}
 auto closed=mgc_session_close(f.session);{std::lock_guard<std::mutex> l(c.mutex);c.proceed=true;}c.cv.notify_all();caller.join();bool ok=closed==MGC_BUSY&&rc==MGC_BUSY&&!t&&c.retained==1&&c.released==1&&c.completed==0;return f.finish()&&ok;}
bool case_exception(){Fixture f;if(!f.create())return false;Context c;c.session=f.session;auto job=init<MgcHostJobDesc>();job.execute=immediate;job.context=&c;job.retain_context=retain_throw;job.release_context=release;MgcTicket *t=nullptr;auto rc=mgc_private_submit_host_job(f.session,&job,&t);bool ok=rc==MGC_OUT_OF_MEMORY&&!t&&c.released==0;return f.finish()&&ok;}
bool case_public(){Fixture f;if(!f.create())return false;Context c;c.session=f.session;MgcResource *resources[2]{};MgcLease *leases[2]{};for(int i=0;i<2;i++){auto r=init<MgcResourceDesc>();r.width=2;r.height=2;r.format=MGC_FORMAT_RGB8;r.row_stride_bytes=6;r.capacity_bytes=12;if(mgc_resource_create(f.domain,&r,&resources[i]))return false;auto l=init<MgcLeaseDesc>();l.content_id=i+1;l.access=i?MGC_ACCESS_WRITE:MGC_ACCESS_READ;l.byte_length=12;if(mgc_lease_create(resources[i],&l,&leases[i]))return false;}
 auto q=init<MgcSubmitDesc>();q.operation=MGC_OPERATION_COPY;q.input=leases[0];q.output=leases[1];q.byte_count=12;q.context=&c;q.retain_context=retain_close;q.release_context=release;q.completion=complete;MgcTicket *t=nullptr;auto rc=mgc_submit(f.session,&q,&t);MgcLeaseInfo a=init<MgcLeaseInfo>(),b=init<MgcLeaseInfo>();bool ok=rc==MGC_BUSY&&!t&&c.close==MGC_BUSY&&c.retained==1&&c.released==1&&c.completed==0&&!mgc_lease_info(leases[0],&a)&&!mgc_lease_info(leases[1],&b);for(int i=0;i<2;i++){mgc_lease_release(leases[i]);mgc_resource_release(resources[i]);}return f.finish()&&ok;}

struct AcceptanceFailure
{
    MgcSession *session = nullptr;
    std::mutex mutex;
    std::condition_variable state_changed;
    bool release_entered = false;
    bool close_finished = false;
    bool allocation_failed = false;
    int retains = 0;
    int releases = 0;
    int completions = 0;
    MgcStatus close_status = MGC_INTERNAL_ERROR;
    MgcStatus drain_status = MGC_INTERNAL_ERROR;
    MgcStatus reentrant_close = MGC_INTERNAL_ERROR;

    static void retain(void *context)
    {
        ++static_cast<AcceptanceFailure *>(context)->retains;
        fail_next_allocation = true;
    }

    static void release(void *context)
    {
        auto &state = *static_cast<AcceptanceFailure *>(context);
        state.allocation_failed = !fail_next_allocation;
        fail_next_allocation = false;
        ++state.releases;
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.release_entered = true;
            state.state_changed.notify_one();
            state.state_changed.wait(lock, [&] { return state.close_finished; });
        }
        if (state.close_status == MGC_BUSY)
        {
            state.drain_status = mgc_session_drain(state.session, 0);
            state.reentrant_close = mgc_session_close(state.session);
        }
    }

    static void complete(void *context, const MgcCompletion *)
    {
        ++static_cast<AcceptanceFailure *>(context)->completions;
    }
};

bool case_acceptance_failure(bool use_public_submit)
{
    Fixture fixture;
    if (!fixture.create())
        return false;

    MgcResource *resources[2]{};
    MgcLease *leases[2]{};
    bool leases_created = true;
    for (int index = 0; index < 2; ++index)
    {
        auto resource = init<MgcResourceDesc>();
        resource.allocation_id = index + 1;
        resource.allocation_generation = 1;
        resource.capacity_bytes = 4;
        resource.format = MGC_FORMAT_BYTES;
        auto lease = init<MgcLeaseDesc>();
        lease.access = index == 0 ? MGC_ACCESS_READ : MGC_ACCESS_WRITE;
        lease.content_id = 1;
        lease.byte_length = 4;
        if (mgc_resource_create(fixture.domain, &resource, &resources[index]) != MGC_SUCCESS ||
            mgc_lease_create(resources[index], &lease, &leases[index]) != MGC_SUCCESS)
        {
            leases_created = false;
            break;
        }
    }

    bool passed = false;
    if (leases_created)
    {
        AcceptanceFailure state;
        state.session = fixture.session;
        auto public_job = init<MgcSubmitDesc>();
        public_job.operation = MGC_OPERATION_COPY;
        public_job.byte_count = 4;
        public_job.input = leases[0];
        public_job.output = leases[1];
        public_job.context = &state;
        public_job.retain_context = AcceptanceFailure::retain;
        public_job.release_context = AcceptanceFailure::release;
        public_job.completion = AcceptanceFailure::complete;
        auto private_job = init<MgcHostJobDesc>();
        private_job.execute = immediate;
        private_job.leases = leases;
        private_job.lease_count = 2;
        private_job.context = &state;
        private_job.retain_context = AcceptanceFailure::retain;
        private_job.release_context = AcceptanceFailure::release;
        private_job.completion = AcceptanceFailure::complete;

        MgcTicket *ticket = nullptr;
        MgcStatus result = MGC_INTERNAL_ERROR;
        std::thread submitter([&] {
            result = use_public_submit ? mgc_submit(state.session, &public_job, &ticket)
                                       : mgc_private_submit_host_job(state.session, &private_job, &ticket);
        });
        bool reached_release;
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            reached_release = state.state_changed.wait_for(
                lock, std::chrono::seconds(5), [&] { return state.release_entered; });
        }
        const auto close_status = reached_release ? mgc_session_close(fixture.session) : MGC_INTERNAL_ERROR;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.close_status = close_status;
            state.close_finished = true;
        }
        state.state_changed.notify_one();
        submitter.join();

        if (result == MGC_ACCEPTED)
            mgc_session_drain(fixture.session, 5000000000ULL);
        passed = reached_release && close_status == MGC_BUSY && state.allocation_failed &&
                 state.drain_status == MGC_SUCCESS && state.reentrant_close == MGC_BUSY &&
                 result == MGC_OUT_OF_MEMORY && !ticket && state.retains == 1 &&
                 state.releases == 1 && state.completions == 0;

        mgc_ticket_release(ticket);
        if (close_status == MGC_SUCCESS || state.reentrant_close == MGC_SUCCESS)
            fixture.session = nullptr;
    }
    if (leases_created && !passed)
        std::fprintf(stderr, "%s submit did not hold its session through exception cleanup\n",
                     use_public_submit ? "Public" : "Private");
    for (int index = 0; index < 2; ++index)
    {
        mgc_lease_release(leases[index]);
        mgc_resource_release(resources[index]);
    }
    if (fixture.session)
        return fixture.finish() && passed;
    mgc_domain_release(fixture.domain);
    mgc_runtime_close(fixture.runtime);
    mgc_runtime_release(fixture.runtime);
    return passed;
}

int main()
{
    const bool close = case_close();
    const bool nested = case_nested();
    const bool concurrent = case_concurrent();
    const bool exception = case_exception();
    const bool public_rejection = case_public();
    const bool public_acceptance_failure = case_acceptance_failure(true);
    const bool private_acceptance_failure = case_acceptance_failure(false);
    std::printf("{\"retain_close\":%s,\"retain_drain_nested_submit\":%s,\"concurrent_close\":%s,"
                "\"throw_before_retain\":%s,\"public_rejected_lease_ownership\":%s,"
                "\"public_acceptance_failure\":%s,\"private_acceptance_failure\":%s}\n",
                close ? "true" : "false", nested ? "true" : "false", concurrent ? "true" : "false",
                exception ? "true" : "false", public_rejection ? "true" : "false",
                public_acceptance_failure ? "true" : "false",
                private_acceptance_failure ? "true" : "false");
    return close && nested && concurrent && exception && public_rejection &&
                   public_acceptance_failure && private_acceptance_failure ? 0 : 1;
}
