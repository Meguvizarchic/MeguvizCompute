#include <meguviz/compute.h>
#include <meguviz/compute_job.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

extern "C" MgcStatus mgc_test_guard_arm(MgcLease *, uint64_t);
extern "C" MgcStatus mgc_test_guard_fence(MgcLease *, uint64_t, MgcStatus);
extern "C" MgcStatus mgc_test_guard_finish(MgcLease *, uint64_t, MgcStatus);
extern "C" uint32_t mgc_test_guard_pin(MgcLease *);
#if defined(MGC_RUNTIME_LEGACY_SNAPSHOT)
/* The legacy benchmark executes only latency; this unused stub permits the
 * same harness translation unit to link before introspection existed. */
extern "C" uint32_t mgc_test_guard_waiters(MgcLease *) { return 0; }
#else
extern "C" uint32_t mgc_test_guard_waiters(MgcLease *);
#endif

static std::atomic<bool> hold_guard_ready{false}, guard_ready_entered{false}, release_guard_ready{false};
extern "C" void mgc_test_after_guard_ready(MgcTicket *) {
    if (!hold_guard_ready.load(std::memory_order_acquire)) return;
    guard_ready_entered.store(true, std::memory_order_release);
    while (!release_guard_ready.load(std::memory_order_acquire)) std::this_thread::yield();
}

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while (0)
template<class T> T init() { T x{}; x.header = MGC_HEADER_INIT(T); return x; }
using Clock = std::chrono::steady_clock;
constexpr uint64_t five_seconds = 5000000000ULL;

struct Fixture {
    MgcRuntime *runtime = nullptr; MgcDomain *domain = nullptr; MgcSession *session = nullptr;
    MgcResource *input = nullptr, *output = nullptr;
    MgcLease *source = nullptr, *destination = nullptr;
    Fixture() {
        auto rd = init<MgcRuntimeDesc>(); rd.provider = MGC_PROVIDER_CPU_REFERENCE; rd.memory_budget_bytes = 4096;
        CHECK(mgc_runtime_create(&rd, &runtime) == MGC_SUCCESS);
        auto dd = init<MgcDomainDesc>(); CHECK(mgc_domain_acquire(runtime, &dd, &domain) == MGC_SUCCESS);
        auto sd = init<MgcSessionDesc>(); sd.max_in_flight = 1;
        CHECK(mgc_session_create(domain, &sd, &session) == MGC_SUCCESS);
        auto desc = init<MgcResourceDesc>(); desc.capacity_bytes = 64; desc.format = MGC_FORMAT_BYTES;
        CHECK(mgc_resource_create(domain, &desc, &input) == MGC_SUCCESS);
        CHECK(mgc_resource_create(domain, &desc, &output) == MGC_SUCCESS);
        auto ld = init<MgcLeaseDesc>(); ld.content_id = 1; ld.byte_length = 64; ld.access = MGC_ACCESS_WRITE;
        MgcLease *writer = nullptr; CHECK(mgc_lease_create(input, &ld, &writer) == MGC_SUCCESS);
        void *p = nullptr; uint64_t n = 0; CHECK(mgc_lease_map(writer, &p, &n) == MGC_SUCCESS && n == 64);
        std::memset(p, 0x5a, static_cast<size_t>(n)); CHECK(mgc_lease_unmap(writer) == MGC_SUCCESS);
        CHECK(mgc_lease_publish_read(writer, &source) == MGC_SUCCESS); mgc_lease_release(writer);
        CHECK(mgc_lease_create(output, &ld, &destination) == MGC_SUCCESS);
    }
    MgcLease *sibling() {
        auto ld = init<MgcLeaseDesc>(); ld.content_id = 1; ld.byte_length = 64; ld.access = MGC_ACCESS_READ;
        MgcLease *result = nullptr; CHECK(mgc_lease_create(input, &ld, &result) == MGC_SUCCESS); return result;
    }
    MgcTicket *copy(MgcLease *selected) {
        auto job = init<MgcSubmitDesc>(); job.operation = MGC_OPERATION_COPY; job.byte_count = 64;
        job.input = selected; job.output = destination;
        mgc_lease_retain(selected); mgc_lease_retain(destination);
        MgcTicket *ticket = nullptr; CHECK(mgc_submit(session, &job, &ticket) == MGC_ACCEPTED && ticket);
        return ticket;
    }
    void reset_destination() {
        mgc_lease_release(destination); mgc_resource_release(output);
        auto desc = init<MgcResourceDesc>(); desc.capacity_bytes = 64; desc.format = MGC_FORMAT_BYTES;
        CHECK(mgc_resource_create(domain, &desc, &output) == MGC_SUCCESS);
        auto ld = init<MgcLeaseDesc>(); ld.content_id = 1; ld.byte_length = 64; ld.access = MGC_ACCESS_WRITE;
        CHECK(mgc_lease_create(output, &ld, &destination) == MGC_SUCCESS);
    }
    void clean() {
        mgc_lease_release(source); mgc_lease_release(destination);
        mgc_resource_release(input); mgc_resource_release(output);
        CHECK(mgc_session_close(session) == MGC_SUCCESS); mgc_domain_release(domain);
        CHECK(mgc_runtime_close(runtime) == MGC_SUCCESS); mgc_runtime_release(runtime);
    }
};

static MgcCompletion completion(MgcTicket *ticket) {
    auto result = init<MgcCompletion>(); CHECK(mgc_ticket_query(ticket, &result) == MGC_SUCCESS); return result;
}
static void held(Fixture &f, MgcTicket *ticket) {
    std::this_thread::sleep_for(std::chrono::milliseconds(18));
    auto state = init<MgcCompletion>();
    CHECK(mgc_ticket_query(ticket, &state) == MGC_BUSY);
    CHECK(mgc_session_drain(f.session, 0) == MGC_TIMEOUT);
    CHECK(mgc_test_guard_pin(f.source) == 1);
}
static void verify_output(Fixture &f) {
    void *p = nullptr; uint64_t n = 0;
    CHECK(mgc_lease_map(f.destination, &p, &n) == MGC_SUCCESS && n == 64);
    for (uint64_t i = 0; i < n; ++i) CHECK(static_cast<unsigned char *>(p)[i] == 0x5a);
    CHECK(mgc_lease_unmap(f.destination) == MGC_SUCCESS);
}

struct Timed { std::atomic<uint64_t> entered{0}; };
static MgcStatus execute(void *context, MgcTicket *) {
    auto *value = static_cast<Timed *>(context);
    value->entered.store(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count()), std::memory_order_release);
    return MGC_SUCCESS;
}
static MgcTicket *submit_timed(Fixture &f, Timed &context, MgcSession *session = nullptr) {
    auto job = init<MgcHostJobDesc>(); job.execute = execute; job.context = &context;
    MgcLease *leases[] = {f.source}; job.leases = leases; job.lease_count = 1;
    mgc_lease_retain(f.source); MgcTicket *ticket = nullptr;
    CHECK(mgc_private_submit_host_job(session ? session : f.session, &job, &ticket) == MGC_ACCEPTED && ticket);
    return ticket;
}

static void wait_value(const std::atomic<int> &value, int expected) {
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (value.load(std::memory_order_acquire) != expected && Clock::now() < deadline) std::this_thread::yield();
    CHECK(value.load(std::memory_order_acquire) == expected);
}
static void wait_bool(const std::atomic<bool> &value) {
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (!value.load(std::memory_order_acquire) && Clock::now() < deadline) std::this_thread::yield();
    CHECK(value.load(std::memory_order_acquire));
}
struct Publish {
    MgcLease *destination = nullptr;
    std::atomic<int> stage{0}, release_status{777};
    std::atomic<bool> proceed{false}, finish{false};
    bool hold_before = false, hold_after = false;
};
static MgcStatus publish_execute(void *context, MgcTicket *ticket) {
    auto &p = *static_cast<Publish *>(context);
    p.stage.store(1, std::memory_order_release);
    if (p.hold_before) while (!p.proceed.load(std::memory_order_acquire)) std::this_thread::yield();
    const auto status = mgc_private_ticket_release_retired_write(ticket, p.destination);
    p.release_status.store(status, std::memory_order_release);
    p.stage.store(2, std::memory_order_release);
    if (p.hold_after) while (!p.finish.load(std::memory_order_acquire)) std::this_thread::yield();
    return status;
}
static MgcTicket *submit_publish(Fixture &f, Publish &context) {
    context.destination = f.destination;
    auto job = init<MgcHostJobDesc>(); job.execute = publish_execute; job.context = &context;
    MgcLease *leases[] = {f.source, f.destination}; job.leases = leases; job.lease_count = 2;
    mgc_lease_retain(f.source); mgc_lease_retain(f.destination); MgcTicket *ticket = nullptr;
    CHECK(mgc_private_submit_host_job(f.session, &job, &ticket) == MGC_ACCEPTED && ticket);
    return ticket;
}
static void open_guard(Fixture &f, uint64_t generation) {
    CHECK(mgc_test_guard_fence(f.source, generation, MGC_SUCCESS) == MGC_SUCCESS);
    CHECK(mgc_test_guard_finish(f.source, generation, MGC_SUCCESS) == MGC_SUCCESS);
}

int main(int argc, char **argv) {
    CHECK(argc == 2); const char *mode = argv[1]; Fixture f;
    if (std::strcmp(mode, "ticket_after_close") == 0) {
        CHECK(mgc_test_guard_arm(f.source, 1) == MGC_SUCCESS);
        MgcTicket *ticket = f.copy(f.source); open_guard(f, 1);
        CHECK(mgc_session_drain(f.session, five_seconds) == MGC_SUCCESS);
        CHECK(completion(ticket).status == MGC_SUCCESS && mgc_test_guard_pin(f.source) == 0);
        /* Retain only the ticket while its former session, all leases and
         * resources, domain and runtime are torn down. Ticket identity and
         * its aliased wake control block must remain alive until release. */
        f.clean();
        CHECK(completion(ticket).status == MGC_SUCCESS);
        CHECK(mgc_ticket_cancel(ticket) == MGC_SUCCESS);
        CHECK(completion(ticket).status == MGC_SUCCESS);
        mgc_ticket_release(ticket); std::puts("PASS ticket outlives closed session and runtime"); return 0;
    }
    if (std::strcmp(mode, "cancel_window") == 0) {
        hold_guard_ready.store(true); CHECK(mgc_test_guard_arm(f.source, 1) == MGC_SUCCESS);
        MgcTicket *ticket = f.copy(f.source); open_guard(f, 1); wait_bool(guard_ready_entered);
        CHECK(mgc_ticket_cancel(ticket) == MGC_SUCCESS); release_guard_ready.store(true, std::memory_order_release);
        CHECK(mgc_session_drain(f.session, five_seconds) == MGC_SUCCESS);
        CHECK(completion(ticket).status == MGC_CANCELLED);
        auto info = init<MgcLeaseInfo>(); CHECK(mgc_lease_info(f.destination, &info) == MGC_CANCELLED);
        CHECK(mgc_test_guard_pin(f.source) == 0);
        mgc_ticket_release(ticket); f.clean(); std::puts("PASS gate-ready pre-execution cancellation invalidates output"); return 0;
    }
    if (!std::strcmp(mode, "publish_before") || !std::strcmp(mode, "publish_after") || !std::strcmp(mode, "ordinary_after")) {
        const bool guarded = std::strcmp(mode, "ordinary_after") != 0;
        const bool before = std::strcmp(mode, "publish_before") == 0;
        if (guarded) CHECK(mgc_test_guard_arm(f.source, 1) == MGC_SUCCESS);
        Publish p; p.hold_before = before; p.hold_after = !before;
        MgcTicket *ticket = submit_publish(f, p);
        if (guarded) open_guard(f, 1);
        wait_value(p.stage, before ? 1 : 2);
        MgcLease *published = nullptr;
        if (before) {
            CHECK(mgc_ticket_cancel(ticket) == MGC_SUCCESS);
        } else {
            CHECK(p.release_status.load() == MGC_SUCCESS);
            CHECK(mgc_lease_publish_read(f.destination, &published) == MGC_SUCCESS);
            CHECK(mgc_ticket_cancel(ticket) == (guarded ? MGC_BUSY : MGC_SUCCESS));
            CHECK(mgc_private_session_cancel(f.session, ticket) == (guarded ? MGC_BUSY : MGC_SUCCESS));
        }
        p.proceed.store(true, std::memory_order_release);
        p.finish.store(true, std::memory_order_release);
        CHECK(mgc_session_drain(f.session, five_seconds) == MGC_SUCCESS);
        CHECK(p.release_status.load() == (before ? MGC_CANCELLED : MGC_SUCCESS));
        const auto done = completion(ticket);
        CHECK(done.status == (guarded && !before ? MGC_SUCCESS : MGC_CANCELLED));
        auto info = init<MgcLeaseInfo>();
        CHECK(mgc_lease_info(f.destination, &info) == (before ? MGC_CANCELLED : MGC_SUCCESS));
        if (published) mgc_lease_release(published);
        mgc_ticket_release(ticket); f.clean(); std::puts("PASS guarded publish cancellation boundary"); return 0;
    }
    if (std::strcmp(mode, "publish_race") == 0) {
        unsigned cancelled = 0, published = 0;
        for (uint64_t i = 1; i <= 256; ++i) {
            CHECK(mgc_test_guard_arm(f.source, i) == MGC_SUCCESS);
            Publish p; p.hold_before = true; p.hold_after = true;
            MgcTicket *ticket = submit_publish(f, p); open_guard(f, i); wait_value(p.stage, 1);
            MgcStatus cancel_status;
            if (i == 1) { cancel_status = mgc_ticket_cancel(ticket); p.proceed.store(true, std::memory_order_release); }
            else {
                p.proceed.store(true, std::memory_order_release);
                if (i == 2) wait_value(p.stage, 2);
                cancel_status = mgc_ticket_cancel(ticket);
            }
            wait_value(p.stage, 2);
            const auto release = p.release_status.load(std::memory_order_acquire);
            if (cancel_status == MGC_SUCCESS) { CHECK(release == MGC_CANCELLED); ++cancelled; }
            else { CHECK(cancel_status == MGC_BUSY && release == MGC_SUCCESS); ++published; }
            p.finish.store(true, std::memory_order_release);
            CHECK(mgc_session_drain(f.session, five_seconds) == MGC_SUCCESS);
            const auto done = completion(ticket);
            CHECK(done.status == (cancel_status == MGC_SUCCESS ? MGC_CANCELLED : MGC_SUCCESS));
            mgc_ticket_release(ticket); f.reset_destination();
        }
        CHECK(cancelled && published); f.clean();
        std::printf("PASS 256 publication/cancellation races cancelled=%u published=%u\n", cancelled, published); return 0;
    }
    if (std::strcmp(mode, "two_deps") == 0) {
        CHECK(mgc_test_guard_arm(f.source, 1) == MGC_SUCCESS);
        CHECK(mgc_test_guard_arm(f.destination, 2) == MGC_SUCCESS);
        Timed observed; auto job = init<MgcHostJobDesc>(); job.execute = execute; job.context = &observed;
        MgcLease *leases[] = {f.source, f.destination}; job.leases = leases; job.lease_count = 2;
        mgc_lease_retain(f.source); mgc_lease_retain(f.destination);
        MgcTicket *ticket = nullptr; CHECK(mgc_private_submit_host_job(f.session, &job, &ticket) == MGC_ACCEPTED);
        held(f, ticket); CHECK(!observed.entered.load(std::memory_order_acquire));
        CHECK(mgc_test_guard_fence(f.source, 1, MGC_SUCCESS) == MGC_SUCCESS);
        CHECK(mgc_test_guard_finish(f.source, 1, MGC_SUCCESS) == MGC_SUCCESS);
        std::this_thread::sleep_for(std::chrono::milliseconds(18));
        CHECK(!observed.entered.load(std::memory_order_acquire) && mgc_session_drain(f.session, 0) == MGC_TIMEOUT);
        CHECK(mgc_test_guard_fence(f.destination, 2, MGC_SUCCESS) == MGC_SUCCESS);
        CHECK(mgc_test_guard_finish(f.destination, 2, MGC_SUCCESS) == MGC_SUCCESS);
        CHECK(mgc_session_drain(f.session, five_seconds) == MGC_SUCCESS);
        CHECK(observed.entered.load(std::memory_order_acquire) && completion(ticket).status == MGC_SUCCESS);
        mgc_ticket_release(ticket); f.clean(); std::puts("PASS actual worker requires both resource gates"); return 0;
    }
    if (std::strcmp(mode, "hol_skip") == 0) {
        /* Two independent guarded copies share one session with
         * max_in_flight=2. The older ticket's producer stays closed; the
         * newer one must still execute instead of waiting behind it. */
        auto desc = init<MgcResourceDesc>(); desc.capacity_bytes = 64; desc.format = MGC_FORMAT_BYTES;
        MgcResource *input2 = nullptr, *output2 = nullptr;
        CHECK(mgc_resource_create(f.domain, &desc, &input2) == MGC_SUCCESS);
        CHECK(mgc_resource_create(f.domain, &desc, &output2) == MGC_SUCCESS);
        auto ld = init<MgcLeaseDesc>(); ld.content_id = 1; ld.byte_length = 64; ld.access = MGC_ACCESS_WRITE;
        MgcLease *writer2 = nullptr; CHECK(mgc_lease_create(input2, &ld, &writer2) == MGC_SUCCESS);
        void *p2 = nullptr; uint64_t n2 = 0;
        CHECK(mgc_lease_map(writer2, &p2, &n2) == MGC_SUCCESS && n2 == 64);
        std::memset(p2, 0x3c, static_cast<size_t>(n2));
        CHECK(mgc_lease_unmap(writer2) == MGC_SUCCESS);
        MgcLease *source2 = nullptr;
        CHECK(mgc_lease_publish_read(writer2, &source2) == MGC_SUCCESS);
        mgc_lease_release(writer2);
        MgcLease *dest2 = nullptr; CHECK(mgc_lease_create(output2, &ld, &dest2) == MGC_SUCCESS);
        auto sd = init<MgcSessionDesc>(); sd.max_in_flight = 2; MgcSession *session = nullptr;
        CHECK(mgc_session_create(f.domain, &sd, &session) == MGC_SUCCESS);
        auto submit_copy = [&](MgcSession *s, MgcLease *in, MgcLease *out) {
            auto job = init<MgcSubmitDesc>(); job.operation = MGC_OPERATION_COPY; job.byte_count = 64;
            job.input = in; job.output = out;
            mgc_lease_retain(in); mgc_lease_retain(out);
            MgcTicket *ticket = nullptr;
            CHECK(mgc_submit(s, &job, &ticket) == MGC_ACCEPTED && ticket);
            return ticket;
        };
        CHECK(mgc_test_guard_arm(f.source, 1) == MGC_SUCCESS);
        MgcTicket *first = submit_copy(session, f.source, f.destination);
        CHECK(mgc_test_guard_arm(source2, 2) == MGC_SUCCESS);
        MgcTicket *second = submit_copy(session, source2, dest2);
        std::this_thread::sleep_for(std::chrono::milliseconds(18));
        auto state = init<MgcCompletion>();
        CHECK(mgc_ticket_query(first, &state) == MGC_BUSY);
        CHECK(mgc_ticket_query(second, &state) == MGC_BUSY);
        /* Open only the newer producer. The older ticket must remain blocked. */
        CHECK(mgc_test_guard_fence(source2, 2, MGC_SUCCESS) == MGC_SUCCESS);
        CHECK(mgc_test_guard_finish(source2, 2, MGC_SUCCESS) == MGC_SUCCESS);
        const auto deadline = Clock::now() + std::chrono::seconds(5);
        for (;;) {
            const auto rc = mgc_ticket_query(second, &state);
            CHECK(rc == MGC_SUCCESS || rc == MGC_BUSY);
            if (rc == MGC_SUCCESS) break;
            CHECK(Clock::now() < deadline);
            std::this_thread::yield();
        }
        CHECK(state.status == MGC_SUCCESS);
        CHECK(mgc_ticket_query(first, &state) == MGC_BUSY);
        CHECK(mgc_test_guard_pin(f.source) == 1);
        CHECK(mgc_test_guard_pin(source2) == 0);
        CHECK(mgc_test_guard_waiters(f.source) == 1);
        /* Open the older producer; both must now finish. */
        CHECK(mgc_test_guard_fence(f.source, 1, MGC_SUCCESS) == MGC_SUCCESS);
        CHECK(mgc_test_guard_finish(f.source, 1, MGC_SUCCESS) == MGC_SUCCESS);
        CHECK(mgc_session_drain(session, five_seconds) == MGC_SUCCESS);
        CHECK(completion(first).status == MGC_SUCCESS);
        CHECK(completion(second).status == MGC_SUCCESS);
        /* The newer ticket must actually have copied its own pattern. */
        void *out = nullptr; uint64_t out_n = 0;
        CHECK(mgc_lease_map(dest2, &out, &out_n) == MGC_SUCCESS && out_n == 64);
        for (uint64_t i = 0; i < out_n; ++i) CHECK(static_cast<unsigned char *>(out)[i] == 0x3c);
        CHECK(mgc_lease_unmap(dest2) == MGC_SUCCESS);
        void *out1 = nullptr; uint64_t out1_n = 0;
        CHECK(mgc_lease_map(f.destination, &out1, &out1_n) == MGC_SUCCESS && out1_n == 64);
        for (uint64_t i = 0; i < out1_n; ++i) CHECK(static_cast<unsigned char *>(out1)[i] == 0x5a);
        CHECK(mgc_lease_unmap(f.destination) == MGC_SUCCESS);
        mgc_ticket_release(first); mgc_ticket_release(second);
        mgc_lease_release(source2); mgc_lease_release(dest2);
        mgc_resource_release(input2); mgc_resource_release(output2);
        CHECK(mgc_session_close(session) == MGC_SUCCESS);
        f.clean(); std::puts("PASS ready ticket runs ahead of an older blocked producer"); return 0;
    }
    if (std::strcmp(mode, "waiter_reuse") == 0) {
        /* Keep 17 sequential sessions alive. Completed tickets must unregister
         * so the single waiter slot is reused; the next session still gets an
         * actual wake notification rather than a fallback. */
        std::vector<MgcSession *> sessions;
        for (uint64_t i = 1; i <= 17; ++i) {
            auto sd = init<MgcSessionDesc>(); sd.max_in_flight = 1; MgcSession *session = nullptr;
            CHECK(mgc_session_create(f.domain, &sd, &session) == MGC_SUCCESS); sessions.push_back(session);
            CHECK(mgc_test_guard_arm(f.source, i) == MGC_SUCCESS);
            Timed observed; MgcTicket *ticket = submit_timed(f, observed, session);
            CHECK(mgc_test_guard_waiters(f.source) == 1);
            std::this_thread::sleep_for(std::chrono::milliseconds(12));
            CHECK(!observed.entered.load(std::memory_order_acquire));
            CHECK(mgc_test_guard_fence(f.source, i, MGC_SUCCESS) == MGC_SUCCESS);
            CHECK(mgc_test_guard_finish(f.source, i, MGC_SUCCESS) == MGC_SUCCESS);
            CHECK(mgc_session_drain(session, five_seconds) == MGC_SUCCESS);
            CHECK(observed.entered.load(std::memory_order_acquire) && completion(ticket).status == MGC_SUCCESS);
            CHECK(mgc_test_guard_waiters(f.source) == 0);
            mgc_ticket_release(ticket);
        }
        for (auto *session : sessions) CHECK(mgc_session_close(session) == MGC_SUCCESS);
        f.clean(); std::puts("PASS waiter deregistration, seventeenth session and teardown"); return 0;
    }
    if (std::strcmp(mode, "race") == 0) {
        for (uint64_t i = 1; i <= 256; ++i) {
            CHECK(mgc_test_guard_arm(f.source, i) == MGC_SUCCESS);
            Timed observed; MgcTicket *ticket = submit_timed(f, observed);
            /* The last gate may open before, during, or after the worker's
             * dependency snapshot. Lost wakeups must not strand the job. */
            if (i & 1) {
                CHECK(mgc_test_guard_fence(f.source, i, MGC_SUCCESS) == MGC_SUCCESS);
                CHECK(mgc_test_guard_finish(f.source, i, MGC_SUCCESS) == MGC_SUCCESS);
            } else {
                CHECK(mgc_test_guard_finish(f.source, i, MGC_SUCCESS) == MGC_SUCCESS);
                CHECK(mgc_test_guard_fence(f.source, i, MGC_SUCCESS) == MGC_SUCCESS);
            }
            CHECK(mgc_session_drain(f.session, five_seconds) == MGC_SUCCESS);
            CHECK(observed.entered.load(std::memory_order_acquire) && completion(ticket).status == MGC_SUCCESS);
            mgc_ticket_release(ticket);
        }
        f.clean(); std::puts("PASS actual worker repeated gate/open race"); return 0;
    }
    if (std::strcmp(mode, "private_cancel") == 0) {
        CHECK(mgc_test_guard_arm(f.source, 1) == MGC_SUCCESS);
        Timed observed; MgcTicket *ticket = submit_timed(f, observed); held(f, ticket);
        CHECK(mgc_private_session_cancel(f.session, ticket) == MGC_SUCCESS);
        CHECK(mgc_session_drain(f.session, five_seconds) == MGC_SUCCESS);
        CHECK(!observed.entered.load(std::memory_order_acquire));
        CHECK(completion(ticket).status == MGC_CANCELLED && mgc_test_guard_pin(f.source) == 1);
        CHECK(mgc_test_guard_fence(f.source, 1, MGC_SUCCESS) == MGC_SUCCESS);
        CHECK(mgc_test_guard_finish(f.source, 1, MGC_SUCCESS) == MGC_SUCCESS);
        mgc_ticket_release(ticket); f.clean(); std::puts("PASS private cancel preserves producer pin"); return 0;
    }
    if (std::strcmp(mode, "latency") == 0) {
        std::vector<uint64_t> fence_first, cleanup_first;
        uint64_t generation = 0;
        for (int pair = 0; pair < 64; ++pair) for (int order = 0; order < 2; ++order) {
            const bool fence_before_cleanup = ((pair & 1) ? 1 - order : order) == 0;
            const auto i = ++generation;
            CHECK(mgc_test_guard_arm(f.source, i) == MGC_SUCCESS);
            Timed observed; MgcTicket *ticket = submit_timed(f, observed);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            CHECK(!observed.entered.load(std::memory_order_acquire));
            if (fence_before_cleanup) CHECK(mgc_test_guard_fence(f.source, i, MGC_SUCCESS) == MGC_SUCCESS);
            else CHECK(mgc_test_guard_finish(f.source, i, MGC_SUCCESS) == MGC_SUCCESS);
            CHECK(!observed.entered.load(std::memory_order_acquire));
            const auto start = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now().time_since_epoch()).count());
            if (fence_before_cleanup) CHECK(mgc_test_guard_finish(f.source, i, MGC_SUCCESS) == MGC_SUCCESS);
            else CHECK(mgc_test_guard_fence(f.source, i, MGC_SUCCESS) == MGC_SUCCESS);
            CHECK(mgc_session_drain(f.session, five_seconds) == MGC_SUCCESS);
            const auto entered = observed.entered.load(std::memory_order_acquire);
            CHECK(entered >= start && completion(ticket).status == MGC_SUCCESS);
            (fence_before_cleanup ? fence_first : cleanup_first).push_back(entered - start);
            mgc_ticket_release(ticket);
        }
        f.clean(); std::printf("{\"scope\":\"host synthetic fence, real Pack worker; gate-open to execute\",\"fenceFirstNs\":[");
        for (size_t i = 0; i < fence_first.size(); ++i) std::printf("%s%llu", i ? "," : "", static_cast<unsigned long long>(fence_first[i]));
        std::printf("],\"cleanupFirstNs\":[");
        for (size_t i = 0; i < cleanup_first.size(); ++i) std::printf("%s%llu", i ? "," : "", static_cast<unsigned long long>(cleanup_first[i]));
        std::puts("]}"); return 0;
    }
    CHECK(mgc_test_guard_arm(f.source, 7) == MGC_SUCCESS);
    const bool use_sibling = std::strcmp(mode, "sibling") == 0;
    MgcLease *sibling = use_sibling ? f.sibling() : nullptr;
    MgcTicket *ticket = f.copy(sibling ? sibling : f.source);
    held(f, ticket);
    CHECK(mgc_test_guard_finish(f.source, 8, MGC_SUCCESS) == MGC_INVALID_ARGUMENT);
    if (!std::strcmp(mode, "success") || !std::strcmp(mode, "closing") || use_sibling) {
        if (!std::strcmp(mode, "closing")) {
            CHECK(mgc_session_close(f.session) == MGC_BUSY);
            held(f, ticket); /* closing must not bypass the producer gate */
        }
        CHECK(mgc_test_guard_fence(f.source, 7, MGC_SUCCESS) == MGC_SUCCESS);
        held(f, ticket);
        CHECK(mgc_test_guard_finish(f.source, 7, MGC_SUCCESS) == MGC_SUCCESS);
    } else if (!std::strcmp(mode, "guard_first")) {
        CHECK(mgc_test_guard_finish(f.source, 7, MGC_SUCCESS) == MGC_SUCCESS);
        held(f, ticket);
        CHECK(mgc_test_guard_fence(f.source, 7, MGC_SUCCESS) == MGC_SUCCESS);
    } else if (!std::strcmp(mode, "failed") || !std::strcmp(mode, "unknown") || !std::strcmp(mode, "cancel_failed")) {
        const auto status = !std::strcmp(mode, "unknown") ? MGC_INTERNAL_ERROR : MGC_PRODUCER_FAILED;
        CHECK(mgc_test_guard_fence(f.source, 7, status) == MGC_SUCCESS);
        if (!std::strcmp(mode, "cancel_failed")) CHECK(mgc_ticket_cancel(ticket) == MGC_SUCCESS);
        CHECK(mgc_session_drain(f.session, five_seconds) == MGC_SUCCESS);
        const auto done = completion(ticket); CHECK(done.status == status && done.terminal == MGC_TERMINAL_FAILED);
        auto info = init<MgcLeaseInfo>(); CHECK(mgc_lease_info(f.destination, &info) == status);
        CHECK(mgc_test_guard_pin(f.source) == 1);
        std::puts("PASS failed/unknown producer preserves synthetic pin and invalidates output");
        std::fflush(stdout); std::_Exit(0);
    } else if (!std::strcmp(mode, "cancelled")) {
        CHECK(mgc_ticket_cancel(ticket) == MGC_SUCCESS);
        CHECK(mgc_session_drain(f.session, five_seconds) == MGC_SUCCESS);
        const auto done = completion(ticket); CHECK(done.status == MGC_CANCELLED && done.terminal == MGC_TERMINAL_CANCELLED);
        auto info = init<MgcLeaseInfo>(); CHECK(mgc_lease_info(f.destination, &info) == MGC_CANCELLED);
        CHECK(mgc_test_guard_pin(f.source) == 1);
        CHECK(mgc_test_guard_fence(f.source, 7, MGC_SUCCESS) == MGC_SUCCESS);
        CHECK(mgc_test_guard_finish(f.source, 7, MGC_SUCCESS) == MGC_SUCCESS);
        CHECK(mgc_test_guard_pin(f.source) == 0);
        mgc_ticket_release(ticket); f.clean(); std::puts("PASS cancel does not retire producer pin"); return 0;
    } else return 2;
    CHECK(mgc_session_drain(f.session, five_seconds) == MGC_SUCCESS);
    const auto done = completion(ticket); CHECK(done.status == MGC_SUCCESS && done.terminal == MGC_TERMINAL_SUCCESS);
    CHECK(mgc_test_guard_pin(f.source) == 0); verify_output(f);
    mgc_ticket_release(ticket); if (sibling) mgc_lease_release(sibling);
    f.clean(); std::puts("PASS queued copy waits for both gates");
}
