#include "../src/core/producer_guard.hpp"
#include "../src/core/sync_file_wait.hpp"
#include <cerrno>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
extern "C" int poll(struct pollfd *, nfds_t, int) { std::abort(); }
extern "C" int ioctl(int, unsigned long, ...) { std::abort(); }

int main() {
    using namespace meguviz_compute_detail;
    uint8_t guard = 1; bool invalid = false; MgcStatus failure = MGC_DEVICE_LOST;
    // Both the original lease and a sibling for the same content may be
    // admitted; neither may consume while the cleanup gate is closed.
    CHECK(same_producer_content(3, 42, 42));
    CHECK(!same_producer_content(3, 41, 42));
    CHECK(guarded_private_lease_allowed(false, false));
    CHECK(guarded_private_lease_allowed(true, true));
    CHECK(!guarded_private_lease_allowed(true, false));
    CHECK(producer_gate_status(guard, invalid, failure) == MGC_BUSY);
    CHECK(finish_producer_guard(guard, 8, 7, MGC_SUCCESS, invalid, failure) == MGC_INVALID_ARGUMENT);
    CHECK(guard == 1 && !invalid);
    CHECK(finish_producer_guard(guard, 8, 8, MGC_SUCCESS, invalid, failure) == MGC_SUCCESS);
    CHECK(producer_gate_status(guard, invalid, failure) == MGC_SUCCESS);
    CHECK(finish_producer_guard(guard, 8, 8, MGC_SUCCESS, invalid, failure) == MGC_INVALID_ARGUMENT);
    guard = 1;
    CHECK(finish_producer_guard(guard, 9, 9, MGC_PRODUCER_FAILED, invalid, failure) == MGC_SUCCESS);
    CHECK(guard == 3 && invalid && producer_gate_status(guard, invalid, failure) == MGC_PRODUCER_FAILED);
    CHECK(finish_producer_guard(guard, 9, 9, MGC_PRODUCER_FAILED, invalid, failure) == MGC_SUCCESS);
    CHECK(finish_producer_guard(guard, 9, 9, MGC_SUCCESS, invalid, failure) == MGC_INVALID_ARGUMENT);
    int state = 0, calls = 0;
    auto ioctl_fn = [&](int, unsigned long request, void *data) {
        ++calls;
        if (request == SYNC_IOC_FILE_INFO) {
            if (state == 3) { errno = ENOTTY; return -1; }
            if (state == 4) { errno = EIO; return -1; }
            static_cast<sync_file_info *>(data)->status = state == 0 ? 0 : state == 1 ? 1 : -EIO;
            return 0;
        }
        CHECK(request == legacy_info_ioctl);
        auto *legacy = static_cast<legacy_fence_buffer *>(data);
        legacy->header.status = 1; return 0;
    };
    CHECK(sync_file_probe_impl(11, ioctl_fn) == 1 && calls == 1);
    state = 1; CHECK(sync_file_probe_impl(11, ioctl_fn) == 0);
    state = 2; CHECK(sync_file_probe_impl(11, ioctl_fn) == -2);
    state = 3; calls = 0; CHECK(sync_file_probe_impl(11, ioctl_fn) == 0 && calls == 2);
    state = 4; CHECK(sync_file_probe_impl(11, ioctl_fn) == -1);
    std::puts("PASS guarded generation, content gate, and actual sync-file status probe");
}
