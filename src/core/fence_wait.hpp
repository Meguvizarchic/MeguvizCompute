#pragma once
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <poll.h>
#include <meguviz/compute.h>

namespace meguviz_compute_detail {
// Clock and poll are injected so the production loop can be exercised
// without waiting days for a capped poll interval to expire.
template<class Now, class Poll>
MgcStatus wait_fd(int fd, uint64_t timeout_ns, Now now, Poll poll_once)
{
    if (fd < 0) return MGC_SUCCESS;
    const auto start = now();
    for (;;)
    {
        int ms = -1;
        if (timeout_ns != UINT64_MAX)
        {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now() - start).count();
            if (uint64_t(elapsed) >= timeout_ns && timeout_ns) return MGC_TIMEOUT;
            uint64_t remaining = timeout_ns ? timeout_ns - uint64_t(elapsed) : 0;
            const uint64_t milliseconds = remaining / 1000000 + (remaining % 1000000 != 0);
            ms = int(std::min<uint64_t>(milliseconds, INT32_MAX));
        }
        pollfd p{fd, POLLIN, 0};
        int result = poll_once(&p, ms);
        if (result < 0 && errno == EINTR) continue;
        // A capped poll interval can expire before a finite overall deadline.
        // Poll mode remains a single nonblocking attempt.
        if (!result) { if (!timeout_ns) return MGC_TIMEOUT; continue; }
        if (result < 0 || (p.revents & (POLLERR | POLLNVAL))) return MGC_INTERNAL_ERROR;
        if (!(p.revents & POLLIN)) return MGC_INTERNAL_ERROR;
        return MGC_SUCCESS;
    }
}
}
