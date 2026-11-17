#pragma once
#include <cstdint>
#include <meguviz/compute.h>

namespace meguviz_compute_detail {
// The caller holds the owning resource mutex.  The generation belongs to the
// reservation, never to a pointer or file descriptor which may be reused.
inline MgcStatus finish_producer_guard(uint8_t &guard, uint64_t actual_generation,
    uint64_t supplied_generation, MgcStatus cleanup_status,
    bool &content_invalid, MgcStatus &content_failure) noexcept {
    if (!supplied_generation || actual_generation != supplied_generation ||
        (cleanup_status != MGC_SUCCESS && cleanup_status != MGC_PRODUCER_FAILED))
        return MGC_INVALID_ARGUMENT;
    if (guard == 3 && cleanup_status == MGC_PRODUCER_FAILED) return MGC_SUCCESS;
    if (guard != 1) return MGC_INVALID_ARGUMENT;
    if (cleanup_status == MGC_PRODUCER_FAILED) {
        content_invalid = true;
        content_failure = MGC_PRODUCER_FAILED;
        guard = 3; // exact producer pin remains quarantined
    } else guard = 2;
    return MGC_SUCCESS;
}
inline MgcStatus producer_gate_status(uint8_t guard, bool invalid, MgcStatus failure) noexcept {
    if (invalid) return failure;
    return guard == 1 || guard == 3 ? MGC_BUSY : MGC_SUCCESS;
}
inline bool same_producer_content(uint32_t external_state, uint64_t lease_content,
    uint64_t current_content) noexcept {
    return external_state != 3 || (lease_content && lease_content == current_content);
}
inline bool guarded_private_lease_allowed(bool guarded, bool declared) noexcept {
    return !guarded || declared;
}
}
