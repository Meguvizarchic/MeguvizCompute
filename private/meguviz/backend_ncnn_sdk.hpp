#ifndef MEGUVIZ_BACKEND_NCNN_SDK_HPP
#define MEGUVIZ_BACKEND_NCNN_SDK_HPP
#include <meguviz/compute_job.h>
#include <meguviz/shaderpack.h>
#include <net.h>
#include <meguviz/ncnn_options_json.hpp>
#include <memory>
#if NCNN_VULKAN
#include <gpu.h>
#include <command.h>
#include <pipeline.h>
#include <string>
#endif
namespace meguviz {
inline std::string ncnn_shader_options_json(const ncnn::Option &options) {
    return meguviz_options_detail::shader_options_json(options);
}
/* Runs under the same domain serial slot used by tickets. CPU receives null
 * device. device and allocator objects may not escape this call unless their
 * owner retains the domain and never uses them outside another such scope.
 * Reentrant calls on the active worker scope are allowed. Normal nested entry
 * on a different execution slot returns INCOMPATIBLE_DOMAIN before locking.
 * GPU scopes retain the established shared serialization order; CPU domains
 * have separate slots so GPU waits do not serialize unrelated CPU work. */
using NcnnDomainFn = MgcStatus (*)(void *context,
#if NCNN_VULKAN
    const ncnn::VulkanDevice *device
#else
    const void *device
#endif
);
MgcStatus with_ncnn_domain(MgcDomain *domain, NcnnDomainFn fn, void *context);
/* Exclusive owned-object destruction only: no submit, record, user callback or
 * reuse of a pending object's resources. This may run while ordinary device
 * admission is blocked. The caller must first establish physical retirement. */
MgcStatus with_ncnn_domain_cleanup(MgcDomain *domain, NcnnDomainFn fn, void *context);
bool ncnn_domain_is_lost(MgcDomain *domain);
bool ncnn_domain_loss_retired(MgcDomain *domain);
void ncnn_report_device_lost(MgcDomain *domain);
#if NCNN_VULKAN
/* Standalone model-upload ownership, not an accepted ticket. The owner pins the
 * domain/instance and this code image. The host must keep its loader reference
 * until every owner has successfully released; this is not self-unload support.
 * All owner operations and destruction of a handle are externally synchronized.
 * A scope marks pending BEFORE invoking fn. Only a callback in that owner scope
 * may clear it after proving no pending work. Throwing restores pending. Pending
 * blocks ordinary admission on every domain using the same actual VkDevice;
 * release returns BUSY and retains the complete owner. Lost handles are not
 * reclaimed automatically. At most 16 owners per actual device are admitted.
 * Callbacks must use finite fence waits; no API interrupts a stuck driver call.
 * Scope entry from a ticket is rejected until ticket deferral is implemented. */
struct NcnnDomainOwner;
MgcStatus ncnn_domain_owner_create(MgcDomain *domain, NcnnDomainOwner **owner);
MgcStatus with_ncnn_domain_owner(NcnnDomainOwner *owner, NcnnDomainFn fn, void *context);
/* Same ownership contract, but returns BUSY rather than waiting to enter the
 * device slot. This does not interrupt the callback or its driver calls. */
MgcStatus try_with_ncnn_domain_owner(NcnnDomainOwner *owner, NcnnDomainFn fn, void *context);
/* Lost-device retirement only: callback may wait/query/cleanup but may not
 * record or submit new GPU work. */
MgcStatus try_with_ncnn_domain_owner_retire(NcnnDomainOwner *owner, NcnnDomainFn fn, void *context);
MgcStatus ncnn_domain_owner_pending(NcnnDomainOwner *owner, bool pending);
MgcStatus ncnn_domain_owner_release(NcnnDomainOwner *owner);
struct NcnnGpuView {
    MgcLease *lease;
    /* Entire allocation. Shader must apply info.byte_offset and stay within
     * info.byte_length. AHB image views always cover the whole allocation. */
    ncnn::VkMat *buffer;
    ncnn::VkImageMat *image;
    MgcLeaseInfo info;
};
struct NcnnGpuScope {
    const ncnn::VulkanDevice *device;
    ncnn::VkAllocator *blob_allocator;
    ncnn::VkAllocator *staging_allocator;
    ncnn::VkCompute *command;
    NcnnGpuView *views;
    uint32_t view_count;
    /* Register before any auxiliary command may submit. keepalive owns the
     * auxiliary objects; SUCCESS certifies their physical retirement only. */
    MgcStatus (*retire_aux)(void *context, uint64_t wait_ns) = nullptr;
    void *aux_context = nullptr;
};
using NcnnGpuFn = MgcStatus (*)(void *context, NcnnGpuScope &scope);
/* Exact-build source SDK. All leases must belong to domain, be GPU storage,
 * and be held by the active ticket (or idle during a standalone scope).
 * Callback records only; it must keep pipeline/Net/Extractor/download targets
 * alive in context until return. Compute submits even on callback rejection,
 * waits for physical retirement and performs host finalize before returning.
 * AHB ownership is acquired from FOREIGN before recording and released to
 * GENERAL/FOREIGN after retirement. No object may escape the scope except an
 * explicitly retained resource/domain owner. Repeated scopes allow small host
 * guard summaries between GPU batches without downloading full images. */
MgcStatus with_ncnn_gpu(MgcDomain *domain, MgcLease *const *leases, uint32_t lease_count,
                        NcnnGpuFn fn, void *context);
/* Accepted-ticket source SDK path. keepalive owns every callback/download
 * target and algorithm object until physical retirement. BUSY may mean a
 * deferred batch; ncnn_gpu_scope_deferred distinguishes that case from ordinary
 * admission rejection. A successful recorded callback resumes only after all
 * batch dependencies and host finalization; a partial callback never replays.
 * Unmanaged standalone scopes keep their separately documented legacy limits. */
MgcStatus with_ncnn_gpu_owned(MgcDomain *domain, MgcLease *const *leases, uint32_t lease_count,
    std::shared_ptr<void> keepalive, NcnnGpuFn fn, void *context);
bool ncnn_gpu_scope_deferred();
/* Pack owns the cached pipeline; retain pack through every recorded command's
 * retirement. Key includes exact artifact/Options, specializations and local
 * size in its retained device domain. Raw-SPV cache remains ncnn's device cache,
 * separate from opt.pipeline_cache for builtin shader indices. */
/* Older no-Option entry is retained as a clear UNSUPPORTED_ABI result. */
MgcStatus ncnn_shaderpack_pipeline(MgcShaderpack *pack, const MgcShaderArtifactRequest *request,
                                  const std::vector<ncnn::vk_specialization_type> &specializations,
                                  ncnn::Pipeline **borrowed_pipeline);
MgcStatus ncnn_shaderpack_pipeline(MgcShaderpack *pack, const MgcShaderArtifactRequest *request,
                                  const ncnn::Option &effective_options,
                                  const std::vector<ncnn::vk_specialization_type> &specializations,
                                  ncnn::Pipeline **borrowed_pipeline);
/* Canonical ShaderTools-compatible owner macro closure, captured from exact
 * enabled VkDevice inputs. Properties remain property macros, never features. */
MgcStatus ncnn_shader_target_facts(const ncnn::VulkanDevice *device, std::string &canonical_json);
#endif
const char *private_sdk_fingerprint();
}
#endif
