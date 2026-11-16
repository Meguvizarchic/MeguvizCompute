#ifndef MEGUVIZ_SHADERPACK_H
#define MEGUVIZ_SHADERPACK_H
#include "compute.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct MgcShaderpack MgcShaderpack;
enum { MGC_SHADER_PROFILE_NCNN_COMPAT = 1, MGC_SHADER_PROFILE_NATIVE_VULKAN = 2 };
typedef struct MgcShaderpackDesc {
    MgcHeader header;
    const char *root_path_utf8;
    uint32_t root_path_length;
    uint32_t profile;
    MgcHash manifest_hash;
    uint32_t required_flags;
    uint32_t reserved[3];
} MgcShaderpackDesc;
typedef struct MgcShaderArtifactRequest {
    MgcHeader header;
    const char *logical_shader_id;
    uint32_t logical_shader_id_length;
    uint32_t required_flags;
    MgcHash variant_key;
    MgcHash effective_options_hash;
    uint32_t reserved[4];
    /* Optional host selection bindings. Execution additionally verifies these
     * canonical objects against the actual device and the caller's Option. */
    const char *effective_options_json;
    uint32_t effective_options_json_length;
    uint32_t reserved_tail;
    const char *target_facts_json;
    uint32_t target_facts_json_length;
    uint32_t reserved_tail2;
    MgcHash target_facts_hash;
} MgcShaderArtifactRequest;
typedef struct MgcShaderArtifactView {
    MgcHeader header;
    const uint32_t *spirv_words;
    uint64_t spirv_size_bytes;
    uint32_t profile;
    uint32_t local_size[3];
    MgcHash spirv_hash;
    uint32_t reserved[4];
} MgcShaderArtifactView;
/* Copies metadata and verified artifact bytes. File content is bounded, paths
 * cannot escape root, canonical manifest/SPV/reflection hashes must match.
 * Open is host validation, not a device qualification. */
MGC_API MgcStatus mgc_shaderpack_open(MgcDomain *domain, const MgcShaderpackDesc *desc, MgcShaderpack **pack);
MGC_API void mgc_shaderpack_retain(MgcShaderpack *pack);
MGC_API void mgc_shaderpack_release(MgcShaderpack *pack);
MGC_API MgcStatus mgc_shaderpack_domain_identity(MgcShaderpack *pack, uint64_t *domain_id, uint64_t *generation);
/* View bytes are borrowed until pack release. Native Vulkan artifacts are not
 * passed through ncnn's named parser; their pipeline owner consumes this view. */
MGC_API MgcStatus mgc_shaderpack_select(MgcShaderpack *pack, const MgcShaderArtifactRequest *request, MgcShaderArtifactView *view);
#ifdef __cplusplus
}
#endif
#endif
