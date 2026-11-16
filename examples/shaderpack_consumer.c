#include "consumer_common.h"
#include <meguviz/shaderpack.h>
static int digest(const char*s,MgcHash*h){if(strlen(s)!=64)return 0;for(int i=0;i<32;++i){unsigned v=0;if(sscanf(s+i*2,"%2x",&v)!=1)return 0;h->bytes[i]=(uint8_t)v;}return 1;}
int main(int argc,char**argv){
    if(argc!=4){fprintf(stderr,"usage: shaderpack_consumer pack-root variant-key effective-options-hash\n");return 2;}
    MgcRuntime*r=0;MgcDomain*d=0;MgcSession*s=0;CHECK(start_runtime(&r,&d,&s)==0);
    MgcShaderpackDesc desc={0};desc.header=(MgcHeader)MGC_HEADER_INIT(MgcShaderpackDesc);desc.root_path_utf8=argv[1];desc.root_path_length=(uint32_t)strlen(argv[1]);desc.profile=MGC_SHADER_PROFILE_NCNN_COMPAT;
    MgcShaderpack*p=0;CHECK(mgc_shaderpack_open(d,&desc,&p)==0);
    MgcShaderArtifactRequest request={0};request.header=(MgcHeader)MGC_HEADER_INIT(MgcShaderArtifactRequest);
    request.logical_shader_id="meguviz-compute/affine_f32";request.logical_shader_id_length=(uint32_t)strlen(request.logical_shader_id);
    CHECK(digest(argv[2],&request.variant_key)&&digest(argv[3],&request.effective_options_hash));
    MgcShaderArtifactView v={0};v.header=(MgcHeader)MGC_HEADER_INIT(MgcShaderArtifactView);CHECK(mgc_shaderpack_select(p,&request,&v)==0);
    CHECK(v.spirv_words&&v.spirv_words[0]==0x07230203&&v.local_size[0]==64);
    request.effective_options_hash.bytes[0]^=1;CHECK(mgc_shaderpack_select(p,&request,&v)==MGC_ARTIFACT_MISMATCH);
    mgc_shaderpack_release(p);CHECK(mgc_session_close(s)==0);mgc_domain_release(d);CHECK(mgc_runtime_close(r)==0);mgc_runtime_release(r);
    printf("PASS: runtime shaderpack loaded %llu verified bytes; Options mismatch rejected\n",(unsigned long long)v.spirv_size_bytes);return 0;
}
