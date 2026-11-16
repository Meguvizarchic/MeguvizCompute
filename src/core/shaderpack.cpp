#include <meguviz/shaderpack.h>
#include "pack_data.hpp"
#include <atomic>
#include <cstring>
#include <mutex>
#include <set>
#include <memory>
#include <functional>
#if MGC_HAVE_NCNN
#include <meguviz/backend_ncnn_sdk.hpp>
#endif
namespace pd = mgc_pack_detail;
namespace {
struct Artifact {
    std::string logical, variant, effective, digest;
    std::vector<uint32_t> words;
    uint32_t local[3]{};
    pd::J layout;
    pd::J options, facts;
    std::string facts_hash, target;
    std::set<uint32_t> capabilities;
};
std::string hex(const MgcHash &h) {
    static const char *digits="0123456789abcdef"; std::string out;
    for(auto b:h.bytes){out+=digits[b>>4];out+=digits[b&15];}return out;
}
void unhex(const std::string &s,MgcHash &h) {
    auto n=[](char c){return c<='9'?c-'0':c-'a'+10;};
    for(int i=0;i<32;++i)h.bytes[i]=uint8_t(n(s[i*2])*16+n(s[i*2+1]));
}
template<class T> bool descriptor(const T *p,T &copy,size_t min) {
    if(!p||p->header.abi_major!=MGC_ABI_MAJOR||p->header.struct_size<min)return false;
    std::memset(&copy,0,sizeof(copy));std::memcpy(&copy,p,std::min<size_t>(sizeof(copy),p->header.struct_size));return true;
}
bool zero(const void *p,size_t n){auto*b=static_cast<const uint8_t*>(p);for(size_t i=0;i<n;++i)if(b[i])return false;return true;}
std::string bytes(const std::filesystem::path &root,const pd::J &a,const char *path,const char *digest) {
    auto expected=a.at(digest).str();pd::require(pd::valid_hash(expected),4,"invalid_hash");
    auto data=pd::read(pd::within(root,a.at(path).str()));pd::require(pd::sha256(data)==expected,4,"artifact_hash");return data;
}
std::string spv_text(const uint32_t *v,size_t n){auto*p=reinterpret_cast<const char*>(v);size_t count=0;while(count<n*4&&p[count])++count;pd::require(count<n*4,4,"spirv_unterminated_name");return std::string(p,count);}
void interface_check(Artifact &a,const pd::J &meta,uint32_t profile) {
    const auto&w=a.words;pd::require(w.size()>=5&&w[0]==0x07230203&&w[3]&&w[3]<=1048576&&w[4]==0,4,"spirv_header");
    a.target=meta.at("target_env").str();
    uint32_t maximum=a.target=="vulkan1.0"?0x00010000:a.target=="vulkan1.1"?0x00010300:a.target=="vulkan1.3"?0x00010600:0;
    pd::require(maximum&&(w[1]==0x00010000||w[1]==0x00010300||w[1]==0x00010600)&&w[1]<=maximum,3,"spirv_target_version");
    std::map<uint32_t,std::string> names;std::map<uint32_t,std::set<uint32_t>> members;
    std::map<uint32_t,std::map<uint32_t,uint32_t>> decorations;bool entry=false;
    for(size_t p=5;p<w.size();) {
        uint32_t count=w[p]>>16,op=w[p]&65535;pd::require(count&&count<=w.size()-p,4,"spirv_bounds");
        if(op==17){pd::require(count==2,4,"spirv_capability_operand");a.capabilities.insert(w[p+1]);}
        // Protect every ID/operand that the pinned ncnn named parser indexes.
        if(op==5||op==6||op==25||op==27||op==32||op==59||op==71)
            pd::require(count>=2&&w[p+1]<w[3],4,"spirv_id_bound");
        if(op==5)pd::require(count>=3,4,"spirv_name_length");
        if(op==6)pd::require(count>=4,4,"spirv_member_length");
        if(op==25)pd::require(count>=9,4,"spirv_image_length");
        if(op==27)pd::require(count==3&&w[p+2]<w[3],4,"spirv_sampled_image");
        if(op==32)pd::require(count==4&&w[p+3]<w[3],4,"spirv_pointer_id");
        if(op==59)pd::require(count>=4&&w[p+2]<w[3],4,"spirv_variable_id");
        if(op==71){pd::require(count>=3&&p+3<w.size(),4,"spirv_decoration_length");if(w[p+2]==33)pd::require(count==4&&w[p+3]<16,4,"spirv_binding_limit");}
        if(op==5&&count>=3)names[w[p+1]]=spv_text(w.data()+p+2,count-2);
        if(op==6&&count>=4)members[w[p+1]].insert(w[p+2]);
        if(op==71&&count>=4&&(w[p+2]==1||w[p+2]==33||w[p+2]==34))decorations[w[p+1]][w[p+2]]=w[p+3];
        if(op==15&&count>=4&&w[p+1]==5&&spv_text(w.data()+p+3,count-3)==meta.at("entry_point").str())entry=true;
        p+=count;
    }
    pd::require(entry,4,"spirv_entry");
    std::set<uint32_t> claimed_caps;
    for(auto &cap:meta.at("required_caps").arr()) {const auto&s=cap.str();size_t n=0;auto value=std::stoul(s,&n);pd::require(n==s.size()&&value<=UINT32_MAX&&std::to_string(value)==s&&claimed_caps.insert(uint32_t(value)).second,4,"capability_encoding");}
    pd::require(claimed_caps==a.capabilities,4,"spirv_capability_reflection_mismatch");
    auto &local=meta.at("layout_abi").at("local_size").arr();pd::require(local.size()==3,4,"local_size");
    for(int i=0;i<3;++i){pd::require(local[i].num()>0&&uint64_t(local[i].num())<=UINT32_MAX,4,"local_size");a.local[i]=uint32_t(local[i].num());}
    if(profile!=MGC_SHADER_PROFILE_NCNN_COMPAT)return;
    pd::require(meta.at("entry_point").str()=="main",3,"ncnn_entry_target");
    auto &bindings=meta.at("layout_abi").at("descriptor_sets").arr();pd::require(bindings.size()<=16,4,"ncnn_binding_limit");
    std::set<uint32_t> actual,claimed;
    for(auto&[id,d]:decorations)if(d.count(33)){pd::require(d.count(34)&&d.at(34)==0,4,"ncnn_set");actual.insert(d.at(33));}
    for(auto &b:bindings){pd::require(b.number("set")==0&&b.number("count")==1,4,"ncnn_descriptor");claimed.insert(uint32_t(b.at("binding").num()));}
    pd::require(actual==claimed&&claimed.size()==bindings.size(),4,"descriptor_reflection");
    for(uint32_t i=0;i<bindings.size();++i)pd::require(claimed.count(i),4,"ncnn_binding_gap");
    auto &pushes=meta.at("layout_abi").at("push_constants").arr();
    if(!pushes.empty()){
        pd::require(pushes.size()==1,4,"ncnn_push_count");size_t named_count=0;uint32_t id=0;
        for(auto &[i,name]:names)if(name=="parameter"){++named_count;id=i;}
        pd::require(named_count==1&&pushes[0].number("size")==int64_t(members[id].size()*4),4,"ncnn_named_parameter");
    }
    auto &spec=meta.at("layout_abi").at("specialization_constants").arr();std::set<int64_t> ids;
    for(auto &s:spec)ids.insert(s.at("id").num());for(size_t i=0;i<spec.size();++i)pd::require(ids.count(int64_t(i)),4,"ncnn_spec_gap");
}
}
#if MGC_HAVE_NCNN && NCNN_VULKAN
#include "shader_device.inc"
#endif
struct MgcShaderpack {
    std::atomic<uint32_t> refs{1};MgcDomain *domain=nullptr;uint32_t profile=0;
    std::vector<Artifact> artifacts;
#if MGC_HAVE_NCNN && NCNN_VULKAN
    std::map<std::string,std::unique_ptr<ncnn::Pipeline>> pipelines;
#endif
    std::mutex mutex;
    ~MgcShaderpack(){
#if MGC_HAVE_NCNN && NCNN_VULKAN
        if (domain) meguviz::with_ncnn_domain_cleanup(domain, [](void *p, const ncnn::VulkanDevice *) -> MgcStatus {
            static_cast<MgcShaderpack *>(p)->pipelines.clear(); return MGC_SUCCESS;
        }, this);
#endif
        if(domain)mgc_domain_release(domain);
    }
};
namespace {
Artifact *select(MgcShaderpack *pack,const MgcShaderArtifactRequest *input) {
    MgcShaderArtifactRequest r{};pd::require(descriptor(input,r,offsetof(MgcShaderArtifactRequest,reserved)),4,"request_abi");
    pd::require(!r.required_flags&&zero(r.reserved,sizeof(r.reserved))&&r.logical_shader_id&&r.logical_shader_id_length&&r.logical_shader_id_length<=256,4,"request_fields");
    pd::require(!r.reserved_tail&&!r.reserved_tail2,4,"request_tail");
    std::string logical(r.logical_shader_id,r.logical_shader_id_length),variant=hex(r.variant_key),effective=hex(r.effective_options_hash);
    for(auto &a:pack->artifacts)if(a.logical==logical&&a.variant==variant&&a.effective==effective) {
        if(r.effective_options_json_length){pd::require(r.effective_options_json&&r.effective_options_json_length<=65536,4,"options_json_bound");std::string json(r.effective_options_json,r.effective_options_json_length);auto options=pd::Parser(json).parse();pd::require(a.options.object()&&pd::dump(options)==pd::dump(a.options)&&pd::hash(options)==a.effective,4,"options_binding");}
        if(r.target_facts_json_length){pd::require(r.target_facts_json&&r.target_facts_json_length<=1024*1024,4,"facts_json_bound");std::string json(r.target_facts_json,r.target_facts_json_length);auto facts=pd::Parser(json).parse();pd::require(a.facts.object()&&pd::dump(facts)==pd::dump(a.facts)&&pd::hash(facts)==a.facts_hash,4,"facts_binding");}
        if(!zero(r.target_facts_hash.bytes,32))pd::require(a.facts_hash==hex(r.target_facts_hash),4,"facts_hash_binding");
        return &a;
    }
    return nullptr;
}
}
extern "C" {
MgcStatus mgc_shaderpack_open(MgcDomain *domain,const MgcShaderpackDesc *input,MgcShaderpack **out) {
    if(!out||!domain)return MGC_INVALID_ARGUMENT;*out=nullptr;
    MgcShaderpackDesc d{};if(!descriptor(input,d,offsetof(MgcShaderpackDesc,reserved)))return MGC_UNSUPPORTED_ABI;
    if(d.required_flags)return MGC_UNSUPPORTED_ABI;
    if(!zero(d.reserved,sizeof(d.reserved))||!d.root_path_utf8||!d.root_path_length||d.root_path_length>32768)return MGC_INVALID_ARGUMENT;
    if(d.profile!=MGC_SHADER_PROFILE_NCNN_COMPAT&&d.profile!=MGC_SHADER_PROFILE_NATIVE_VULKAN)return MGC_UNSUPPORTED_CAPABILITY;
    try {
        auto pack=std::make_unique<MgcShaderpack>();pack->profile=d.profile;
        std::string path(d.root_path_utf8,d.root_path_length);pd::require(path.find('\0')==std::string::npos&&pd::utf8(path),4,"root_path");
        auto root=std::filesystem::u8path(path);auto m=pd::load(root/"manifest.json");
        pd::keys(m,{"schema_version","abi_version","profile_id","profile_version","required_flags","source_closure_hash","toolchain_lock_hash","artifacts","manifest_hash"});
        pd::require(m.at("schema_version").num()==1&&m.at("abi_version").number("major")==0&&m.at("abi_version").number("minor")>=1,4,"manifest_abi");
        pd::require(m.at("required_flags").arr().empty(),4,"required_flags");
        pd::require(m.at("profile_id").str()==(d.profile==1?"ncnn-compat":"native-vulkan"),4,"profile");
        if(d.profile==1)pd::require(m.at("profile_version").str()=="20261020",4,"ncnn_profile_version");
        const auto claimed=m.at("manifest_hash").str();pd::require(pd::valid_hash(claimed),4,"manifest_digest");m.obj().erase("manifest_hash");
        pd::require(pd::hash(m)==claimed&&(zero(d.manifest_hash.bytes,32)||hex(d.manifest_hash)==claimed),4,"manifest_hash");
        pd::require(pd::valid_hash(m.at("source_closure_hash").str())&&pd::valid_hash(m.at("toolchain_lock_hash").str()),4,"manifest_identity");
        auto &list=m.at("artifacts").arr();pd::require(!list.empty()&&list.size()<=65536,4,"artifact_count");std::set<std::string> variants;uint64_t total=0;
        for(auto &meta:list){
            pd::keys(meta,{"logical_shader_id","variant_key","spirv_path","spirv_sha256","spirv_size","entry_point","target_env","effective_options_hash","layout_abi","required_caps","reflection_path","reflection_sha256","effective_options","target_facts","target_facts_hash"});
            Artifact a;a.logical=meta.at("logical_shader_id").str();a.variant=meta.at("variant_key").str();a.effective=meta.at("effective_options_hash").str();a.digest=meta.at("spirv_sha256").str();
            pd::require(!a.logical.empty()&&a.logical.size()<=256&&pd::valid_hash(a.variant)&&pd::valid_hash(a.effective)&&variants.insert(a.variant).second,4,"artifact_identity");
            auto spv=bytes(root,meta,"spirv_path","spirv_sha256");pd::require(spv.size()>=20&&spv.size()%4==0&&int64_t(spv.size())==meta.at("spirv_size").num(),4,"spirv_size");
            total+=spv.size();pd::require(total<=256*1024*1024,4,"pack_memory_bound");a.words.resize(spv.size()/4);std::memcpy(a.words.data(),spv.data(),spv.size());
            auto reflection_bytes=bytes(root,meta,"reflection_path","reflection_sha256");auto reflection=pd::Parser(reflection_bytes).parse();
            pd::require(pd::dump(reflection.at("layout_abi"))==pd::dump(meta.at("layout_abi"))&&reflection.at("entry_point").str()==meta.at("entry_point").str(),4,"reflection_layout");
            if(meta.has("effective_options")||meta.has("target_facts")||meta.has("target_facts_hash")) {
                pd::require(meta.has("effective_options")&&meta.has("target_facts")&&meta.has("target_facts_hash"),4,"partial_execution_binding");
                a.options=meta.at("effective_options");a.facts=meta.at("target_facts");a.facts_hash=meta.at("target_facts_hash").str();
                pd::require(a.options.object()&&a.facts.object()&&pd::valid_hash(a.facts_hash)&&pd::hash(a.options)==a.effective&&pd::hash(a.facts)==a.facts_hash,4,"execution_binding_hash");
            }
            a.layout=meta.at("layout_abi");interface_check(a,meta,d.profile);pack->artifacts.push_back(std::move(a));
        }
        pack->domain=domain;mgc_domain_retain(domain);*out=pack.release();return MGC_SUCCESS;
    }catch(const std::bad_alloc&){return MGC_OUT_OF_MEMORY;}catch(const pd::Error &e){return e.code==3?MGC_UNSUPPORTED_CAPABILITY:MGC_ARTIFACT_MISMATCH;}catch(...){return MGC_ARTIFACT_MISMATCH;}
}
void mgc_shaderpack_retain(MgcShaderpack *p){if(p)++p->refs;}
void mgc_shaderpack_release(MgcShaderpack *p){if(p&&!--p->refs)delete p;}
MgcStatus mgc_shaderpack_domain_identity(MgcShaderpack *p,uint64_t *id,uint64_t *generation){return p?mgc_domain_identity(p->domain,id,generation):MGC_INVALID_ARGUMENT;}
MgcStatus mgc_shaderpack_select(MgcShaderpack *p,const MgcShaderArtifactRequest *r,MgcShaderArtifactView *out){
    if(!p||!r||!out)return MGC_INVALID_ARGUMENT;
    if(out->header.abi_major!=MGC_ABI_MAJOR||out->header.struct_size<offsetof(MgcShaderArtifactView,reserved))return MGC_UNSUPPORTED_ABI;
    const auto caller_size=out->header.struct_size;
    try{auto*a=select(p,r);if(!a)return MGC_ARTIFACT_MISMATCH;
        MgcShaderArtifactView v{};v.header=MGC_HEADER_INIT(MgcShaderArtifactView);v.header.struct_size=caller_size;v.spirv_words=a->words.data();v.spirv_size_bytes=a->words.size()*4;v.profile=p->profile;
        std::copy(a->local,a->local+3,v.local_size);unhex(a->digest,v.spirv_hash);std::memcpy(out,&v,std::min<size_t>(caller_size,sizeof(v)));return MGC_SUCCESS;
    }catch(...){return MGC_ARTIFACT_MISMATCH;}
}
}
#if MGC_HAVE_NCNN && NCNN_VULKAN
namespace meguviz {
MgcStatus ncnn_shaderpack_pipeline(MgcShaderpack *pack,const MgcShaderArtifactRequest *request,
                                  const std::vector<ncnn::vk_specialization_type> &specializations,ncnn::Pipeline **out){
    (void)pack;(void)request;(void)specializations;if(out)*out=nullptr;
    // This older exact-source entry cannot establish a caller's actual Option.
    // Host selection remains available; execution must use the typed overload.
    return out?MGC_UNSUPPORTED_ABI:MGC_INVALID_ARGUMENT;
}
MgcStatus ncnn_shaderpack_pipeline(MgcShaderpack *pack,const MgcShaderArtifactRequest *request,
                                  const ncnn::Option &effective,
                                  const std::vector<ncnn::vk_specialization_type> &specializations,ncnn::Pipeline **out){
    if(!pack||!out)return MGC_INVALID_ARGUMENT;*out=nullptr;
    if(pack->profile!=MGC_SHADER_PROFILE_NCNN_COMPAT)return MGC_UNSUPPORTED_CAPABILITY;
    try{
        auto*a=select(pack,request);if(!a)return MGC_ARTIFACT_MISMATCH;
        if(specializations.size()!=a->layout.at("specialization_constants").arr().size())return MGC_ARTIFACT_MISMATCH;
        std::string key=a->variant+":"+a->effective+":"+a->digest+":"+a->facts_hash+":"+private_sdk_fingerprint();
        if(!specializations.empty())key.append(reinterpret_cast<const char*>(specializations.data()),specializations.size()*sizeof(ncnn::vk_specialization_type));
        struct Context{MgcShaderpack *pack;Artifact *a;const ncnn::Option *effective;const std::vector<ncnn::vk_specialization_type> *spec;std::string *key;ncnn::Pipeline **out;}ctx{pack,a,&effective,&specializations,&key,out};
        return with_ncnn_domain(pack->domain,[](void *ptr,const ncnn::VulkanDevice *device) -> MgcStatus {
            auto&c=*static_cast<Context*>(ptr);if(!device)return MGC_UNSUPPORTED_CAPABILITY;
            try { device_shader_check(device,*c.a,*c.effective,*c.spec); }
            catch(const pd::Error &e) { return e.code==3?MGC_UNSUPPORTED_CAPABILITY:MGC_ARTIFACT_MISMATCH; }
            std::lock_guard<std::mutex> lock(c.pack->mutex);auto it=c.pack->pipelines.find(*c.key);
            if(it!=c.pack->pipelines.end()){*c.out=it->second.get();return MGC_SUCCESS;}
            ncnn::ShaderInfo info{};if(ncnn::resolve_shader_info(c.a->words.data(),c.a->words.size()*4,info))return MGC_ARTIFACT_MISMATCH;
            if(info.binding_count!=int(c.a->layout.at("descriptor_sets").arr().size()))return MGC_ARTIFACT_MISMATCH;
            auto p=std::make_unique<ncnn::Pipeline>(device);p->set_local_size_xyz(c.a->local[0],c.a->local[1],c.a->local[2]);
            if(p->create(c.a->words.data(),c.a->words.size()*4,*c.spec))return MGC_UNSUPPORTED_CAPABILITY;
            *c.out=p.get();c.pack->pipelines.emplace(*c.key,std::move(p));return MGC_SUCCESS;
        },&ctx);
    }catch(const std::bad_alloc&){return MGC_OUT_OF_MEMORY;}catch(const pd::Error &e){return e.code==3?MGC_UNSUPPORTED_CAPABILITY:MGC_ARTIFACT_MISMATCH;}catch(...){return MGC_ARTIFACT_MISMATCH;}
}
}
#endif
