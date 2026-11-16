/* Included only in Vulkan builds. The feature chain is the exact immutable
 * pNext accepted by vkCreateDevice, after ncnn's driver blacklists. */
namespace {
struct FeatureNode { VkStructureType sType; const void *pNext; };
template<class T> const T *enabled_feature(const ncnn::VulkanDevice *device,VkStructureType type) {
    auto *node=static_cast<const FeatureNode*>(device->enabled_extension_features());
    for(unsigned i=0;node&&i<128;++i,node=static_cast<const FeatureNode*>(node->pNext))
        if(node->sType==type)return reinterpret_cast<const T*>(node);
    return nullptr;
}
struct EnabledShaderDevice {
    const ncnn::VulkanDevice *device;
    const VkPhysicalDeviceFeatures &core;
    const VkPhysicalDevice8BitStorageFeaturesKHR *s8;
    const VkPhysicalDevice16BitStorageFeaturesKHR *s16;
    const VkPhysicalDeviceFloat16Int8FeaturesKHR *arithmetic;
    const VkPhysicalDeviceMaintenance4FeaturesKHR *maintenance;
    const VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR *subgroup_types;
    const VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR *rotate;
    const VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR *dot;
    const VkPhysicalDeviceVulkanMemoryModelFeaturesKHR *memory;
    const VkPhysicalDeviceShaderBfloat16FeaturesKHR *bf16;
    const VkPhysicalDeviceShaderAtomicFloatFeaturesEXT *atomic;
    const VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT *atomic2;
    const VkPhysicalDeviceCooperativeMatrixFeaturesKHR *matrix;
    const VkPhysicalDeviceCooperativeMatrixFeaturesNV *matrix_nv;
    explicit EnabledShaderDevice(const ncnn::VulkanDevice *d):device(d),core(d->enabled_features()),
        s8(enabled_feature<VkPhysicalDevice8BitStorageFeaturesKHR>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR)),
        s16(enabled_feature<VkPhysicalDevice16BitStorageFeaturesKHR>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR)),
        arithmetic(enabled_feature<VkPhysicalDeviceFloat16Int8FeaturesKHR>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR)),
        maintenance(enabled_feature<VkPhysicalDeviceMaintenance4FeaturesKHR>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR)),
        subgroup_types(enabled_feature<VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR)),
        rotate(enabled_feature<VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES_KHR)),
        dot(enabled_feature<VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR)),
        memory(enabled_feature<VkPhysicalDeviceVulkanMemoryModelFeaturesKHR>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR)),
        bf16(enabled_feature<VkPhysicalDeviceShaderBfloat16FeaturesKHR>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR)),
        atomic(enabled_feature<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT)),
        atomic2(enabled_feature<VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT)),
        matrix(enabled_feature<VkPhysicalDeviceCooperativeMatrixFeaturesKHR>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR)),
        matrix_nv(enabled_feature<VkPhysicalDeviceCooperativeMatrixFeaturesNV>(d,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV)) {}
    bool subgroup(VkFlags bit) const {
        const auto &p=device->info.querySubgroupProperties();
        return device->enabled_api_version()>=VK_MAKE_VERSION(1,1,0)&&
            (p.supportedStages&VK_SHADER_STAGE_COMPUTE_BIT)&&(p.supportedOperations&bit)==bit;
    }
    bool float_mode(uint32_t mode,uint32_t width) const {
        if(device->enabled_api_version()<VK_MAKE_VERSION(1,2,0)&&!device->extension_enabled("VK_KHR_shader_float_controls"))return false;
        const auto &p=device->info.queryFloatControlsProperties();
        switch(mode) {
        case 4459:return width==16?p.shaderDenormPreserveFloat16:width==32?p.shaderDenormPreserveFloat32:width==64?p.shaderDenormPreserveFloat64:false;
        case 4460:return width==16?p.shaderDenormFlushToZeroFloat16:width==32?p.shaderDenormFlushToZeroFloat32:width==64?p.shaderDenormFlushToZeroFloat64:false;
        case 4461:return width==16?p.shaderSignedZeroInfNanPreserveFloat16:width==32?p.shaderSignedZeroInfNanPreserveFloat32:width==64?p.shaderSignedZeroInfNanPreserveFloat64:false;
        case 4462:return width==16?p.shaderRoundingModeRTEFloat16:width==32?p.shaderRoundingModeRTEFloat32:width==64?p.shaderRoundingModeRTEFloat64:false;
        case 4463:return width==16?p.shaderRoundingModeRTZFloat16:width==32?p.shaderRoundingModeRTZFloat32:width==64?p.shaderRoundingModeRTZFloat64:false;
        default:return false;
        }
    }
    bool capability(uint32_t cap) const {
        switch(cap) {
        case 0:case 1:return true; // Matrix is core scalar/vector matrix, not cooperative matrix.
        case 8:return s16&&s16->storageBuffer16BitAccess;
        case 9:return arithmetic&&arithmetic->shaderFloat16;
        case 10:return core.shaderFloat64;
        case 11:return core.shaderInt64;
        case 22:return core.shaderInt16;
        case 25:return core.shaderImageGatherExtended;
        case 27:return core.shaderStorageImageMultisample;
        case 28:return core.shaderUniformBufferArrayDynamicIndexing;
        case 29:return core.shaderSampledImageArrayDynamicIndexing;
        case 30:return core.shaderStorageBufferArrayDynamicIndexing;
        case 31:return core.shaderStorageImageArrayDynamicIndexing;
        case 34:case 45:return core.imageCubeArray;
        case 39:return arithmetic&&arithmetic->shaderInt8;
        case 43:case 44:return device->info.max_image_dimension_1d()>0;
        case 46:case 47:case 48:case 50:return true;
        case 49:return core.shaderStorageImageExtendedFormats;
        case 55:return core.shaderStorageImageReadWithoutFormat;
        case 56:return core.shaderStorageImageWriteWithoutFormat;
        case 61:return subgroup(VK_SUBGROUP_FEATURE_BASIC_BIT);
        case 62:return subgroup(VK_SUBGROUP_FEATURE_VOTE_BIT);
        case 63:return subgroup(VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
        case 64:return subgroup(VK_SUBGROUP_FEATURE_BALLOT_BIT);
        case 65:return subgroup(VK_SUBGROUP_FEATURE_SHUFFLE_BIT);
        case 66:return subgroup(VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT);
        case 67:return subgroup(VK_SUBGROUP_FEATURE_CLUSTERED_BIT);
        case 68:return subgroup(VK_SUBGROUP_FEATURE_QUAD_BIT);
        case 4433:return s16&&s16->storageBuffer16BitAccess;
        case 4434:return s16&&s16->uniformAndStorageBuffer16BitAccess;
        case 4435:return s16&&s16->storagePushConstant16;
        case 4436:return s16&&s16->storageInputOutput16;
        case 4448:return s8&&s8->storageBuffer8BitAccess;
        case 4449:return s8&&s8->uniformAndStorageBuffer8BitAccess;
        case 4450:return s8&&s8->storagePushConstant8;
        case 4464:case 4465:case 4466:case 4467:case 4468:
            return float_mode(cap-5,16)||float_mode(cap-5,32)||float_mode(cap-5,64);
        case 5116:return bf16&&bf16->shaderBFloat16Type;
        case 5117:return bf16&&bf16->shaderBFloat16DotProduct;
        case 5118:return bf16&&bf16->shaderBFloat16CooperativeMatrix;
        case 5345:return memory&&memory->vulkanMemoryModel;
        case 5346:return memory&&memory->vulkanMemoryModelDeviceScope;
        case 5357:return matrix_nv&&matrix_nv->cooperativeMatrix&&!device->info.queryCooperativeMatrixSubPropertiesNV().empty();
        case 6022:return matrix&&matrix->cooperativeMatrix&&!device->info.queryCooperativeMatrixSubProperties().empty();
        case 6016:case 6017:case 6018:case 6019:return dot&&dot->shaderIntegerDotProduct;
        case 6026:return rotate&&rotate->shaderSubgroupRotate&&subgroup(VK_SUBGROUP_FEATURE_ROTATE_BIT_KHR);
        case 6029:{auto *f=enabled_feature<VkPhysicalDeviceShaderFloatControls2FeaturesKHR>(device,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR);return f&&f->shaderFloatControls2;}
        case 6033:return atomic&&(atomic->shaderBufferFloat32AtomicAdd||atomic->shaderSharedFloat32AtomicAdd||atomic->shaderImageFloat32AtomicAdd);
        case 6034:return atomic&&(atomic->shaderBufferFloat64AtomicAdd||atomic->shaderSharedFloat64AtomicAdd);
        case 6095:return atomic2&&(atomic2->shaderBufferFloat16AtomicAdd||atomic2->shaderSharedFloat16AtomicAdd);
        case 5612:return atomic2&&(atomic2->shaderBufferFloat32AtomicMinMax||atomic2->shaderSharedFloat32AtomicMinMax||atomic2->shaderImageFloat32AtomicMinMax);
        case 5613:return atomic2&&(atomic2->shaderBufferFloat64AtomicMinMax||atomic2->shaderSharedFloat64AtomicMinMax);
        case 5616:return atomic2&&(atomic2->shaderBufferFloat16AtomicMinMax||atomic2->shaderSharedFloat16AtomicMinMax);
        default:return false; // Never infer an unenabled feature from an extension advertisement.
        }
    }
    bool extension(const std::string &s) const {
        if(s=="SPV_KHR_16bit_storage")return s16;
        if(s=="SPV_KHR_8bit_storage")return s8;
        if(s=="SPV_KHR_storage_buffer_storage_class")return device->enabled_api_version()>=VK_MAKE_VERSION(1,1,0)||device->extension_enabled("VK_KHR_storage_buffer_storage_class");
        if(s=="SPV_KHR_float_controls")return capability(4464);
        if(s=="SPV_KHR_vulkan_memory_model")return capability(5345);
        if(s=="SPV_KHR_cooperative_matrix")return capability(6022);
        if(s=="SPV_NV_cooperative_matrix")return capability(5357);
        if(s=="SPV_KHR_integer_dot_product")return capability(6019);
        if(s=="SPV_KHR_shader_subgroup_rotate")return capability(6026);
        if(s=="SPV_KHR_float_controls2")return capability(6029);
        if(s=="SPV_KHR_bfloat16")return capability(5116);
        if(s=="SPV_EXT_shader_atomic_float_add")return capability(6033)||capability(6034)||capability(6095);
        if(s=="SPV_EXT_shader_atomic_float16_add")return capability(6095);
        if(s=="SPV_EXT_shader_atomic_float_min_max")return capability(5612)||capability(5613)||capability(5616);
        if(s=="SPV_KHR_non_semantic_info")return device->enabled_api_version()>=VK_MAKE_VERSION(1,3,0)||device->extension_enabled("VK_KHR_shader_non_semantic_info");
        if(s=="SPV_KHR_no_integer_wrap_decoration")return true;
        return false;
    }
};
pd::J shader_options(const ncnn::Option &o) {
    auto json=meguviz::ncnn_shader_options_json(o);return pd::Parser(json).parse();
}
pd::J shader_facts(const ncnn::VulkanDevice *device) {
    using pd::J;EnabledShaderDevice enabled(device);const auto &p=device->info.physicalDeviceProperties();
    J::A macros;auto define=[&](const char *key,uint64_t value){macros.push_back(J::O{{"name",key},{"token",std::to_string(value)}});};
    const bool fp16=enabled.s16&&enabled.s16->storageBuffer16BitAccess;
    const bool maintenance=enabled.maintenance&&enabled.maintenance->maintenance4;
    define("storageBuffer16BitAccess",fp16);define("shaderInt64",enabled.core.shaderInt64);define("shaderInt16",enabled.core.shaderInt16);
    define("maintenance4",maintenance);define("apiVersion",p.apiVersion);define("driverVersion",p.driverVersion);define("vendorID",p.vendorID);define("deviceID",p.deviceID);
    bool moltenvk=false;
#if __APPLE__
    moltenvk=true;
#endif
    return J::O{{"provenance","captured"},{"maintenance4_enabled",maintenance},{"support_fp16_storage",fp16},
        {"shader_int64",bool(enabled.core.shaderInt64)},{"shader_int16",bool(enabled.core.shaderInt16)},
        {"moltenvk",moltenvk},{"device_defines",macros}};
}
VkFormat image_format(uint32_t format) {
    // SPIR-V ImageFormat -> Vulkan typed-format mapping (registry enum values).
    static const VkFormat formats[]={VK_FORMAT_UNDEFINED,VK_FORMAT_R32G32B32A32_SFLOAT,VK_FORMAT_R16G16B16A16_SFLOAT,VK_FORMAT_R32_SFLOAT,
        VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_R8G8B8A8_SNORM,VK_FORMAT_R32G32_SFLOAT,VK_FORMAT_R16G16_SFLOAT,VK_FORMAT_B10G11R11_UFLOAT_PACK32,
        VK_FORMAT_R16_SFLOAT,VK_FORMAT_R16G16B16A16_UNORM,VK_FORMAT_A2B10G10R10_UNORM_PACK32,VK_FORMAT_R16G16_UNORM,VK_FORMAT_R8G8_UNORM,
        VK_FORMAT_R16_UNORM,VK_FORMAT_R8_UNORM,VK_FORMAT_R16G16B16A16_SNORM,VK_FORMAT_R16G16_SNORM,VK_FORMAT_R8G8_SNORM,VK_FORMAT_R16_SNORM,
        VK_FORMAT_R8_SNORM,VK_FORMAT_R32G32B32A32_SINT,VK_FORMAT_R16G16B16A16_SINT,VK_FORMAT_R8G8B8A8_SINT,VK_FORMAT_R32_SINT,VK_FORMAT_R32G32_SINT,
        VK_FORMAT_R16G16_SINT,VK_FORMAT_R8G8_SINT,VK_FORMAT_R16_SINT,VK_FORMAT_R8_SINT,VK_FORMAT_R32G32B32A32_UINT,VK_FORMAT_R16G16B16A16_UINT,
        VK_FORMAT_R8G8B8A8_UINT,VK_FORMAT_R32_UINT,VK_FORMAT_A2B10G10R10_UINT_PACK32,VK_FORMAT_R32G32_UINT,VK_FORMAT_R16G16_UINT,VK_FORMAT_R8G8_UINT,
        VK_FORMAT_R16_UINT,VK_FORMAT_R8_UINT};
    return format<sizeof(formats)/sizeof(formats[0])?formats[format]:VK_FORMAT_UNDEFINED;
}
void device_shader_check(const ncnn::VulkanDevice *device,const Artifact &a,const ncnn::Option &options,
                         const std::vector<ncnn::vk_specialization_type> &specializations) {
    pd::require(device&&device->is_valid(),3,"device_invalid");EnabledShaderDevice enabled(device);
    pd::require(a.options.object()&&a.facts.object()&&!a.facts_hash.empty(),3,"artifact_missing_execution_binding");
    const auto actual_options=shader_options(options),actual_facts=shader_facts(device);
    pd::require(pd::dump(a.options)==pd::dump(actual_options)&&pd::hash(actual_options)==a.effective,4,"actual_options_mismatch");
    pd::require(pd::dump(a.facts)==pd::dump(actual_facts)&&pd::hash(actual_facts)==a.facts_hash,4,"actual_device_facts_mismatch");
    auto flag=[&](const char *key){return actual_options.flag(key);};
    pd::require(!flag("use_fp16_storage")||(enabled.s16&&enabled.s16->storageBuffer16BitAccess),3,"fp16_storage_not_enabled");
    pd::require(!flag("use_fp16_uniform")||(enabled.s16&&enabled.s16->uniformAndStorageBuffer16BitAccess),3,"fp16_uniform_not_enabled");
    pd::require(!flag("use_fp16_arithmetic")||(enabled.arithmetic&&enabled.arithmetic->shaderFloat16),3,"fp16_arithmetic_not_enabled");
    pd::require(!flag("use_int8_storage")||(enabled.s8&&enabled.s8->storageBuffer8BitAccess),3,"int8_storage_not_enabled");
    pd::require(!flag("use_int8_uniform")||(enabled.s8&&enabled.s8->uniformAndStorageBuffer8BitAccess),3,"int8_uniform_not_enabled");
    pd::require(!flag("use_int8_arithmetic")||(enabled.arithmetic&&enabled.arithmetic->shaderInt8),3,"int8_arithmetic_not_enabled");
    pd::require(!flag("use_int16_storage")||(enabled.s16&&enabled.s16->storageBuffer16BitAccess),3,"int16_storage_not_enabled");
    pd::require(!flag("use_bf16_storage")||(enabled.s16&&enabled.s16->storageBuffer16BitAccess&&enabled.bf16&&enabled.bf16->shaderBFloat16Type),3,"bf16_storage_not_enabled");
    pd::require(!flag("use_subgroup_ops")||enabled.subgroup(VK_SUBGROUP_FEATURE_BASIC_BIT),3,"subgroups_not_enabled");
    pd::require(!flag("use_cooperative_matrix")||(enabled.capability(6022)||enabled.capability(5357)),3,"cooperative_matrix_not_enabled");
    const bool maintenance=enabled.maintenance&&enabled.maintenance->maintenance4;
    const std::string expected=maintenance?"vulkan1.3":flag("use_subgroup_ops")||flag("use_cooperative_matrix")?"vulkan1.1":"vulkan1.0";
    pd::require(a.target==expected,4,"effective_target_mismatch");
    uint32_t minimum=a.target=="vulkan1.3"?VK_MAKE_VERSION(1,3,0):a.target=="vulkan1.1"?VK_MAKE_VERSION(1,1,0):VK_MAKE_VERSION(1,0,0);
    pd::require(device->enabled_api_version()>=minimum,3,"vulkan_target_not_enabled");
    for(uint32_t cap:a.capabilities)pd::require(enabled.capability(cap),3,"spirv_capability_not_enabled:"+std::to_string(cap));
    const auto &limits=device->info.physicalDeviceProperties().limits;
    uint64_t invocations=1;for(unsigned i=0;i<3;++i){pd::require(a.local[i]&&a.local[i]<=limits.maxComputeWorkGroupSize[i],3,"workgroup_axis_limit");pd::require(invocations<=uint64_t(limits.maxComputeWorkGroupInvocations)/a.local[i],3,"workgroup_product_limit");invocations*=a.local[i];}
    pd::require(invocations<=limits.maxComputeWorkGroupInvocations,3,"workgroup_invocation_limit");
    uint32_t buffers=0,storage_images=0,sampled_images=0;
    for(auto &b:a.layout.at("descriptor_sets").arr()) {auto type=b.at("type").num();if(type==7)++buffers;else if(type==3)++storage_images;else if(type==1)++sampled_images;else pd::require(false,4,"ncnn_descriptor_type");}
    pd::require(buffers<=limits.maxPerStageDescriptorStorageBuffers&&storage_images<=limits.maxPerStageDescriptorStorageImages&&
        sampled_images<=limits.maxPerStageDescriptorSampledImages&&sampled_images<=limits.maxPerStageDescriptorSamplers&&
        buffers+storage_images+sampled_images<=limits.maxPerStageResources,3,"descriptor_device_limits");
    for(auto &p:a.layout.at("push_constants").arr())pd::require(p.number("offset")==0&&uint64_t(p.at("size").num())<=limits.maxPushConstantsSize,3,"push_constant_limit");
    const auto &w=a.words;
    std::map<uint32_t,std::vector<uint32_t>> types;
    std::map<uint32_t,uint64_t> constants;
    std::map<uint32_t,uint32_t> specids,value_types;
    for(size_t p=5;p<w.size();) {
        uint32_t count=w[p]>>16,op=w[p]&65535;
        if(op>=20&&op<=32&&count>=2)types[w[p+1]]=std::vector<uint32_t>(w.begin()+p,w.begin()+p+count);
        if((op==43||op==50)&&count>=4)constants[w[p+2]]=w[p+3];
        if(op==71&&count==4&&w[p+2]==1)specids[w[p+1]]=w[p+3];
        if(op>=41&&count>=3&&w[p+1]<w[3]&&types.count(w[p+1]))value_types[w[p+2]]=w[p+1];
        if(op==10)pd::require(enabled.extension(spv_text(w.data()+p+1,count-1)),3,"spirv_extension_not_enabled");
        if(op==25&&count>=9&&w[p+8]) {
            VkFormat format=image_format(w[p+8]);pd::require(format!=VK_FORMAT_UNDEFINED,3,"image_format_unsupported");
            VkFormatProperties properties{};ncnn::vkGetPhysicalDeviceFormatProperties(device->info.physicalDevice(),format,&properties);
            VkFormatFeatureFlags required=w[p+7]==2?VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT:VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            pd::require((properties.optimalTilingFeatures&required)==required,3,"typed_image_format_not_supported");
        }
        p+=count;
    }
    for(auto &[id,spec]:specids)if(spec<specializations.size()){uint32_t bits=0;std::memcpy(&bits,&specializations[spec],4);constants[id]=bits;}
    auto width=[&](uint32_t id){auto found=types.find(id);if(found==types.end())return 0u;const auto&t=found->second;if((t[0]&65535)==23){found=types.find(t[2]);if(found==types.end())return 0u;return found->second.size()>2?found->second[2]:0u;}return t.size()>2?t[2]:0u;};
    using TypeLayout=std::pair<uint64_t,uint64_t>; // bytes, natural Workgroup alignment
    auto aligned=[](uint64_t size,uint64_t alignment){pd::require(alignment&&size<=UINT64_MAX-(alignment-1),4,"workgroup_alignment_overflow");return ((size+alignment-1)/alignment)*alignment;};
    std::function<TypeLayout(uint32_t,unsigned)> type_layout=[&](uint32_t id,unsigned depth)->TypeLayout {
        pd::require(depth<32,4,"spirv_type_cycle");auto found=types.find(id);pd::require(found!=types.end(),4,"spirv_type_missing");const auto&t=found->second;uint32_t op=t[0]&65535;
        if((op==21||op==22)&&t.size()>=3){pd::require(t[2]==8||t[2]==16||t[2]==32||t[2]==64,4,"workgroup_scalar_width");uint64_t bytes=t[2]/8;return {bytes,bytes};}
        if(op==20)return {4,4};
        if((op==23||op==24)&&t.size()>=4){auto element=type_layout(t[2],depth+1);pd::require(t[3]>=2&&t[3]<=4,4,"workgroup_vector_dimension");if(op==23)return {element.first*t[3],element.second*(t[3]==3?4:t[3])};return {aligned(element.first,element.second)*t[3],element.second};}
        if(op==28&&t.size()>=4){pd::require(constants.count(t[3]),3,"dynamic_workgroup_array");auto element=type_layout(t[2],depth+1);uint64_t stride=aligned(element.first,element.second);pd::require(!stride||constants[t[3]]<=UINT64_MAX/stride,4,"workgroup_array_overflow");return {stride*constants[t[3]],element.second};}
        if(op==30){uint64_t size=0,alignment=1;for(size_t i=2;i<t.size();++i){auto element=type_layout(t[i],depth+1);size=aligned(size,element.second);pd::require(size<=UINT64_MAX-element.first,4,"workgroup_struct_overflow");size+=element.first;alignment=std::max(alignment,element.second);}return {aligned(size,alignment),alignment};}
        pd::require(false,3,"unsupported_workgroup_type");return {0,1};
    };
    uint64_t shared_bytes=0;
    for(size_t p=5;p<w.size();) {
        uint32_t count=w[p]>>16,op=w[p]&65535;
        if(op==59&&count>=4&&w[p+3]==4){auto t=types.find(w[p+1]);pd::require(t!=types.end()&&t->second.size()>=4,4,"workgroup_pointer");auto layout=type_layout(t->second[3],0);shared_bytes=aligned(shared_bytes,layout.second);pd::require(shared_bytes<=UINT64_MAX-layout.first,4,"workgroup_sum_overflow");shared_bytes+=layout.first;}
        if(op>=333&&op<=366&&count>=3) {uint32_t bits=width(w[p+1]);if(bits&&bits!=32)pd::require(enabled.subgroup_types&&enabled.subgroup_types->shaderSubgroupExtendedTypes,3,"subgroup_extended_type_not_enabled");}
        if(op==331&&count>=6&&w[p+2]==38){pd::require(maintenance,3,"local_size_id_requires_maintenance4");for(unsigned i=0;i<3;++i)pd::require(constants.count(w[p+3+i])&&constants[w[p+3+i]]==a.local[i],4,"local_size_id_reflection_mismatch");}
        if(op==16&&count>=6&&w[p+2]==17)for(unsigned i=0;i<3;++i)pd::require(w[p+3+i]==a.local[i],4,"local_size_reflection_mismatch");
        if(op==16&&count>=4&&w[p+2]>=4459&&w[p+2]<=4463)pd::require(enabled.float_mode(w[p+2],w[p+3]),3,"float_control_width_not_supported");
        if((op==6035||op==5614||op==5615)&&count>=4) {
            auto v=value_types.find(w[p+3]);pd::require(v!=value_types.end(),4,"atomic_pointer_type");auto pt=types.find(v->second);pd::require(pt!=types.end()&&pt->second.size()>=4,4,"atomic_pointer");
            uint32_t storage=pt->second[2],bits=width(pt->second[3]);bool shared=storage==4,image=storage==11,buffer=storage==2||storage==12;bool allowed=false;
            if(op==6035&&enabled.atomic){if(bits==32)allowed=buffer?enabled.atomic->shaderBufferFloat32AtomicAdd:shared?enabled.atomic->shaderSharedFloat32AtomicAdd:image?enabled.atomic->shaderImageFloat32AtomicAdd:false;if(bits==64)allowed=buffer?enabled.atomic->shaderBufferFloat64AtomicAdd:shared?enabled.atomic->shaderSharedFloat64AtomicAdd:false;}
            if(enabled.atomic2){if(op==6035&&bits==16)allowed=buffer?enabled.atomic2->shaderBufferFloat16AtomicAdd:shared?enabled.atomic2->shaderSharedFloat16AtomicAdd:false;if(op!=6035){if(bits==16)allowed=buffer?enabled.atomic2->shaderBufferFloat16AtomicMinMax:shared?enabled.atomic2->shaderSharedFloat16AtomicMinMax:false;if(bits==32)allowed=buffer?enabled.atomic2->shaderBufferFloat32AtomicMinMax:shared?enabled.atomic2->shaderSharedFloat32AtomicMinMax:image?enabled.atomic2->shaderImageFloat32AtomicMinMax:false;if(bits==64)allowed=buffer?enabled.atomic2->shaderBufferFloat64AtomicMinMax:shared?enabled.atomic2->shaderSharedFloat64AtomicMinMax:false;}}
            pd::require(allowed,3,"atomic_storage_feature_not_enabled");
        }
        p+=count;
    }
    pd::require(shared_bytes<=limits.maxComputeSharedMemorySize,3,"workgroup_shared_memory_limit");
}
} // namespace
namespace meguviz {
MgcStatus ncnn_shader_target_facts(const ncnn::VulkanDevice *device,std::string &json) {
    if(!device||!device->is_valid())return MGC_UNSUPPORTED_CAPABILITY;
    try{json=pd::dump(shader_facts(device));return MGC_SUCCESS;}catch(const std::bad_alloc&){return MGC_OUT_OF_MEMORY;}catch(...){return MGC_INTERNAL_ERROR;}
}
}
