#pragma once
#include <map>
#include <string>

/* Pure header contract. The legacy engine may copy this file without linking
 * Compute or introducing another ncnn owner. Only actual Option members enter
 * canonical JSON; allocator pointers and reserved bytes are never serialized. */
namespace meguviz_options_detail {
template<class Option> inline std::map<std::string,std::string> shader_fields(const Option &o) {
    auto value=[](bool flag){return flag?"true":"false";};
    return {{"use_bf16_storage",value(o.use_bf16_storage)},{"use_bf16_packed",value(o.use_bf16_packed)},
        {"use_fp16_storage",value(o.use_fp16_storage)},{"use_fp16_packed",value(o.use_fp16_packed)},
        {"use_fp16_uniform",value(o.use_fp16_uniform)},{"use_fp16_arithmetic",value(o.use_fp16_arithmetic)},
        {"use_int8_storage",value(o.use_int8_storage)},{"use_int8_packed",value(o.use_int8_packed)},
        {"use_int8_uniform",value(o.use_int8_uniform)},{"use_int8_arithmetic",value(o.use_int8_arithmetic)},
        {"use_int16_storage",value(o.use_int16_storage)},{"use_int16_packed",value(o.use_int16_packed)},
        {"use_shader_local_memory",value(o.use_shader_local_memory)},{"use_subgroup_ops",value(o.use_subgroup_ops)},
        {"use_cooperative_matrix",value(o.use_cooperative_matrix)}};
}
inline std::string canonical(const std::map<std::string,std::string> &fields) {
    std::string result="{";bool first=true;
    for(const auto &entry:fields){if(!first)result+=',';first=false;result+='"';result+=entry.first;result+="\":";result+=entry.second;}
    result+='}';return result;
}
template<class Option> inline std::string shader_options_json(const Option &o) {return canonical(shader_fields(o));}
template<class Option> inline std::string execution_options_json(const Option &o) {
    auto fields=shader_fields(o);
    fields["num_threads"]=std::to_string(o.num_threads);
    fields["use_vulkan_compute"]=o.use_vulkan_compute?"true":"false";
    fields["use_packing_layout"]=o.use_packing_layout?"true":"false";
    fields["use_int8_inference"]=o.use_int8_inference?"true":"false";
    return canonical(fields);
}
} // namespace meguviz_options_detail
