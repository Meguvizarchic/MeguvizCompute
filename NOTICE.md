# Third-party provenance

MeguvizCompute 1.2.1 development snapshot. ncnn source is locked to Tencent/ncnn 7e36d25cb153092ba87a60530455c7376fedf452 (tag 20261020), BSD-3-Clause. Vulkan builds retain nihui/glslang 780b6a209132dc74e99458f62ebe5fa7e57c41cf and its bundled license notices. The installed ncnn package carries its original notices. No RIFE models, converted research weights or model downloaders are included.

The optional Android static OpenMP profile links the Android NDK LLVM OpenMP runtime, under Apache-2.0 with LLVM exceptions and applicable legacy notices. Its exact ABI-specific archive and bundled LLVM NOTICE are identified in `third_party/locks/ncnn.lock.json`; the complete NDK LLVM NOTICE is included with that profile. Other builds default to OpenMP OFF.
