ncnn::Pipeline *kernel_pipeline(MgcDomain *d, const char *source)
{
    auto found = d->kernel_pipelines.find(source);
    if (found != d->kernel_pipelines.end()) return found->second.get();
    auto p = std::make_unique<ncnn::Pipeline>(d->device);
    // Keep ncnn's default 1x1x1 dispatch. Some drivers execute a different
    // width from the injected local size; shaders use workgroup IDs and one
    // invocation per group so dispatch remains correct on either behavior.
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(source, static_cast<int>(std::strlen(source)), gpu_options(d), spirv) || p->create(spirv.data(), spirv.size() * 4, {})) return nullptr;
    auto *borrowed = p.get(); d->kernel_pipelines.emplace(source, std::move(p)); return borrowed;
}
static const char *reduce_f32_shader = R"glsl(#version 450
layout(local_size_x=1,local_size_y=1,local_size_z=1) in;
layout(binding=0) readonly buffer Input { float values[]; } a;
layout(binding=1) writeonly buffer Output { vec4 values[]; } b;
layout(push_constant) uniform parameter { uint count; uint groups; } p;
void main(){if(gl_LocalInvocationIndex!=0u)return;for(uint g=gl_WorkGroupID.x;g<p.groups;g+=gl_NumWorkGroups.x){
float lo=uintBitsToFloat(0x7f800000u),hi=-lo,c=0.0;
precise float sum=0.0,correction=0.0;
for(uint i=g*256u;i<min(p.count,(g+1u)*256u);++i){float x=a.values[i];if(!isnan(x)&&!isinf(x)){lo=min(lo,x);hi=max(hi,x);c+=1.0;}}
float magnitude=max(abs(lo),abs(hi));float scale=magnitude>1e36?magnitude:1.0;
if(c>0.0)for(uint i=g*256u;i<min(p.count,(g+1u)*256u);++i){float x=a.values[i];if(!isnan(x)&&!isinf(x)){precise float v=x/scale;precise float t=sum+v;if(abs(sum)>=abs(v))correction+=(sum-t)+v;else correction+=(v-t)+sum;sum=t;}}
b.values[2u*g]=vec4(lo,hi,sum,c);b.values[2u*g+1u]=vec4(correction,0.0,0.0,0.0);}})glsl";
static const char *normalize_f32_shader = R"glsl(#version 450
layout(local_size_x=1,local_size_y=1,local_size_z=1) in;
layout(binding=0) readonly buffer Input { float values[]; } a;
layout(binding=1) writeonly buffer Output { float values[]; } b;
layout(push_constant) uniform parameter { uint count; float low; float high; } p;
void main(){if(gl_LocalInvocationIndex!=0u)return;for(uint i=gl_WorkGroupID.x;i<p.count;i+=gl_NumWorkGroups.x){float s=max(abs(p.low),abs(p.high))>8e37?0.5:1.0;float lo=p.low*s,hi=p.high*s,range=hi-lo;b.values[i]=range==0.0?0.0:(a.values[i]*s-lo)/range;}})glsl";
static const char *relu_f32_shader = R"glsl(#version 450
layout(local_size_x=1,local_size_y=1,local_size_z=1) in;
layout(binding=0) readonly buffer Input { float values[]; } a;
layout(binding=1) writeonly buffer Output { float values[]; } b;
layout(push_constant) uniform parameter { uint count; } p;
void main(){if(gl_LocalInvocationIndex!=0u)return;for(uint i=gl_WorkGroupID.x;i<p.count;i+=gl_NumWorkGroups.x){float x=a.values[i];b.values[i]=x>0.0?x:0.0;}})glsl";
static const char *resize_rgb_shader = R"glsl(#version 450
#extension GL_ARB_gpu_shader_fp64 : require
layout(local_size_x=1,local_size_y=1,local_size_z=1) in;
layout(binding=0) readonly buffer Input { uint values[]; } a;
layout(binding=1) buffer Output { uint values[]; } b;
layout(push_constant) uniform parameter { uint sw;uint sh;uint ss;uint dw;uint dh;uint ds;uint words; } p;
double at(int x,int y,uint c){uint n=uint(clamp(y,0,int(p.sh)-1))*p.ss+uint(clamp(x,0,int(p.sw)-1))*3u+c;return double((a.values[n/4u]>>((n%4u)*8u))&255u);}
void main(){if(gl_LocalInvocationIndex!=0u)return;for(uint word=gl_WorkGroupID.x;word<p.words;word+=gl_NumWorkGroups.x){uint result=b.values[word];
for(uint k=0u;k<4u;++k){uint offset=word*4u+k;if(offset>=p.ds*p.dh)continue;uint y=offset/p.ds,xbyte=offset%p.ds;if(xbyte>=p.dw*3u)continue;
uint x=xbyte/3u,c=xbyte%3u;
precise double fx=(double(x)+0.5lf)*double(p.sw)/double(p.dw)-0.5lf;
precise double fy=(double(y)+0.5lf)*double(p.sh)/double(p.dh)-0.5lf;
int ix=int(floor(fx)),iy=int(floor(fy));precise double ax=fx-double(ix),ay=fy-double(iy);
precise double top=at(ix,iy,c)*(1.0lf-ax)+at(ix+1,iy,c)*ax;
precise double bottom=at(ix,iy+1,c)*(1.0lf-ax)+at(ix+1,iy+1,c)*ax;
precise double v=top*(1.0lf-ay)+bottom*ay;
result=(result&~(255u<<(k*8u)))|(uint(floor(v+0.5lf))<<(k*8u));}b.values[word]=result;}})glsl";
