#include "consumer_common.h"
#include <meguviz/kernels.h>
#include <math.h>
static int run(MgcSession *s,uint32_t operation,MgcLease *input,MgcLease *output,uint64_t bytes)
{
    MgcSubmitDesc d={0};d.header=(MgcHeader)MGC_HEADER_INIT(MgcSubmitDesc);
    d.operation=operation;d.input=input;d.output=output;d.byte_count=bytes;
    mgc_lease_retain(input);mgc_lease_retain(output);
    MgcTicket *t=0;MgcStatus status=mgc_submit(s,&d,&t);
    if(status!=MGC_ACCEPTED){mgc_lease_release(input);mgc_lease_release(output);return 1;}
    if(mgc_session_drain(s,UINT64_MAX))return 1;
    MgcCompletion c={0};c.header=(MgcHeader)MGC_HEADER_INIT(MgcCompletion);
    MgcExecutionInfo trace={0};trace.header=(MgcHeader)MGC_HEADER_INIT(MgcExecutionInfo);
    int failed=mgc_ticket_query(t,&c)||c.status||!c.execution_retired||!c.host_finalized||mgc_ticket_execution_info(t,&trace)||
        !trace.submitted_batches||trace.submitted_batches!=trace.retired_batches||trace.submitted_batches!=trace.finalized_batches||!trace.final_submit_async||trace.full_record_async;
    printf("operation=%u submitted=%llu recording=%llu finalized=%llu\n",operation,
        (unsigned long long)trace.submitted_batches,(unsigned long long)trace.recording_submitted_batches,(unsigned long long)trace.finalized_batches);
    mgc_ticket_release(t);return failed;
}
int main(void)
{
    MgcRuntime*r=0;MgcDomain*d=0;MgcSession*s=0;
    MgcStatus start=start_runtime_provider(&r,&d,&s,MGC_PROVIDER_NCNN_VULKAN);
    if(start==MGC_UNSUPPORTED_CAPABILITY){puts("SKIP: Vulkan device unavailable");if(r)mgc_runtime_release(r);return 77;}
    CHECK(start==MGC_SUCCESS);
    MgcResource *resources[5]={0};MgcLease *leases[5]={0};
    for(int i=0;i<5;++i){MgcResourceDesc rd={0};rd.header=(MgcHeader)MGC_HEADER_INIT(MgcResourceDesc);
        rd.capacity_bytes=20;rd.format=MGC_FORMAT_FLOAT32;rd.usage=i>0&&i<4?MGC_RESOURCE_GPU_BUFFER:0;
        CHECK(mgc_resource_create(d,&rd,&resources[i])==0);CHECK(make_lease(resources[i],1,20,MGC_ACCESS_WRITE,&leases[i])==0);}
    void*ptr=0;uint64_t bytes=0;CHECK(mgc_lease_map(leases[0],&ptr,&bytes)==0);
    const float values[5]={-2.f,-1.f,0.f,2.f,4.f};memcpy(ptr,values,sizeof(values));CHECK(mgc_lease_unmap(leases[0])==0);
    MgcLease *read=0;CHECK(mgc_lease_publish_read(leases[0],&read)==0);mgc_lease_release(read);
    CHECK(run(s,MGC_OPERATION_COPY,leases[0],leases[1],20)==0);
    CHECK(mgc_lease_publish_read(leases[1],&read)==0);mgc_lease_release(read);
    CHECK(run(s,MGC_OPERATION_RELU_F32,leases[1],leases[2],20)==0);
    CHECK(mgc_lease_publish_read(leases[2],&read)==0);mgc_lease_release(read);
    CHECK(run(s,MGC_OPERATION_NORMALIZE_F32,leases[2],leases[3],20)==0);
    CHECK(mgc_lease_publish_read(leases[3],&read)==0);mgc_lease_release(read);
    CHECK(run(s,MGC_OPERATION_COPY,leases[3],leases[4],20)==0);
    CHECK(mgc_lease_map(leases[4],&ptr,&bytes)==0);const float expected[5]={0.f,0.f,0.f,.5f,1.f};
    for(int i=0;i<5;++i)CHECK(fabsf(((float*)ptr)[i]-expected[i])<1e-6f);CHECK(mgc_lease_unmap(leases[4])==0);
    MgcResource *sum_input=0,*sum_output=0;MgcLease *sum_write=0,*sum_read=0,*sum_result=0;
    const float cancellation[3]={1e10f,1.f,-1e10f};
    CHECK(make_resource(d,0,sizeof(cancellation),MGC_FORMAT_FLOAT32,&sum_input)==0);
    CHECK(make_resource(d,0,sizeof(MgcStatisticsF32),MGC_FORMAT_BYTES,&sum_output)==0);
    CHECK(make_lease(sum_input,1,sizeof(cancellation),MGC_ACCESS_WRITE,&sum_write)==0);
    CHECK(mgc_lease_map(sum_write,&ptr,&bytes)==0);memcpy(ptr,cancellation,sizeof(cancellation));CHECK(mgc_lease_unmap(sum_write)==0);
    CHECK(mgc_lease_publish_read(sum_write,&sum_read)==0);mgc_lease_release(sum_write);
    CHECK(make_lease(sum_output,1,sizeof(MgcStatisticsF32),MGC_ACCESS_WRITE,&sum_result)==0);
    CHECK(run(s,MGC_OPERATION_STATISTICS_F32,sum_read,sum_result,sizeof(cancellation))==0);
    CHECK(mgc_lease_map(sum_result,&ptr,&bytes)==0);MgcStatisticsF32 sum;memcpy(&sum,ptr,sizeof(sum));
    CHECK(sum.finite_count==3&&sum.sum==1.0);CHECK(mgc_lease_unmap(sum_result)==0);
    mgc_lease_release(sum_read);mgc_lease_release(sum_result);mgc_resource_release(sum_input);mgc_resource_release(sum_output);
    CHECK(mgc_session_close(s)==0);
    for(int i=0;i<5;++i){mgc_lease_release(leases[i]);mgc_resource_release(resources[i]);}
    mgc_domain_release(d);CHECK(mgc_runtime_close(r)==0);mgc_runtime_release(r);
    puts("PASS: resident Vulkan ReLU -> normalization -> oracle; no intermediate full-frame host transfer");return 0;
}
