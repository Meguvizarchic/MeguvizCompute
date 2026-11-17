#include <meguviz/compute.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %d %s\n", __LINE__, #x); return 1; } } while (0)
#define INIT(v,T) T v = {0}; v.header = (MgcHeader)MGC_HEADER_INIT(T)

struct backing { unsigned char data[64]; int refs, retained, released; };
static void retain(void *value) { struct backing *b = value; ++b->refs; ++b->retained; }
static void release(void *value) { struct backing *b = value; --b->refs; ++b->released; }

int main(void) {
    INIT(rd, MgcRuntimeDesc); rd.provider = MGC_PROVIDER_CPU_REFERENCE; rd.memory_budget_bytes = 128;
    MgcRuntime *runtime = NULL; CHECK(mgc_runtime_create(&rd, &runtime) == MGC_SUCCESS);
    INIT(dd, MgcDomainDesc); MgcDomain *domain = NULL; CHECK(mgc_domain_acquire(runtime, &dd, &domain) == MGC_SUCCESS);
    INIT(cap, MgcCapability); CHECK(mgc_domain_capability(domain, MGC_CAP_EXTERNAL_HOST_MEMORY, &cap) == MGC_SUCCESS);
    CHECK(cap.enabled && cap.advertised && cap.workload_qualified);
    struct backing b = {{0}, 1, 0, 0};
    INIT(desc, MgcResourceDesc); desc.capacity_bytes = 64; desc.format = MGC_FORMAT_BYTES;
    INIT(host, MgcHostMemoryDesc); host.data = b.data; host.capacity_bytes = 64; host.access = MGC_ACCESS_READ;
    host.context = &b; host.retain = retain; host.release = release;
    MgcResource *resource = (MgcResource *)1;
    host.capacity_bytes = 63;
    CHECK(mgc_resource_import_host_v1(domain, &desc, &host, &resource) == MGC_INVALID_ARGUMENT && !resource);
    CHECK(b.refs == 1 && b.retained == 1 && b.released == 1);
    host.capacity_bytes = 64;
    CHECK(mgc_resource_import_host_v1(domain, &desc, &host, &resource) == MGC_SUCCESS && resource);
    CHECK(b.refs == 2 && b.retained == 2);
    INIT(ld, MgcLeaseDesc); ld.content_id = 1; ld.byte_offset = 4; ld.byte_length = 8; ld.access = MGC_ACCESS_WRITE;
    MgcLease *lease = (MgcLease *)1;
    CHECK(mgc_lease_create(resource, &ld, &lease) == MGC_UNSUPPORTED_CAPABILITY && !lease);
    ld.access = MGC_ACCESS_READ;
    CHECK(mgc_lease_create(resource, &ld, &lease) == MGC_SUCCESS);
    mgc_resource_release(resource); CHECK(b.refs == 2);
    void *address = NULL; uint64_t bytes = 0;
    CHECK(mgc_lease_map(lease, &address, &bytes) == MGC_SUCCESS && address == b.data + 4 && bytes == 8);
    CHECK(mgc_lease_unmap(lease) == MGC_SUCCESS);
    mgc_lease_release(lease); CHECK(b.refs == 1 && b.released == 2);
    host.access = MGC_ACCESS_WRITE;
    CHECK(mgc_resource_import_host_v1(domain, &desc, &host, &resource) == MGC_SUCCESS);
    ld.access = MGC_ACCESS_WRITE; ld.content_id = 2;
    CHECK(mgc_lease_create(resource, &ld, &lease) == MGC_SUCCESS);
    CHECK(mgc_lease_map(lease, &address, &bytes) == MGC_SUCCESS);
    memset(address, 0x4e, (size_t)bytes);
    CHECK(mgc_lease_unmap(lease) == MGC_SUCCESS && b.data[4] == 0x4e && b.data[11] == 0x4e);
    mgc_resource_release(resource); mgc_lease_release(lease);
    CHECK(b.refs == 1 && b.retained == 3 && b.released == 3);
    mgc_domain_release(domain); CHECK(mgc_runtime_close(runtime) == MGC_SUCCESS); mgc_runtime_release(runtime);
    puts("PASS external host resource read/write and rollback");
    return 0;
}
