#include <meguviz/compute.h>
#include <meguviz/kernels.h>
#include "statistics_sha256.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {
using mgc_example::sha256;
constexpr size_t max_input = 2 * 1024 * 1024, max_values = 65536;
void check(bool value, const char *why) { if (!value) throw std::runtime_error(why); }
void status(MgcStatus value, MgcStatus expected, const char *why) { if (value != expected) throw std::runtime_error(std::string(why) + ":" + std::to_string(value)); }
struct Grid { size_t width = 0, height = 0; std::vector<float> values; };
struct Parser {
    const std::string &s; size_t p = 0;
    void ws() { while (p < s.size() && (s[p]==' ' || s[p]=='\t' || s[p]=='\r' || s[p]=='\n')) ++p; }
    bool consume(char c) { ws(); if (p<s.size() && s[p]==c) { ++p; return true; } return false; }
    void need(char c) { check(consume(c), "invalid_json_separator"); }
    std::string key() { ws(); check(p<s.size() && s[p++]=='"', "invalid_json_key"); size_t start=p;
        while(p<s.size() && ((s[p]>='a'&&s[p]<='z') || s[p]=='_')) ++p;
        check(p>start && p<s.size() && s[p]=='"', "invalid_json_key"); auto result=s.substr(start,p-start); ++p; return result; }
    std::string number() { ws(); size_t start=p; if(p<s.size() && s[p]=='-') ++p;
        check(p<s.size() && s[p]>='0' && s[p]<='9', "invalid_number");
        if(s[p]=='0') ++p; else while(p<s.size() && s[p]>='0' && s[p]<='9') ++p;
        if(p<s.size() && s[p]=='.') { ++p; size_t first=p; while(p<s.size()&&s[p]>='0'&&s[p]<='9') ++p; check(p>first,"invalid_number"); }
        if(p<s.size() && (s[p]=='e'||s[p]=='E')) { ++p; if(p<s.size()&&(s[p]=='+'||s[p]=='-')) ++p; size_t first=p;
            while(p<s.size()&&s[p]>='0'&&s[p]<='9') ++p; check(p>first,"invalid_number"); }
        check(p-start<=128,"number_too_long"); return s.substr(start,p-start); }
    size_t integer() { auto token=number(); unsigned long long n=0; auto r=std::from_chars(token.data(),token.data()+token.size(),n);
        check(r.ec==std::errc{}&&r.ptr==token.data()+token.size()&&n>0&&n<=max_values,"invalid_integer"); return size_t(n); }
    float value() { auto token=number(); double d=0; auto r=std::from_chars(token.data(),token.data()+token.size(),d,std::chars_format::general);
        check(r.ec==std::errc{}&&r.ptr==token.data()+token.size()&&std::isfinite(d),"invalid_float");
        float f=static_cast<float>(d); check(std::isfinite(f),"float_overflow"); return f; }
    Grid parse() { Grid g; std::set<std::string> seen; need('{'); bool first=true;
        while(!consume('}')) { if(!first) need(','); first=false; auto k=key(); check(seen.insert(k).second,"duplicate_field"); need(':');
            if(k=="schema_version") check(integer()==1,"schema_version");
            else if(k=="width") g.width=integer(); else if(k=="height") g.height=integer();
            else if(k=="values") { need('['); bool fv=true; while(!consume(']')) { if(!fv) need(','); fv=false;
                check(g.values.size()<max_values,"too_many_values"); g.values.push_back(value()); } }
            else throw std::runtime_error("unknown_field"); }
        ws(); check(p==s.size(),"trailing_input"); check(seen.size()==4 && seen.count("values") && g.width && g.height &&
            g.width<=max_values/g.height && g.width*g.height==g.values.size(),"grid_shape_mismatch"); return g; }
};
struct State {
    MgcRuntime *runtime=nullptr; MgcDomain *domain=nullptr; MgcSession *session=nullptr;
    MgcResource *input=nullptr,*stats=nullptr,*output=nullptr; MgcLease *read=nullptr,*stat_write=nullptr,*out_write=nullptr;
    MgcTicket *ticket=nullptr;
    ~State() { if(session) mgc_session_drain(session, 10000000000ULL); if(ticket) mgc_ticket_release(ticket);
        if(read) mgc_lease_release(read); if(stat_write) mgc_lease_release(stat_write); if(out_write) mgc_lease_release(out_write);
        if(input) mgc_resource_release(input); if(stats) mgc_resource_release(stats); if(output) mgc_resource_release(output);
        if(session) mgc_session_close(session); if(domain) mgc_domain_release(domain);
        if(runtime) { mgc_runtime_close(runtime); mgc_runtime_release(runtime); } }
};
MgcResource *resource(State &s, uint64_t id, uint64_t size, uint32_t format) { MgcResourceDesc d{}; d.header=MGC_HEADER_INIT(MgcResourceDesc);
    d.allocation_id=id; d.allocation_generation=1; d.capacity_bytes=size; d.format=format; MgcResource *v=nullptr;
    status(mgc_resource_create(s.domain,&d,&v),MGC_SUCCESS,"resource_create"); return v; }
MgcLease *lease(MgcResource *r, uint64_t size, uint32_t access) { MgcLeaseDesc d{}; d.header=MGC_HEADER_INIT(MgcLeaseDesc);
    d.content_id=1; d.byte_length=size; d.access=access; MgcLease *v=nullptr; status(mgc_lease_create(r,&d,&v),MGC_SUCCESS,"lease_create"); return v; }
void *map(MgcLease *l, uint64_t expected) { void *p=nullptr; uint64_t size=0; status(mgc_lease_map(l,&p,&size),MGC_SUCCESS,"map");
    if(!p||size<expected) { mgc_lease_unmap(l); throw std::runtime_error("map_size"); } return p; }
void submit(State &s, uint32_t op, MgcLease *out, uint64_t bytes) { MgcSubmitDesc d{}; d.header=MGC_HEADER_INIT(MgcSubmitDesc);
    d.input=s.read; d.output=out; d.operation=op; d.byte_count=bytes; mgc_lease_retain(s.read); mgc_lease_retain(out);
    MgcStatus result=mgc_submit(s.session,&d,&s.ticket); if(result!=MGC_ACCEPTED) { mgc_lease_release(s.read); mgc_lease_release(out); status(result,MGC_ACCEPTED,"submit"); }
    status(mgc_session_drain(s.session,10000000000ULL),MGC_SUCCESS,"drain"); MgcCompletion c{}; c.header=MGC_HEADER_INIT(MgcCompletion);
    status(mgc_ticket_query(s.ticket,&c),MGC_SUCCESS,"ticket_query"); check(c.terminal==MGC_TERMINAL_SUCCESS && c.status==MGC_SUCCESS && c.execution_retired && c.host_finalized,"job_failed");
    mgc_ticket_release(s.ticket); s.ticket=nullptr; }
void write_exclusive(const std::filesystem::path &path, const std::string &text) {
#ifdef _WIN32
    int fd=_wopen(path.c_str(),_O_WRONLY|_O_CREAT|_O_EXCL|_O_BINARY,_S_IREAD|_S_IWRITE);
#else
    int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
#endif
    check(fd>=0,"report_exists_or_create_failed"); size_t off=0; bool ok=true;
    while(off<text.size()) {
#ifdef _WIN32
        int n=_write(fd,text.data()+off,static_cast<unsigned int>(std::min<size_t>(text.size()-off,1u<<20)));
#else
        auto n=write(fd,text.data()+off,text.size()-off);
#endif
        if(n<=0) { ok=false; break; } off+=size_t(n); }
#ifdef _WIN32
    if(_close(fd)!=0) ok=false;
#else
    if(close(fd)!=0) ok=false;
#endif
    check(ok,"report_write_failed_partial_file_retained");
}
}
extern "C" int mgc_statistics_cli(int argc, char **argv) {
    try {
        std::string provider,fixture,report; std::set<std::string> args;
        for(int i=1;i<argc;i+=2) { check(i+1<argc,"missing_argument_value"); std::string key=argv[i];
            check(args.insert(key).second,"duplicate_argument"); if(key=="--provider") provider=argv[i+1];
            else if(key=="--fixture") fixture=argv[i+1]; else if(key=="--report") report=argv[i+1]; else throw std::runtime_error("unknown_argument"); }
        check(args.size()==3 && !fixture.empty() && !report.empty(),"provider_fixture_report_required");
        uint32_t selected=provider=="cpu-reference"?MGC_PROVIDER_CPU_REFERENCE:provider=="ncnn-cpu"?MGC_PROVIDER_NCNN_CPU:0;
        check(selected!=0,"unsupported_provider"); check(std::numeric_limits<float>::is_iec559 && sizeof(float)==4,"float32_required");
        auto path=std::filesystem::u8path(fixture); auto fs=std::filesystem::symlink_status(path);
        check(std::filesystem::is_regular_file(fs) && !std::filesystem::is_symlink(fs),"fixture_not_regular");
        std::ifstream input(path,std::ios::binary); check(bool(input),"fixture_open"); std::string raw(max_input+1,'\0');
        input.read(raw.data(),static_cast<std::streamsize>(raw.size())); auto n=input.gcount(); check(!input.bad() && n>0 && size_t(n)<=max_input,"fixture_size"); raw.resize(size_t(n));
        auto grid=Parser{raw}.parse(); uint64_t bytes=grid.values.size()*sizeof(float); State s;
        MgcRuntimeDesc rd{}; rd.header=MGC_HEADER_INIT(MgcRuntimeDesc); rd.provider=selected; rd.memory_budget_bytes=16*1024*1024;
        status(mgc_runtime_create(&rd,&s.runtime),MGC_SUCCESS,"runtime_create"); MgcDomainDesc dd{};dd.header=MGC_HEADER_INIT(MgcDomainDesc);
        status(mgc_domain_acquire(s.runtime,&dd,&s.domain),MGC_SUCCESS,"domain_acquire"); MgcSessionDesc sd{};sd.header=MGC_HEADER_INIT(MgcSessionDesc);sd.max_in_flight=1;
        status(mgc_session_create(s.domain,&sd,&s.session),MGC_SUCCESS,"session_create");
        s.input=resource(s,11,bytes,MGC_FORMAT_FLOAT32);s.stats=resource(s,12,sizeof(MgcStatisticsF32),MGC_FORMAT_BYTES);s.output=resource(s,13,bytes,MGC_FORMAT_FLOAT32);
        { auto w=lease(s.input,bytes,MGC_ACCESS_WRITE); try { std::memcpy(map(w,bytes),grid.values.data(),size_t(bytes)); status(mgc_lease_unmap(w),MGC_SUCCESS,"unmap_input"); }
          catch(...) { mgc_lease_release(w); throw; } mgc_lease_release(w); }
        s.read=lease(s.input,bytes,MGC_ACCESS_READ);s.stat_write=lease(s.stats,sizeof(MgcStatisticsF32),MGC_ACCESS_WRITE);s.out_write=lease(s.output,bytes,MGC_ACCESS_WRITE);
        submit(s,MGC_OPERATION_STATISTICS_F32,s.stat_write,bytes);MgcStatisticsF32 stats{};std::memcpy(&stats,map(s.stat_write,sizeof(stats)),sizeof(stats));status(mgc_lease_unmap(s.stat_write),MGC_SUCCESS,"unmap_stats");
        check(stats.count==grid.values.size() && stats.finite_count==stats.count && std::isfinite(stats.sum),"invalid_statistics");
        submit(s,MGC_OPERATION_NORMALIZE_F32,s.out_write,bytes);std::vector<float> normalized(grid.values.size());std::memcpy(normalized.data(),map(s.out_write,bytes),size_t(bytes));status(mgc_lease_unmap(s.out_write),MGC_SUCCESS,"unmap_output");
        float expected_min=grid.values.front(),expected_max=expected_min; double expected_sum=0;
        for(float v:grid.values) { expected_min=std::min(expected_min,v); expected_max=std::max(expected_max,v); expected_sum+=v; }
        check(stats.minimum==expected_min && stats.maximum==expected_max && stats.sum==expected_sum,"statistics_oracle_mismatch");
        double range=double(expected_max)-expected_min;
        for(size_t i=0;i<normalized.size();++i) { float expected=range?float((double(grid.values[i])-expected_min)/range):0.f;
            check(std::isfinite(normalized[i]) && std::fabs(double(normalized[i])-expected)<=1e-6,"normalization_oracle_mismatch"); }
        const uint32_t endian_probe=1;const char *byte_order=*reinterpret_cast<const unsigned char*>(&endian_probe)?"little":"big";
        std::ostringstream o;o.imbue(std::locale::classic());o<<std::setprecision(std::numeric_limits<double>::max_digits10);
        o<<"{\"schema_version\":1,\"status\":\"PASS\",\"scope\":\"CPU_GRID_CONSUMER_ONLY\",\"runtime_provider\":\""<<provider
         <<"\",\"kernel_provider\":\"cpu-reference\",\"float32_byte_order\":\""<<byte_order<<"\",\"width\":"<<grid.width<<",\"height\":"<<grid.height
         <<",\"fixture_sha256\":\""<<sha256(raw)<<"\",\"input_float32_native_sha256\":\""<<sha256(std::string(reinterpret_cast<const char*>(grid.values.data()),size_t(bytes)))
         <<"\",\"output_float32_native_sha256\":\""<<sha256(std::string(reinterpret_cast<const char*>(normalized.data()),size_t(bytes)))
         <<"\",\"count\":"<<stats.count<<",\"finite_count\":"<<stats.finite_count<<",\"min\":"<<stats.minimum<<",\"max\":"<<stats.maximum<<",\"sum\":"<<stats.sum<<",\"normalized\":[";
        for(size_t i=0;i<normalized.size();++i) { if(i) o<<','; o<<normalized[i]; } o<<"]}\n";
        write_exclusive(std::filesystem::u8path(report),o.str()); std::cout<<"PASS CPU grid report written\n"; return 0;
    } catch(const std::exception &e) { std::cerr<<"statistics_cli: "<<e.what()<<'\n'; return 1; }
}
