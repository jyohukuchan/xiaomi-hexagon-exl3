#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <rpcmem.h>
#include <remote.h>
#include "gpu_iface.h"
#include "exl3_codec.h"
#include "gpu-pmu.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cl_int status, const char * operation) {
    if (status != CL_SUCCESS) throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}

struct OpenCL {
    OpenCL(const OpenCL &) = delete;
    OpenCL & operator=(const OpenCL &) = delete;
    void * library = nullptr;
#define CL_FUNCTIONS(X) \
    X(clGetPlatformIDs) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext) \
    X(clCreateCommandQueue) X(clCreateBuffer) X(clReleaseMemObject) X(clReleaseCommandQueue) \
    X(clReleaseContext) X(clCreateProgramWithSource) X(clBuildProgram) X(clGetProgramBuildInfo) \
    X(clCreateKernel) X(clSetKernelArg) X(clEnqueueNDRangeKernel) X(clFinish) X(clFlush) \
    X(clEnqueueReadBuffer) X(clEnqueueWriteBuffer) X(clEnqueueMapBuffer) X(clEnqueueUnmapMemObject) \
    X(clGetEventProfilingInfo) X(clReleaseEvent) X(clReleaseKernel) X(clReleaseProgram)
#define DECLARE_FUNCTION(name) decltype(&name) name = nullptr;
    CL_FUNCTIONS(DECLARE_FUNCTION)
#undef DECLARE_FUNCTION
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    std::string info(cl_device_info key) {
        size_t size = 0;
        check(clGetDeviceInfo(device, key, 0, nullptr, &size), "device info size");
        std::string result(size, '\0');
        check(clGetDeviceInfo(device, key, size, result.data(), nullptr), "device info");
        if (!result.empty() && result.back() == '\0') result.pop_back();
        return result;
    }
    OpenCL() {
        library = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
#define LOAD_FUNCTION(name) name = reinterpret_cast<decltype(name)>(dlsym(library, #name)); if (!name) throw std::runtime_error("Missing " #name);
        CL_FUNCTIONS(LOAD_FUNCTION)
#undef LOAD_FUNCTION
        cl_uint count = 0;
        check(clGetPlatformIDs(0, nullptr, &count), "platform count");
        std::vector<cl_platform_id> platforms(count);
        check(clGetPlatformIDs(count, platforms.data(), nullptr), "platforms");
        for (auto platform : platforms) {
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) == CL_SUCCESS) break;
        }
        if (!device) throw std::runtime_error("No OpenCL GPU");
        cl_int status;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        check(status, "create context");
        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
        check(status, "create queue");
    }
    ~OpenCL() {
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
        if (library) dlclose(library);
    }
};

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

struct Buffer {
    OpenCL & cl;
    void * memory = nullptr;
    cl_mem object = nullptr;
    int fd = -1;
    size_t bytes;
    Buffer(OpenCL & api, size_t size, bool shared = true) : cl(api), bytes(size) {
        cl_int status;
        if (shared) {
            cl_uint padding = 0, page = 0;
            check(cl.clGetDeviceInfo(cl.device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(padding), &padding, nullptr), "padding");
            check(cl.clGetDeviceInfo(cl.device, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(page), &page, nullptr), "page");
            if (bytes > 64 * 1024 * 1024) throw std::runtime_error("Probe buffer too large");
            memory = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_CACHED, int(bytes + padding + page));
            if (!memory) throw std::runtime_error("rpcmem_alloc failed");
            fd = rpcmem_to_fd(memory);
            if (fd < 0) throw std::runtime_error("rpcmem_to_fd failed");
            cl_mem_ion_host_ptr external{{CL_MEM_ION_HOST_PTR_QCOM, CL_MEM_HOST_WRITEBACK_QCOM}, fd, memory};
            object = cl.clCreateBuffer(cl.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                                      bytes, &external, &status);
        } else object = cl.clCreateBuffer(cl.context, CL_MEM_READ_WRITE, bytes, nullptr, &status);
        check(status, "buffer creation/import");
    }
    Buffer(const Buffer &) = delete;
    Buffer & operator=(const Buffer &) = delete;
    ~Buffer() {
        if (object) cl.clReleaseMemObject(object);
        if (memory) rpcmem_free(memory);
    }
    void ownership(cl_map_flags flags) {
        cl_int status;
        void * mapped = cl.clEnqueueMapBuffer(cl.queue, object, CL_TRUE, flags, 0, bytes, 0, nullptr, nullptr, &status);
        check(status, "map ownership");
        if (memory && mapped != memory) throw std::runtime_error("External map returned a different host address");
        check(cl.clEnqueueUnmapMemObject(cl.queue, object, mapped, 0, nullptr, nullptr), "unmap ownership");
        check(cl.clFinish(cl.queue), "ownership finish");
    }
    void write(const void * data) {
        check(cl.clEnqueueWriteBuffer(cl.queue, object, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr), "write buffer");
    }
    void read(void * data) {
        check(cl.clEnqueueReadBuffer(cl.queue, object, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr), "read buffer");
    }
};

struct Program {
    Program(const Program &) = delete;
    Program & operator=(const Program &) = delete;
    OpenCL & cl;
    cl_program object;
    Program(OpenCL & api, const char * path) : cl(api) {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("Cannot read OpenCL source");
        const std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        const char * text = source.c_str();
        cl_int status;
        object = cl.clCreateProgramWithSource(cl.context, 1, &text, nullptr, &status);
        check(status, "create program");
        status = cl.clBuildProgram(object, 1, &cl.device, "-cl-std=CL1.2", nullptr, nullptr);
        if (status) {
            size_t size;
            cl.clGetProgramBuildInfo(object, cl.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            std::string log(size, '\0');
            cl.clGetProgramBuildInfo(object, cl.device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
            throw std::runtime_error("OpenCL compile failed: " + log);
        }
    }
    ~Program() { cl.clReleaseProgram(object); }
};

struct Kernel {
    Kernel(const Kernel &) = delete;
    Kernel & operator=(const Kernel &) = delete;
    OpenCL & cl;
    cl_kernel object;
    Kernel(Program & program, const char * name) : cl(program.cl) {
        cl_int status;
        object = cl.clCreateKernel(program.object, name, &status);
        check(status, name);
    }
    ~Kernel() { cl.clReleaseKernel(object); }
    template<class T> void arg(unsigned index, const T & value) {
        check(cl.clSetKernelArg(object, index, sizeof(value), &value), "kernel arg");
    }
    double run(size_t count) {
        const size_t local = 128, global = ((count + local - 1) / local) * local;
        cl_event event;
        check(cl.clEnqueueNDRangeKernel(cl.queue, object, 1, nullptr, &global, &local, 0, nullptr, &event), "enqueue kernel");
        check(cl.clFinish(cl.queue), "kernel finish");
        cl_ulong begin = 0, end = 0;
        check(cl.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(begin), &begin, nullptr), "event start");
        check(cl.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr), "event end");
        cl.clReleaseEvent(event);
        if (end <= begin) throw std::runtime_error("Invalid GPU timestamps");
        return double(end - begin) / 1000;
    }
};

struct DSP {
    DSP(const DSP &) = delete;
    DSP & operator=(const DSP &) = delete;
    remote_handle64 handle = 0;
    DSP() {
        remote_rpc_control_unsigned_module control{};
        control.domain = 3; control.enable = 1;
        check(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &control, sizeof(control)), "unsigned session");
        check(gpu_iface_open("file:///libexl3_gpu_htp.so?gpu_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp", &handle), "DSP open");
    }
    ~DSP() { if (handle) gpu_iface_close(handle); }
    double transfer(Buffer & input, Buffer & output, unsigned mode, unsigned seed, uint64 & ticks, uint64 & cycles) {
        uint32 mismatches = 0;
        auto start = Clock::now();
        check(gpu_iface_transfer(handle, mode, seed, static_cast<unsigned char *>(input.memory), int(input.bytes),
            static_cast<unsigned char *>(output.memory), int(output.bytes), &cycles, &ticks, &mismatches), "DSP transfer");
        const double us = elapsed(start);
        if (mismatches) throw std::runtime_error("DSP input pattern mismatches: " + std::to_string(mismatches));
        return us;
    }
};

static double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) throw std::runtime_error("No samples");
    std::sort(values.begin(), values.end());
    return values[size_t(fraction * double(values.size() - 1))];
}

static void pmu_probe(OpenCL & cl, Program & program, unsigned repeats, unsigned event_set) {
    if (event_set >= GPU_PMU_EVENT_SETS || repeats > 101) throw std::runtime_error("Invalid PMU settings");
    DSP dsp;
    Kernel fill(program, "fill_pattern");
    const char * source_names[] = {"cpu", "gpu"};
    const char * mode_names[] = {"empty", "hvx", "scalar", "hvx_noinv"};
    const char * phase_names[] = {"arrival", "cold", "warm", "outer"};
    std::cout << "pmu_schema version=" << GPU_PMU_VERSION << " report_bytes=" << sizeof(gpu_pmu_report)
              << " set=" << event_set;
    for (unsigned i = 0; i < 8; ++i) std::cout << " e" << i << '=' << gpu_pmu_events[event_set][i];
    std::cout << '\n';
    for (size_t bytes : {size_t(128), size_t(4096), size_t(32768), size_t(262144), size_t(1048576), size_t(8388608), size_t(33554432)}) {
        Buffer input(cl, bytes), output(cl, sizeof(gpu_pmu_report));
        const cl_uint count = cl_uint(bytes / 4);
        fill.arg(0, input.object); fill.arg(1, count);
        for (unsigned mode = 0; mode < 4; ++mode) {
            std::vector<double> times[2][4], counters[2][4][8];
            for (unsigned iteration = 0; iteration < repeats + 3; ++iteration) {
                // CPU/GPU, GPU/CPU ordering alternates to reduce slow drift bias.
                for (unsigned order = 0; order < 2; ++order) {
                    const unsigned source = order ^ (iteration & 1u);
                    for (unsigned attempt = 0; ; ++attempt) {
                    const cl_uint seed = 0x13bac907u + iteration * 71317u + source * 1793u + mode * 997u + attempt * 1031u;
                    check(gpu_iface_pmu_begin(dsp.handle, event_set), "PMU begin");
                    if (source) {
                        fill.arg(2, seed); fill.run(count);
                        input.ownership(CL_MAP_READ);
                    } else {
                        cl_int status;
                        void * ptr = cl.clEnqueueMapBuffer(cl.queue, input.object, CL_TRUE, CL_MAP_WRITE,
                            0, bytes, 0, nullptr, nullptr, &status);
                        check(status, "CPU producer map");
                        if (ptr != input.memory) throw std::runtime_error("CPU producer map address changed");
                        auto * words = static_cast<uint32_t *>(ptr);
                        for (unsigned i = 0; i < count; ++i) words[i] = i * 0x9e3779b9u + seed;
                        check(cl.clEnqueueUnmapMemObject(cl.queue, input.object, ptr, 0, nullptr, nullptr), "CPU producer unmap");
                        check(cl.clFinish(cl.queue), "CPU producer finish");
                    }
                    check(gpu_iface_pmu_read(dsp.handle, mode, seed, static_cast<const unsigned char *>(input.memory),
                        int(bytes), static_cast<unsigned char *>(output.memory), int(output.bytes)), "PMU read");
                    gpu_pmu_report report;
                    output.ownership(CL_MAP_READ);
                    output.read(&report);
                    if (report.version != GPU_PMU_VERSION || report.bytes != bytes || report.mode != mode ||
                        report.event_set != event_set || report.mismatches ||
                        std::memcmp(report.events, gpu_pmu_events[event_set], sizeof(report.events)))
                        throw std::runtime_error("Invalid PMU report or stale GPU/CPU data");
                    uint32_t cfg = 0, low = 0, high = 0;
                    for (unsigned i = 0; i < 8; ++i) {
                        const unsigned event = gpu_pmu_events[event_set][i];
                        cfg |= ((event >> 8) & 3u) << (i * 2);
                        if (i < 4) low |= (event & 255u) << (i * 8);
                        else high |= (event & 255u) << ((i - 4) * 8);
                    }
                    if (report.invalid_flags) {
                        std::cout << "pmu_rejected set=" << event_set << " src=" << source_names[source] << " mode=" << mode_names[mode]
                                  << " bytes=" << bytes << " index=" << int(iteration) - 3 << " attempt=" << attempt
                                  << " flags=" << report.invalid_flags << " requested=" << cfg << ',' << low << ',' << high
                                  << " observed=" << report.configured[0] << ',' << report.configured[1] << ',' << report.configured[2]
                                  << " final=" << report.final_config[0] << ',' << report.final_config[1] << ',' << report.final_config[2] << '\n';
                        if (attempt >= 2) throw std::runtime_error("PMU configuration repeatedly changed; measurement unavailable");
                        continue;
                    }
                    if ((report.configured[0] & 65535u) != cfg || report.configured[1] != low || report.configured[2] != high ||
                        (report.final_config[0] & 65535u) != cfg || report.final_config[1] != low || report.final_config[2] != high)
                        throw std::runtime_error("Unflagged PMU configuration mismatch");
                    uint32_t expected[32]{};
                    if (mode) for (unsigned i = 0; i < count; ++i) expected[i & 31] ^= i * 0x9e3779b9u + seed;
                    for (unsigned pass = 0; pass < 2; ++pass)
                        if (std::memcmp(report.checksum[pass], expected, sizeof(expected)))
                            throw std::runtime_error("PMU cold/warm checksum mismatch");
                    if (iteration >= 3) for (unsigned phase = 0; phase < 4; ++phase) {
                        const double us = double(report.ticks[phase]) / 19.2;
                        // A no-work window can be shorter than one 19.2 MHz tick.
                        if (!report.cycles[phase]) throw std::runtime_error("Invalid PMU cycle timing");
                        times[source][phase].push_back(us);
                        std::cout << "pmu_sample set=" << event_set << " src=" << source_names[source] << " mode=" << mode_names[mode]
                                  << " bytes=" << bytes << " index=" << iteration - 3 << " phase=" << phase_names[phase]
                                  << " usec=" << us << " cycles=" << report.cycles[phase];
                        for (unsigned i = 0; i < 8; ++i) {
                            counters[source][phase][i].push_back(report.counters[phase][i]);
                            std::cout << " c" << i << '=' << report.counters[phase][i];
                        }
                        std::cout << " mismatches=0\n";
                    }
                    break;
                    }
                }
            }
            for (unsigned source = 0; source < 2; ++source) for (unsigned phase = 0; phase < 4; ++phase) {
                std::cout << "pmu_summary set=" << event_set << " src=" << source_names[source] << " mode=" << mode_names[mode]
                          << " bytes=" << bytes << " samples=" << repeats << " phase=" << phase_names[phase]
                          << " p50_us=" << percentile(times[source][phase], 0.5) << " p95_us=" << percentile(times[source][phase], 0.95);
                for (unsigned i = 0; i < 8; ++i) std::cout << " c" << i << '=' << percentile(counters[source][phase][i], 0.5);
                std::cout << " mismatches=0\n";
            }
        }
    }
}

static void interop(OpenCL & cl, Program & program, bool map_sync, unsigned repeats) {
    DSP dsp;
    Kernel fill(program, "fill_pattern"), validate(program, "check_pattern");
    Buffer errors(cl, 4, false);
    validate.arg(1, errors.object);
    validate.arg(4, cl_uint(0xa5c39e17));
    for (size_t bytes : {size_t(128), size_t(4096), size_t(32768), size_t(262144), size_t(1048576), size_t(8388608), size_t(33554432)}) {
        Buffer a(cl, bytes), b(cl, bytes), reduction(cl, 128);
        const cl_uint count = cl_uint(bytes / 4);
        fill.arg(0, a.object); fill.arg(1, count);
        validate.arg(0, b.object); validate.arg(2, count);
        std::vector<double> total, gpu, sync, rpc, copy, noop, read_rpc, read_us;
        uint64 ticks = 0, cycles = 0;
        for (unsigned iteration = 0; iteration < repeats + 3; ++iteration) {
            const cl_uint seed = 0x38ef91a7u + iteration * 71317;
            fill.arg(2, seed); validate.arg(3, seed);
            const auto begin = Clock::now();
            const double gpu_us = fill.run(count);
            const auto sync_start = Clock::now();
            if (map_sync) a.ownership(CL_MAP_READ);
            const double sync_us = elapsed(sync_start);
            const double rpc_us = dsp.transfer(a, b, 1, seed, ticks, cycles);
            const double total_us = elapsed(begin);
            const double copy_us = double(ticks) / 19.2;
            // Synchronize external DSP writes before GPU validation. No CPU payload read or copy.
            if (map_sync) b.ownership(CL_MAP_READ | CL_MAP_WRITE);
            cl_uint mismatch = 0;
            errors.write(&mismatch);
            validate.run(count);
            errors.read(&mismatch);
            if (mismatch) throw std::runtime_error("GPU round-trip mismatches: " + std::to_string(mismatch));
            const double read_rpc_us = dsp.transfer(a, reduction, 3, seed, ticks, cycles);
            const double read_kernel_us = double(ticks) / 19.2;
            uint32_t actual_reduction[32], expected_reduction[32]{};
            reduction.ownership(CL_MAP_READ);
            reduction.read(actual_reduction);
            for (unsigned i = 0; i < count; ++i) expected_reduction[i % 32] ^= i * 0x9e3779b9u + seed;
            if (std::memcmp(actual_reduction, expected_reduction, 128)) throw std::runtime_error("DSP read reduction mismatch");
            // Independent DSP-side full input check, outside the timed transfer sample.
            if (iteration == 0 || iteration == repeats + 2) dsp.transfer(a, b, 2, seed, ticks, cycles);
            const double noop_us = dsp.transfer(a, b, 0, seed, ticks, cycles);
            if (iteration >= 3) {
                total.push_back(total_us); gpu.push_back(gpu_us); sync.push_back(sync_us);
                rpc.push_back(rpc_us); copy.push_back(copy_us); noop.push_back(noop_us);
                read_rpc.push_back(read_rpc_us); read_us.push_back(read_kernel_us);
                std::cout << "interop_sample map=" << map_sync << " bytes=" << bytes << " index=" << iteration - 3
                          << " total_us=" << total_us << " gpu_us=" << gpu_us << " sync_us=" << sync_us
                          << " rpc_us=" << rpc_us << " hvx_copy_us=" << copy_us << " noop_rpc_us=" << noop_us
                          << " read_rpc_us=" << read_rpc_us << " hvx_read_us=" << read_kernel_us << " mismatches=0\n";
            }
        }
        const double median = percentile(total, 0.5);
        std::cout << "interop_summary map=" << map_sync << " bytes=" << bytes << " samples=" << repeats
                  << " total_p50_us=" << median << " total_p95_us=" << percentile(total, 0.95)
                  << " gpu_p50_us=" << percentile(gpu, 0.5) << " sync_p50_us=" << percentile(sync, 0.5)
                  << " rpc_p50_us=" << percentile(rpc, 0.5) << " hvx_copy_p50_us=" << percentile(copy, 0.5)
                  << " noop_rpc_p50_us=" << percentile(noop, 0.5) << " payload_GBps=" << double(bytes) / median / 1000
                  << " read_rpc_p50_us=" << percentile(read_rpc, 0.5) << " hvx_read_p50_us=" << percentile(read_us, 0.5)
                  << " read_effective_GBps=" << double(bytes) / percentile(read_rpc, 0.5) / 1000
                  << " hvx_read_GBps=" << double(bytes) / percentile(read_us, 0.5) / 1000
                  << " mismatches=0\n";
    }
}

static std::vector<uint16_t> hmx_layout(const std::vector<uint16_t> & row_major, unsigned k, unsigned n) {
    std::vector<uint16_t> result(row_major.size());
    for (unsigned r = 0; r < k; ++r) for (unsigned c = 0; c < n; ++c) {
        const size_t group = (c / 128) * (k / 128) + r / 128;
        const unsigned rr = r % 128, cc = c % 128;
        const size_t destination = group * 16384 + ((cc / 32) * 4 + rr / 32) * 1024 +
                                   ((rr % 32) / 2) * 64 + (cc % 32) * 2 + rr % 2;
        result[destination] = row_major[size_t(r) * n + c];
    }
    return result;
}

static void equal(const std::vector<uint16_t> & actual, const std::vector<uint16_t> & expected, const char * stage) {
    size_t mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) {
        if (mismatches++ < 4) std::cerr << stage << " mismatch=" << i << " actual=" << actual[i] << " expected=" << expected[i] << '\n';
    }
    if (mismatches) throw std::runtime_error(std::string(stage) + " mismatches: " + std::to_string(mismatches));
}

static void decode_case(OpenCL & cl, Program & program, DSP & dsp, unsigned k, unsigned n, unsigned bits,
                        unsigned cb, const std::vector<uint16_t> & packed, const std::vector<uint16_t> & expected,
                        const char * label, unsigned repeats) {
    Buffer input(cl, packed.size() * 2), output(cl, expected.size() * 2), dsp_output(cl, expected.size() * 2);
    input.write(packed.data());
    for (cl_uint layout : {0u, 1u, 2u}) {
        if (layout == 2 && (cb != 2 || (bits != 4 && bits != 6 && bits != 8))) continue;
        std::vector<uint16_t> native;
        if (layout == 2) {
            const size_t group_words = 1024 * bits + 256;
            native.resize(size_t(k / 128) * (n / 128) * group_words);
            for (unsigned nb = 0; nb < n / 128; ++nb) for (unsigned kb = 0; kb < k / 128; ++kb)
                for (unsigned r = 0; r < 8; ++r) for (unsigned c = 0; c < 8; ++c) {
                    const size_t src = ((kb * 8 + r) * (n / 16) + nb * 8 + c) * 16 * bits;
                    const size_t dst = (nb * (k / 128) + kb) * group_words + (r * 8 + c) * 16 * bits;
                    std::copy_n(packed.data() + src, 16 * bits, native.data() + dst);
                }
        }
        Buffer native_input(cl, layout == 2 ? native.size() * 2 : 128);
        const std::string name = layout == 2 ? "dequant_hmx_" + std::to_string(bits) : "dequant_inner";
        Kernel selected(program, name.c_str());
        if (layout == 2) {
            native_input.write(native.data());
            selected.arg(0, native_input.object); selected.arg(1, output.object); selected.arg(2, cl_uint(expected.size()));
        } else {
            selected.arg(0, input.object); selected.arg(1, output.object); selected.arg(2, cl_uint(k)); selected.arg(3, cl_uint(n));
            selected.arg(4, cl_uint(bits)); selected.arg(5, cl_uint(cb)); selected.arg(6, layout);
        }
        const auto reference = layout ? hmx_layout(expected, k, n) : expected;
        std::vector<double> gpu, total, rpc;
        for (unsigned i = 0; i < repeats + 3; ++i) {
            const auto start = Clock::now();
            const double gpu_us = selected.run(expected.size());
            output.ownership(CL_MAP_READ);
            uint64 ticks, cycles;
            const double rpc_us = dsp.transfer(output, dsp_output, 1, 0, ticks, cycles);
            const double total_us = elapsed(start);
            if (i >= 3) { gpu.push_back(gpu_us); total.push_back(total_us); rpc.push_back(rpc_us); }
        }
        std::vector<uint16_t> actual(expected.size());
        output.read(actual.data());
        equal(actual, reference, "GPU dequant");
        dsp_output.ownership(CL_MAP_READ);
        dsp_output.read(actual.data());
        // DSP XOR operates on each uint32, so undo the corresponding two half masks.
        for (size_t i = 0; i < actual.size(); ++i) actual[i] ^= i & 1 ? 0xa5c3 : 0x9e17;
        equal(actual, reference, "GPU to DSP dequant round trip");
        std::cout << "dequant_summary label=" << label << " k=" << k << " n=" << n << " bits=" << bits << " cb=" << cb
                  << " layout=" << layout << " samples=" << repeats << " gpu_p50_us=" << percentile(gpu, 0.5)
                  << " gpu_p95_us=" << percentile(gpu, 0.95) << " gpu_to_dsp_p50_us=" << percentile(total, 0.5)
                  << " rpc_p50_us=" << percentile(rpc, 0.5) << " output_GBps=" << double(expected.size() * 2) / percentile(gpu, 0.5) / 1000
                  << " mismatches=0\n";
    }
}

static void dequant(OpenCL & cl, Program & program, unsigned repeats, const std::vector<std::string> & fixtures) {
    Kernel codebook(program, "codebook_all");
    Buffer values(cl, 65536 * 2);
    codebook.arg(0, values.object);
    for (cl_uint cb = 0; cb < 3; ++cb) {
        codebook.arg(1, cb); codebook.run(65536);
        std::vector<uint16_t> actual(65536), expected(65536);
        values.read(actual.data());
        for (unsigned i = 0; i < 65536; ++i) expected[i] = exl3::decode_codebook(uint16_t(i), static_cast<exl3::Codebook>(cb));
        equal(actual, expected, "Codebook");
        std::cout << "codebook_test cb=" << cb << " states=65536 mismatches=0\n";
    }
    DSP dsp;
    uint32_t random = 0x7ba4139d;
    for (unsigned bits = 1; bits <= 8; ++bits) for (unsigned cb = 0; cb < 3; ++cb) {
        const unsigned k = 256, n = 384;
        std::vector<uint16_t> packed(k * n * bits / 16), expected(k * n);
        for (auto & value : packed) {
            random ^= random << 13; random ^= random >> 17; random ^= random << 5; value = uint16_t(random);
        }
        exl3::decode_inner(packed.data(), k, n, bits, static_cast<exl3::Codebook>(cb), expected.data());
        decode_case(cl, program, dsp, k, n, bits, cb, packed, expected, "synthetic", repeats);
    }
    for (const auto & path : fixtures) {
        std::ifstream file(path, std::ios::binary);
        char magic[8]; uint32_t geometry[4];
        file.read(magic, 8); file.read(reinterpret_cast<char *>(geometry), 16);
        if (!file || std::string(magic, 8) != "EXL3REF1") throw std::runtime_error("Invalid fixture");
        const unsigned k = geometry[0], n = geometry[1], bits = geometry[2], cb = geometry[3];
        if (!k || !n || k > 16384 || n > 16384 || k % 128 || n % 128 || bits < 1 || bits > 8 || cb > 2 ||
            uint64_t(k) * n > 32 * 1024 * 1024) throw std::runtime_error("Invalid fixture shape");
        std::vector<uint16_t> packed(size_t(k) * n * bits / 16), expected(size_t(k) * n);
        file.read(reinterpret_cast<char *>(packed.data()), std::streamsize(packed.size() * 2));
        file.seekg(std::streamoff((k + n) * 2), std::ios::cur);
        file.read(reinterpret_cast<char *>(expected.data()), std::streamsize(expected.size() * 2));
        if (!file) throw std::runtime_error("Truncated fixture");
        decode_case(cl, program, dsp, k, n, bits, cb, packed, expected, "cuda_fixture", repeats);
    }
    // Small runtime group and larger streaming chunks. These are synthetic,
    // warmed repeated buffers, not whole-model or sustained-speed measurements.
    for (unsigned bits : {4u, 6u, 8u}) {
        for (unsigned n : {128u, bits == 8 ? 128u : 6144u}) {
            if (bits == 8 && n != 128) continue;
            const unsigned k = n == 128 ? 128 : 2048;
            std::vector<uint16_t> packed(size_t(k) * n * bits / 16), expected(size_t(k) * n);
            for (auto & value : packed) {
                random ^= random << 13; random ^= random >> 17; random ^= random << 5; value = uint16_t(random);
            }
            exl3::decode_inner(packed.data(), k, n, bits, exl3::Codebook::mul1, expected.data());
            decode_case(cl, program, dsp, k, n, bits, 2, packed, expected,
                        n == 128 ? "single_group" : "streaming_chunk", repeats);
            if (bits == 8) break;
        }
    }
}

static void inventory_vulkan() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "exl3-gpu-npu-probe";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    const VkResult result = vkCreateInstance(&create, nullptr, &instance);
    std::cout << "vulkan_create=" << result << '\n';
    if (result) return;
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    for (auto device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        std::cout << "vulkan_device=" << properties.deviceName << '\n';
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
        for (const auto & ext : extensions) std::cout << "vulkan_extension=" << ext.extensionName << '\n';
        VkPhysicalDeviceMemoryProperties memory;
        vkGetPhysicalDeviceMemoryProperties(device, &memory);
        for (unsigned i = 0; i < memory.memoryTypeCount; ++i)
            std::cout << "vulkan_memory_type=" << i << " flags=" << memory.memoryTypes[i].propertyFlags << '\n';
    }
    vkDestroyInstance(instance, nullptr);
}

int main(int argc, char ** argv) try {
    std::cout.setf(std::ios::unitbuf);
    inventory_vulkan();
    OpenCL cl;
    std::cout << "opencl_device=" << cl.info(CL_DEVICE_NAME) << '\n'
              << "opencl_version=" << cl.info(CL_DEVICE_VERSION) << '\n'
              << "opencl_driver=" << cl.info(CL_DRIVER_VERSION) << '\n'
              << "opencl_extensions=" << cl.info(CL_DEVICE_EXTENSIONS) << '\n';
    cl_uint padding = 0, page = 0;
    check(cl.clGetDeviceInfo(cl.device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(padding), &padding, nullptr), "padding");
    check(cl.clGetDeviceInfo(cl.device, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(page), &page, nullptr), "page");
    std::cout << "external_padding=" << padding << " external_page=" << page << '\n';
    void * memory = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_CACHED, 65536 + padding + page);
    if (!memory) throw std::runtime_error("rpcmem_alloc failed");
    const int fd = rpcmem_to_fd(memory);
    cl_mem_ion_host_ptr external{{CL_MEM_ION_HOST_PTR_QCOM, CL_MEM_HOST_WRITEBACK_QCOM}, fd, memory};
    cl_int status;
    cl_mem buffer = cl.clCreateBuffer(cl.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                                     65536, &external, &status);
    std::cout << "rpcmem_fd=" << fd << " opencl_ion_import_status=" << status << '\n';
    if (buffer) cl.clReleaseMemObject(buffer);
    rpcmem_free(memory);
    check(status, "ION/DMA-BUF import");
    if (argc > 1) {
        const std::string mode = argv[1];
        if (mode != "interop" && mode != "interop-finish" && mode != "dequant" && mode != "pmu") throw std::runtime_error("Unknown mode");
        const unsigned repeats = argc > 2 ? unsigned(std::stoul(argv[2])) : 31;
        if (!repeats || repeats > 1000) throw std::runtime_error("Invalid repetitions");
        Program program(cl, "exl3_dequant.cl");
        if (mode == "pmu") {
            pmu_probe(cl, program, repeats, argc > 3 ? unsigned(std::stoul(argv[3])) : 0);
        } else if (mode == "dequant") {
            std::vector<std::string> fixtures;
            for (int i = 3; i < argc; ++i) fixtures.emplace_back(argv[i]);
            dequant(cl, program, repeats, fixtures);
        } else interop(cl, program, mode == "interop", repeats);
        std::cout << "PASS mode=" << mode << '\n';
    }
    return 0;
} catch (const std::exception & error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
