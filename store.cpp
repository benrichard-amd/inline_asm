#include <hip/hip_runtime.h>
#include <iostream>

// #ifdef __NVCC__
// #define STG_WRAP_S1(func, P, T, cache, type, reg) \
// __device__ __forceinline__ void func(P *p, T v) { asm volatile ("st.global." cache "." type " [%0], %1;" :: "l"(p), reg(v)); }
// STG_WRAP_S1(stg,    uint64,  uint64, "wb", "u64", "l")
// STG_WRAP_S1(stg_cg, uint64,  uint64, "cg", "u64", "l")
// STG_WRAP_S1(stg,     float,   float, "wb", "f32", "f")
// STG_WRAP_S1(stg_cg,  float,   float, "cg", "f32", "f")
// __device__ __forceinline__ void store(uint64* p, uint64 v, bool b=true) { if (b) stg(p, v); }
// __device__ __forceinline__ void store(float* p,  float v, bool b=true) { if (b) stg(p, v); }
// #endif

// flat_store_ instruction
#define FLAT_STORE(p, v, type, cache) \
asm volatile ("flat_store_" type " %0, %1 " cache :: "v"(p), "v"(v))

// flat_store_ instruction followed by s_waitcnt
#define FLAT_STORE_WAIT(p, v, type, cache, wait_inst, wait_cmd) \
asm volatile ("flat_store_" type " %0, %1 " cache "\n\t" \
wait_inst " " wait_cmd "(%2)" \
:: "v"(p), "v"(v), "I"(0x00))

// flat_store_ with memory scope 
#define FLAT_STORE_WB(p, v, type) FLAT_STORE(p, v, type, "sc0")
#define FLAT_STORE_CG(p, v, type) FLAT_STORE(p, v, type, "sc0 nt")
#define FLAT_STORE_CS(p, v, type) FLAT_STORE(p, v, type, "sc0")
#define FLAT_STORE_WT(p, v, type) FLAT_STORE(p, v, type, "sc0")

#define FLAT_STORE_WB_WAIT(p, v, type) FLAT_STORE_WAIT(p, v, type, "sc0", "s_waitcnt", "")
#define FLAT_STORE_CG_WAIT(p, v, type) FLAT_STORE_WAIT(p, v, type, "sc0 nt", "s_waitcnt", "")
#define FLAT_STORE_CS_WAIT(p, v, type) FLAT_STORE_WAIT(p, v, type, "sc0", "s_waitcnt", "vmcnt")
#define FLAT_STORE_WT_WAIT(p, v, type) FLAT_STORE_WAIT(p, v, type, "sc0", "s_waitcnt", "vmcnt")

// Function wrappers
#define STORE_WB_FUNC(T, type) __device__ __forceinline__ void store_wb(T* p, T v) { FLAT_STORE_WB(p, v, type); }
#define STORE_WB_WAIT_FUNC(T, type) __device__ __forceinline__ void store_wb_wait(T* p, T v) { FLAT_STORE_WB_WAIT(p, v, type); }

#define STORE_CG_FUNC(T, type) __device__ __forceinline__ void store_cg(T* p, T v) { FLAT_STORE_CG(p, v, type); }
#define STORE_CG_WAIT_FUNC(T, type) __device__ __forceinline__ void store_cg_wait(T* p, T v) { FLAT_STORE_CG_WAIT(p, v, type); }

#define STORE_CS_FUNC(T, type) __device__ __forceinline__ void store_cs(T* p, T v) { FLAT_STORE_CS(p, v, type); }
#define STORE_CS_WAIT_FUNC(T, type) __device__ __forceinline__ void store_cs_wait(T* p, T v) { FLAT_STORE_CS_WAIT(p, v, type); }

#define STORE_WT_FUNC(T, type) __device__ __forceinline__ void store_wt(T* p, T v) { FLAT_STORE_WT(p, v, type); }
#define STORE_WT_WAIT_FUNC(T, type) __device__ __forceinline__ void store_wt_wait(T* p, T v) { FLAT_STORE_WT_WAIT(p, v, type); }

typedef unsigned __int128 uint128_t;

STORE_WB_FUNC(float, "dword")
STORE_WB_WAIT_FUNC(float, "dword")
STORE_WB_FUNC(long long, "dwordx2")
STORE_WB_WAIT_FUNC(long long, "dwordx2")
STORE_WB_FUNC(uint128_t, "dwordx4")
STORE_WB_WAIT_FUNC(uint128_t, "dwordx4")

STORE_CG_FUNC(float, "dword")
STORE_CG_WAIT_FUNC(float, "dword")
STORE_CG_FUNC(long long, "dwordx2")
STORE_CG_WAIT_FUNC(long long, "dwordx2")
STORE_CG_FUNC(uint128_t, "dwordx4")
STORE_CG_WAIT_FUNC(uint128_t, "dwordx4")

STORE_CS_FUNC(float, "dword")
STORE_CS_WAIT_FUNC(float, "dword")
STORE_CS_FUNC(long long, "dwordx2")
STORE_CS_WAIT_FUNC(long long, "dwordx2")
STORE_CS_FUNC(uint128_t, "dwordx4")
STORE_CS_WAIT_FUNC(uint128_t, "dwordx4")

STORE_WT_FUNC(float, "dword")
STORE_WT_WAIT_FUNC(float, "dword")
STORE_WT_FUNC(long long, "dwordx2")
STORE_WT_WAIT_FUNC(long long, "dwordx2")
STORE_WT_FUNC(uint128_t, "dwordx4")
STORE_WT_WAIT_FUNC(uint128_t, "dwordx4")

enum {
    MODE_DEFAULT,
    MODE_WB,
    MODE_CG,
    MODE_CS,
    MODE_WT
};

template<typename T, int mode, bool wait> __device__ void store(T* p, T v)
{
    if(wait) {
        if(mode == MODE_WB) {
            store_wb_wait(p, v);
        } else if(mode == MODE_CG) {
            store_cg_wait(p, v);
        } else if(mode == MODE_CS) {
            store_cs_wait(p, v);
        } else if(mode == MODE_WT) {
            store_wt_wait(p, v);
        }
    } else {
        if(mode == MODE_WB) {
            store_wb(p, v);
        } else if(mode == MODE_CG) {
            store_cg(p, v);
        } else if(mode == MODE_CS) {
            store_cs(p, v);
        } else if(mode == MODE_WT) {
            store_wt(p, v);
        }
    }
}



void HIP_CALL(hipError_t err)
{
    if(err != hipSuccess) {
        std::cout << "HIP Error: " << (int)err << " " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
}

// Timer for measuring kernel duration
class HIPTimer {

private:
    hipEvent_t m_start;
    hipEvent_t m_stop;

public:
    HIPTimer()
    {
        HIP_CALL(hipEventCreate(&m_start));
        HIP_CALL(hipEventCreate(&m_stop));
    }

    void start()
    {
        HIP_CALL(hipEventRecord(m_start));
    }

    void stop()
    {
        HIP_CALL(hipEventRecord(m_stop));
    }

    double elapsed()
    {
        float ms;
        HIP_CALL(hipEventElapsedTime(&ms, m_start, m_stop));

        return (double)ms / 1000.0;
    }
};


template<typename T, int mode, bool wait> __global__ void hip_memset(T* x, T v, size_t count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    for(size_t i = tid; i < count; i += grid_size) {
        T *p = x + i;

        if(mode == MODE_DEFAULT) {
            *p = v;
        } else {
            store<T, mode, wait>(p, v);
        }
    }
}



template<int mode, bool wait> bool store_unit_test(size_t size_gb)
{
    float* buffer = nullptr;
    int bufsize = size_gb * 1024 * 1024 * 1024 / sizeof(float);
    float value = 3;
 
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, 0));
    int cu_count = prop.multiProcessorCount;

    HIP_CALL(hipMallocManaged(&buffer, sizeof(int) * bufsize));

    hip_memset<float, mode, wait><<<cu_count * 4, 64>>>(buffer, value, bufsize);
    HIP_CALL(hipDeviceSynchronize());

    bool pass = true;
    for(int i = 0; i < bufsize; i++) {
        if(buffer[i] != value) {
            pass = false;
            break;
        }
    }
    HIP_CALL(hipFree(buffer));

    return pass;
}

template<typename T, int mode, bool wait> void store_perf_test(size_t size_gb)
{
    T* buffer = nullptr;
    size_t bufsize = size_gb * 1024 * 1024 * 1024 / sizeof(T);
    T value = 3;
 
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, 0));
    int cu_count = prop.multiProcessorCount;

    HIP_CALL(hipMalloc(&buffer, sizeof(T) * bufsize));

    HIPTimer t;
    t.start();

    int count = 8;

    for(int i = 0; i < count; i++) {
        hip_memset<T, mode, wait><<<cu_count * 16, 64>>>(buffer, value, bufsize);
    }
    t.stop(); 
    HIP_CALL(hipDeviceSynchronize());

    size_t total = bufsize * count * sizeof(T);
    double speed = (double)total / count / t.elapsed();
    std::cout << "      " << speed / 1e9 << " GB/sec" << std::endl;
    HIP_CALL(hipFree(buffer));
}

int main(int argc, char**argv)
{
    bool pass = true;

    pass &= store_unit_test<MODE_WB, true>(1);
    pass &= store_unit_test<MODE_WB, false>(1);
    pass &= store_unit_test<MODE_CG, true>(1);
    pass &= store_unit_test<MODE_CG, false>(1);
    pass &= store_unit_test<MODE_CS, true>(1);
    pass &= store_unit_test<MODE_CS, false>(1);
    pass &= store_unit_test<MODE_WT, true>(1);
    pass &= store_unit_test<MODE_WT, false>(1);

    if(pass) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }

    size_t size = 32;

    // default
    std::cout << "  mode = default" << std::endl;
    store_perf_test<float, MODE_DEFAULT, false>(size);
    store_perf_test<long long, MODE_DEFAULT, false>(size);
    store_perf_test<uint128_t, MODE_DEFAULT, false>(size);

    // wb
    std::cout << "  mode = wb" << std::endl;
    std::cout << "    waitcnt = off" << std::endl;
    store_perf_test<float, MODE_WB, false>(size);
    store_perf_test<long long, MODE_WB, false>(size);
    store_perf_test<uint128_t, MODE_WB, false>(size);
    
    std::cout << "    waitcnt = on" << std::endl;
    store_perf_test<float, MODE_WB, true>(size);
    store_perf_test<long long, MODE_WB, true>(size);
    store_perf_test<uint128_t, MODE_WB, true>(size);
    
    // cg 
    std::cout << "  mode = cg" << std::endl;
    std::cout << "    waitcnt = off" << std::endl;
    store_perf_test<float, MODE_CG, false>(size);
    store_perf_test<long long, MODE_CG, false>(size);
    store_perf_test<uint128_t, MODE_CG, false>(size);
    
    std::cout << "    waitcnt = on" << std::endl;
    store_perf_test<float, MODE_CG, true>(size);
    store_perf_test<long long, MODE_CG, true>(size);
    store_perf_test<uint128_t, MODE_CG, true>(size);

    // cs
    std::cout << "  mode = cs" << std::endl;
    std::cout << "    waitcnt = off" << std::endl;
    store_perf_test<float, MODE_CS, false>(size);
    store_perf_test<long long, MODE_CS, false>(size);
    store_perf_test<uint128_t, MODE_CS, false>(size);
    
    std::cout << "    waitcnt = on" << std::endl;
    store_perf_test<float, MODE_CS, true>(size);
    store_perf_test<long long, MODE_CS, true>(size);
    store_perf_test<uint128_t, MODE_CS, true>(size);

    // wt
    std::cout << "  mode = wt" << std::endl;
    std::cout << "    waitcnt = off" << std::endl;
    store_perf_test<float, MODE_WT, false>(size);
    store_perf_test<long long, MODE_WT, false>(size);
    store_perf_test<uint128_t, MODE_WT, false>(size);
    
    std::cout << "    waitcnt = on" << std::endl;
    store_perf_test<float, MODE_WT, true>(size);
    store_perf_test<long long, MODE_WT, true>(size);
    store_perf_test<uint128_t, MODE_WT, true>(size);

    return 0;
}