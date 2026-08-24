#include <cuda/std/limits>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <cuda/ptx>
#include <cassert>
#include <cudaTypedefs.h>
#include "moe_w8a8.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cuErrchk(ans) { cuAssert((ans), __FILE__, __LINE__); }
#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Driver-API counterpart of gpuAssert -- cuTensorMapEncodeTiled returns CUresult
// (driver API), not cudaError_t (runtime API); these are distinct enums with
// distinct string-lookup functions, so gpuErrchk/cudaGetErrorString cannot be
// reused here. Without this check, a failed tensor-map encode (bad alignment,
// an unsupported box_size/swizzle combination, etc.) was previously silent --
// the kernel would then read through a garbage tensor map and produce wrong
// output with no error at all, not a crash.
inline void cuAssert(CUresult code, const char *file, int line, bool abort=true)
{
    if (code != CUDA_SUCCESS)
    {
        const char *err_str = nullptr;
        cuGetErrorString(code, &err_str);
        fprintf(stderr, "CUassert: %s %s %d\n", err_str ? err_str : "<unknown CUresult>", file, line);
        if (abort) exit(static_cast<int>(code));
    }
}

// fp8 alias now comes from moe_w8a8.h (included above), which also declares
// this file's public entry point so the definition below is checked against
// it by the compiler.

namespace aa4
{
    PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled()
    {
        cudaDriverEntryPointQueryResult driver_status;
        void* cuTensorMapEncodeTiled_ptr = nullptr;
        gpuErrchk(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
        ASSERT(driver_status == cudaDriverEntryPointSuccess, "Failed driver status %d", 0);

        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
    }
}
using namespace aa4;

constexpr __device__ __forceinline__ int32_t const_ceil(float num)
{
    return (static_cast<float>(static_cast<int32_t>(num)) == num)
        ? static_cast<int32_t>(num)
        : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x)
{
    return (((x) & 0x3FFFF) >> 4);
}

template <int BITS, int BASE = 4, int SHIFT = 3>
__device__ __forceinline__ int32_t swizzle(const int32_t i)
{
    if constexpr(BITS == 0)
        return i;
    constexpr uint32_t S_MASK = ((1 << BITS) - 1) << (BASE + SHIFT);
    return i ^ ((i & S_MASK) >> SHIFT);
}

template<int SDO, int LDO, uint64_t S_MODE>
__device__ __forceinline__ uint64_t make_smem_descriptor(fp8* ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = matrix_descriptor_encode(addr);
    desc |= ((uint64_t)LDO) << 16;
    desc |= ((uint64_t)SDO) << 32;
    desc |= S_MODE << 62;
    return desc;
}

__device__ __forceinline__ float swiglu_mul(float x, float w)
{
    return (x / (1 + __expf(-x))) * w;
}

__device__ __forceinline__ void warpgroup_arrive()
{
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch()
{
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_wait()
{
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma8(float d[1][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3},   "
        " %4,"
        " %5,"
        " %6, %7, %8;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma16(float d[2][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},   "
        " %8,"
        " %9,"
        " %10, %11, %12;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma24(float d[3][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11},  "
        " %12,"
        " %13,"
        " %14, %15, %16;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma32(float d[4][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15},  "
        " %16,"
        " %17,"
        " %18, %19, %20;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma40(float d[5][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n40k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19},  "
        " %20,"
        " %21,"
        " %22, %23, %24;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma48(float d[6][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23},  "
        " %24,"
        " %25,"
        " %26, %27, %28;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma56(float d[7][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n56k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27},  "
        " %28,"
        " %29,"
        " %30, %31, %32;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma64(float d[8][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},  "
        " %32,"
        " %33,"
        " %34, %35, %36;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma72(float d[9][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35},  "
        " %36,"
        " %37,"
        " %38, %39, %40;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma80(float d[10][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n80k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39},  "
        " %40,"
        " %41,"
        " %42, %43, %44;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma88(float d[11][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n88k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43},  "
        " %44,"
        " %45,"
        " %46, %47, %48;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma96(float d[12][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n96k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},  "
        " %48,"
        " %49,"
        " %50, %51, %52;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
         "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma104(float d[13][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n104k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51},  "
        " %52,"
        " %53,"
        " %54, %55, %56;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
         "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
         "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma112(float d[14][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n112k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55},  "
        " %56,"
        " %57,"
        " %58, %59, %60;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
         "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
         "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
         "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma120(float d[15][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n120k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  %56,  %57,  %58,  %59},  "
        " %60,"
        " %61,"
        " %62, %63, %64;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
         "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
         "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
         "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
         "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma128(float d[16][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},  "
        " %64,"
        " %65,"
        " %66, %67, %68;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
         "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
         "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
         "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
         "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]),
         "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int BM>
__device__ __forceinline__ void wgmma(float d[BM/8][4], uint64_t desc_a, uint64_t desc_b)
{
    if constexpr (BM == 8)
        wgmma8<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 16)
        wgmma16<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 24)
        wgmma24<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 32)
        wgmma32<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 40)
        wgmma40<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 48)
        wgmma48<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 56)
        wgmma56<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 64)
        wgmma64<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 72)
        wgmma72<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 80)
        wgmma80<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 88)
        wgmma88<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 96)
        wgmma96<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 104)
        wgmma104<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 112)
        wgmma112<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 120)
        wgmma120<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 128)
        wgmma128<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
}

__device__ inline void load_async(fp8 *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx)
{
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile (
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
        "r"(global_row_idx), "r"(global_col_idx)
        : "memory"
    );
}

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG4(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

__device__ __forceinline__ void init_barrier(uint64_t* bar, int thread_count, int transaction_count)
{
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count + transaction_count)
    );
}

__device__ __forceinline__ void cp_async_mbarrier_arrive(uint64_t* bar)
{
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n"
        :: "r"(bar_ptr)
    );
}

__device__ __forceinline__ void arrive(uint64_t* bar, uint32_t count=1)
{
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.arrive.shared.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}

__device__ __forceinline__ void expect_bytes(uint64_t* bar, uint32_t bytes)
{
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile ("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "r"(bytes));
}

__device__ __forceinline__ void wait(uint64_t* bar, int kPhaseBit)
{
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
        "@!P1                       bra.uni WAIT;\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ __forceinline__ void ld_matrix_x2(uint32_t* tile, uint32_t mat)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
            : "=r"(tile[0]), "=r"(tile[1]) : "r"(mat));
}

__device__ __forceinline__ void ld_matrix_x4(uint32_t* tile, uint32_t mat)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(tile[0]), "=r"(tile[1]), "=r"(tile[2]), "=r"(tile[3]) : "r"(mat));
}

__device__ __forceinline__ void st_matrix_x4(uint32_t* tile, uint32_t mat)
{
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
            :
            : "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
            : "memory"
            );
}

__device__ __forceinline__ void st_matrix_x4_trans(uint32_t* tile, uint32_t mat)
{
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
            :
            : "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
            : "memory"
            );
}

__device__ __forceinline__ void st_matrix_x2_trans(uint32_t* tile, uint32_t mat)
{
    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};"
            :
            : "r"(mat), "r"(tile[0]), "r"(tile[1])
            : "memory"
            );
}

__device__ __forceinline__ void st_matrix_x1_trans(uint32_t* tile, uint32_t mat)
{
    asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};"
            :
            : "r"(mat), "r"(tile[0])
            : "memory"
            );
}

template <uint32_t RegCount>
__device__ void reg_alloc()
{
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void reg_dealloc()
{
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template<int STAGES, int WN, int BM, int BK, int BN>
struct smem_up
{
    alignas(1024) fp8 w[STAGES*WN*BK*BN];
    alignas(1024) fp8 x[STAGES*BK*BM];
    float scale_w_up[2*STAGES * 2];
    float scale_x_up[2*STAGES * BM];
};

template<int STAGES, int WN, int BM, int BK, int BN>
struct smem_down
{
    alignas(1024) fp8 w[STAGES*WN*BK*BN];
    alignas(1024) fp8 x[BM*WN*BN/2];
    alignas(16) __nv_bfloat16 out[BM*(BK*2 + 8)];
};

template <int BM, int BK, int BN, int WN, int STAGES, int PRODUCER_THREADS>
__global__ __launch_bounds__(WN*32 + PRODUCER_THREADS) void fused_moe_w8a8_wgmma_up_down_acc_kernel(
        const fp8* __restrict__ x,
        const float* __restrict__ x_scale,
        const __grid_constant__ CUtensorMap tensor_map_w,
        const float* __restrict__ w_scale,
        const __grid_constant__ CUtensorMap tensor_map_w2,
        const float* __restrict__ w2_scale,
        __nv_bfloat16* __restrict__ out,
        const int* __restrict__ sorted_token_ids,
        const int* __restrict__ expert_ids,
        const int* __restrict__ num_tokens_post_padded,
        const float* __restrict__ topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        float scaling_factor
        )
{
    constexpr int CONSUMER_THREADS = WN*32;
    constexpr int WARPGROUPS = WN / 4;
    const int32_t warpM = blockIdx.y;
    const int exp_idx = expert_ids[warpM];
    if(warpM * BM >= num_tokens_post_padded[0])
        return;

    constexpr int block_shape[2] = {128, 128};

    constexpr uint32_t S_BITS_UP = 3;
    constexpr uint32_t S_MODE_UP = 4 - S_BITS_UP;

    constexpr uint32_t S_BITS_DOWN = WN*BN == 256 ? 3 :
                                     WN*BN == 128 ? 2 :
                                                    1;

    constexpr uint32_t S_MODE_DOWN = 4 - S_BITS_DOWN;

    const int K2 = N/2;
    const int N2 = K;
    const int lane_id = threadIdx.x%32;
    const bool is_producer = threadIdx.x < PRODUCER_THREADS;
    const int warp_id = is_producer ? threadIdx.x/32 : (threadIdx.x-PRODUCER_THREADS)/32;

    constexpr int BK2 = WN*BN/2;
    constexpr int BN2 = BK*2;

    constexpr int WS = WN*BK*BN;
    constexpr int XS = BK*BM;
    constexpr int TB = 16;
    constexpr int TO = TB/sizeof(fp8);
    constexpr int TPT = BM/4;

    constexpr int TN2 = BN2/(64*WARPGROUPS);
    constexpr int scales_per_stage_down = BN2 / block_shape[0];

    extern __shared__ __align__(1024) uint8_t sh[];
    smem_up<STAGES, WN, BM, BK, BN>& s = *reinterpret_cast<smem_up<STAGES, WN, BM, BK, BN>*>(sh);

    __shared__ __align__(8) uint64_t bar[2*STAGES];
    __shared__ float topk_scales[BM];
    __shared__ float scale_w_down[2*STAGES * (BK*2)/128];
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < STAGES; i++)
        {
            init_barrier(&bar[i], PRODUCER_THREADS + 1, 0);
            init_barrier(&bar[i + STAGES], CONSUMER_THREADS, 0);
        }
    }
    __syncthreads();
    auto consumer_sync = [&]()
    {
        asm volatile("bar.sync 0, %0;\n" :: "n"(CONSUMER_THREADS));
    };

    int n_stages_up = K/block_shape[0];
    int n_stages_down = (N2 + BN2 - 1)/BN2;

    constexpr int TM = BM/8;
    constexpr int TN = BN/16;
    nv_bfloat162 f_acc[TN][TM][2];
    memset(f_acc, 0, sizeof(f_acc));

    int p = 0;
    if (is_producer)
    {
        constexpr int X_IT = const_ceil(XS/float(PRODUCER_THREADS*TO));
        int tsrc[X_IT];
        int i = (threadIdx.x)*TO;
        for(int r = 0; r < X_IT; r += 1)
        {
            if (i < XS)
            {
                int tdest = sorted_token_ids[warpM*BM + r*(PRODUCER_THREADS/8) + threadIdx.x/8];
                tsrc[r] = tdest / top_k;
                if(threadIdx.x % 8 == 0 && tsrc[r] < M && i < XS)
                {
                    uint32_t smem = __cvta_generic_to_shared(&topk_scales[r*(PRODUCER_THREADS/8) + threadIdx.x/8]);
                    CP_ASYNC_CG4(smem, topk_weights + tdest, 4);
                }
            }
            i += PRODUCER_THREADS*TO;
        }
        int smem_stage = 0;
        const int w_row_up = exp_idx * N + (blockIdx.x)*WN*BN;
        for (int load_stage = 0; load_stage < n_stages_up; load_stage++)
        {
            if (smem_stage == STAGES)
            {
                p ^= 1;
                smem_stage = 0;
            }
            const int off = load_stage * block_shape[0];
            int i = (threadIdx.x)*TO;
            int col = off + i%BK;
            int swizzled = swizzle<S_BITS_UP>(i);

            wait(bar + STAGES + smem_stage, p);

            for(int r = 0; r < X_IT; r += 1)
            {
                int row = tsrc[r];
                if(row < M && i < XS)
                {
                    uint32_t sm = __cvta_generic_to_shared(s.x + smem_stage*XS + swizzled);
                    CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row*K + col), TB);
                    if(smem_stage == 0 && threadIdx.x % 8 < STAGES && load_stage + threadIdx.x%8 < n_stages_up)
                    {
                        uint32_t smem = __cvta_generic_to_shared(s.scale_x_up + p*STAGES*BM +
                                + (threadIdx.x%8) * BM + r * (PRODUCER_THREADS/8) + threadIdx.x/8);
                        CP_ASYNC_CG4(smem, &x_scale[row*(K/block_shape[1]) + load_stage + threadIdx.x%8], 4);
                    }
                }
                i += PRODUCER_THREADS*TO;
                swizzled += PRODUCER_THREADS*TO;
            }

            const int scale_rows_w_up = N/block_shape[1];
            const int scale_cols_w_up = K/block_shape[0];
            if(smem_stage == 0 && load_stage + threadIdx.x/2 < n_stages_up && threadIdx.x/2 < STAGES)
            {
                uint32_t smem = __cvta_generic_to_shared(s.scale_w_up + p*STAGES*2 + threadIdx.x);
                CP_ASYNC_CG4(smem,
                        &w_scale[exp_idx * scale_rows_w_up * scale_cols_w_up +
                        (blockIdx.x*2 + threadIdx.x%2)*scale_cols_w_up + load_stage + threadIdx.x/2], 4);
            }
            cp_async_mbarrier_arrive(bar + smem_stage);

            if(threadIdx.x == 0)
            {
                expect_bytes(bar + smem_stage, WS*sizeof(fp8));
                load_async(s.w + smem_stage*WS, &tensor_map_w, bar + smem_stage, w_row_up, off);
            }
            smem_stage++;
        }

        smem_down<STAGES, WN, BM, BK, BN>& s_d = *reinterpret_cast<smem_down<STAGES, WN, BM, BK, BN>*>(sh);

        for (int load_stage = 0; load_stage < n_stages_down; load_stage++)
        {
            if (smem_stage == STAGES)
            {
                p ^= 1;
                smem_stage = 0;
            }
            wait(bar + STAGES + smem_stage, p);
            const int scale_rows_w = N2/block_shape[1];
            const int scale_cols_w = K2/block_shape[0];
            if((load_stage)%STAGES == 0 && threadIdx.x < STAGES * scales_per_stage_down && load_stage + (threadIdx.x/scales_per_stage_down) < n_stages_down)
            {
                uint32_t smem = __cvta_generic_to_shared(scale_w_down + ((load_stage/STAGES)%2)*STAGES * scales_per_stage_down + threadIdx.x);
                CP_ASYNC_CG4(smem,
                        &w2_scale[exp_idx * scale_rows_w * scale_cols_w +
                        blockIdx.x * (BK2/block_shape[0]) +
                        (load_stage*scales_per_stage_down + threadIdx.x)*gridDim.x*(BK2/block_shape[0])], 4);
            }
            if(threadIdx.x == 0)
            {
                const int w_row_down = blockIdx.x*BK2;
                const int w_col_down = exp_idx * N2 + load_stage*BN2;
                expect_bytes(bar + smem_stage, WS*sizeof(fp8));
                load_async(s_d.w + smem_stage*WS, &tensor_map_w2, bar + smem_stage, w_col_down, w_row_down);
            }
            cp_async_mbarrier_arrive(bar + smem_stage);
            smem_stage++;
        }
    }
    else
    {
        int token_src = M;
        if(threadIdx.x < PRODUCER_THREADS + BM)
            token_src = sorted_token_ids[warpM*BM + threadIdx.x-PRODUCER_THREADS] / top_k;
        for (int i = 0; i < STAGES; i++)
            arrive(&bar[STAGES + i]);

        int smem_stage = 0;
        for (int compute_stage = 0; compute_stage < n_stages_up; compute_stage += 1)
        {
            if (smem_stage == STAGES)
            {
                p ^= 1;
                smem_stage = 0;
            }
            wait(bar + smem_stage, p);

            float scale_x[TPT];
            for(int t = 0; t < TPT; t += 2)
            {
                int token_idx = (t/2)*8 + (lane_id%4)*2;
                float2 sx = *reinterpret_cast<const float2*>(&s.scale_x_up[p*STAGES*BM + smem_stage * BM + token_idx]);
                scale_x[t] = sx.x;
                scale_x[t+1] = sx.y;
            }
            float scale_w[2];
            float2 sw = *reinterpret_cast<const float2*>(&s.scale_w_up[p*STAGES*2 + smem_stage * 2]);
            scale_w[0] = sw.x;
            scale_w[1] = sw.y;

            // TEMPORARY debug probe, 2026-08-24 -- the down-proj combine
            // checked out clean twice (s_w/token_scale both sane, internally
            // consistent), so the remaining ~145% same-coordinate error must
            // be upstream: either the up-proj's OWN scale read-back (this
            // shared-memory round-trip for scale_x_up/scale_w_up) or the raw
            // wgmma K-reduction itself. Dumps local BM-position (decode via
            // sorted_ids_debug.pt, same technique as DEBUGXY), smem_stage
            // (the K-block index), and both scales, so Python can cross-check
            // scale_x directly against the real x_scale[token,k_block] value
            // and scale_w against w_scale[exp_idx,...]. Remove once found.
            if (blockIdx.x == 0 && blockIdx.y == 0 && lane_id == 0 && warp_id == 0)
            {
                printf("DEBUGUPSCALE compute_stage=%d smem_stage=%d exp_idx=%d local_pos0=%d "
                       "scale_x0=%.9f scale_w0=%.9f scale_w1=%.9f\n",
                       compute_stage, smem_stage, exp_idx, (lane_id%4)*2,
                       scale_x[0], scale_w[0], scale_w[1]);
                // TEMPORARY debug probe, 2026-08-24 -- RAW fp8 activation
                // bytes for local BM-slot 0 (token 0, per sorted_ids), first
                // 8 elements of this K-block's 128-wide chunk. i=0..7 is
                // swizzle-transparent for any S_BITS_UP (the swizzle mask
                // only touches bits >= 7, all zero for i<128), so these
                // shared-memory values should be directly, exactly
                // comparable to x_fp8[0, compute_stage*128 : +8] in Python
                // with zero decoding needed -- checks whether cp.async
                // loaded the right raw bytes at all, independent of any
                // scale. Remove once found.
                printf("DEBUGUPDATA compute_stage=%d s.x[0:8]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                       compute_stage,
                       float(s.x[smem_stage*XS+0]), float(s.x[smem_stage*XS+1]),
                       float(s.x[smem_stage*XS+2]), float(s.x[smem_stage*XS+3]),
                       float(s.x[smem_stage*XS+4]), float(s.x[smem_stage*XS+5]),
                       float(s.x[smem_stage*XS+6]), float(s.x[smem_stage*XS+7]));
            }

            float tile_acc[TN][TM][4];
            memset(tile_acc, 0, sizeof(tile_acc));
            for (int tn = 0; tn < TN; tn++)
            {
                for (int tm = 0; tm < TM; tm++)
                {
                    for (int t = 0; t < 4; t++)
                        asm volatile("" : "+f"(tile_acc[tn][tm][t]) :: "memory");
                }
            }

            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
            warpgroup_arrive();
            for(int tn = 0; tn < TN; tn++)
            {
                fp8* sw = s.w + smem_stage*WS + (warp_id/4)*(BN*4)*BK + tn*64*BK;
                fp8* sx = s.x + smem_stage*XS;
                uint64_t desc_w = make_smem_descriptor<64, 1, S_MODE_UP>(sw);
                uint64_t desc_x = make_smem_descriptor<64, 1, S_MODE_UP>(sx);
                wgmma<1,1,1, BM>(tile_acc[tn], desc_w, desc_x);
                wgmma<1,1,1, BM>(tile_acc[tn], desc_w+1*(32>>4), desc_x+1*(32>>4));
                wgmma<1,1,1, BM>(tile_acc[tn], desc_w+2*(32>>4), desc_x+2*(32>>4));
                wgmma<1,1,1, BM>(tile_acc[tn], desc_w+3*(32>>4), desc_x+3*(32>>4));
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(bar + STAGES + smem_stage);

            // TEMPORARY debug probe, 2026-08-24 -- raw, UNSCALED wgmma
            // accumulator for (tn=0,tm=0,lane=0,warp=0) = feature-row 0,
            // token col0=0 (local BM-slot 0 -> token 0, per sorted_ids),
            // reduced over this compute_stage's 128-wide K-chunk. Both
            // activation data (s.x) and its scale round-tripped byte-exact
            // already, so this checks the one remaining unverified piece:
            // does the weight data (TMA-loaded, hardware-swizzled -- can't
            // be read back as a plain linear index the way s.x could) times
            // the verified-correct activation actually reduce to what an
            // independent Python dot product over gu_fp8[0,0,k0:k0+128] @
            // x_fp8[0,k0:k0+128] gives? If they match, weights + wgmma are
            // BOTH cleared and the bug is elsewhere (SiLU combine, f_acc
            // bookkeeping); if not, isolates it to the weight side. Remove
            // once found.
            if (blockIdx.x == 0 && blockIdx.y == 0 && lane_id == 0 && warp_id == 0)
                printf("DEBUGRAWACC compute_stage=%d raw_tile_acc[0][0][0]=%.6f\n",
                       compute_stage, tile_acc[0][0][0]);

            for(int tm = 0; tm < TM; tm++)
            {
                for(int tn = 0; tn < TN; tn++)
                {
                    f_acc[tn][tm][0] = __hadd2(f_acc[tn][tm][0],
                            __floats2bfloat162_rn(
                                scale_w[0] * scale_x[tm*2] * tile_acc[tn][tm][0],
                                scale_w[0] * scale_x[tm*2 + 1] * tile_acc[tn][tm][1]
                                )
                            );

                    f_acc[tn][tm][1] = __hadd2(f_acc[tn][tm][1],
                            __floats2bfloat162_rn(
                                scale_w[1] * scale_x[tm*2] * tile_acc[tn][tm][2],
                                scale_w[1] * scale_x[tm*2 + 1] * tile_acc[tn][tm][3]
                                )
                            );
                }
            }
            smem_stage++;
        }
        consumer_sync();
        smem_down<STAGES, WN, BM, BK, BN>& s_d = *reinterpret_cast<smem_down<STAGES, WN, BM, BK, BN>*>(sh);
        float4* block_max = reinterpret_cast<float4*>(s_d.out);
        constexpr float EPS = 1e-10;
        nv_bfloat162 token_max[TM] = { __floats2bfloat162_rn(EPS, EPS) };
        for(int tn = 0; tn < TN; tn++)
        {
            for(int tm = 0; tm < TM; tm++)
            {
                f_acc[tn][tm][0].x = __float2bfloat16_rn(swiglu_mul(__bfloat162float(f_acc[tn][tm][0].x), __bfloat162float(f_acc[tn][tm][1].x)));
                f_acc[tn][tm][0].y = __float2bfloat16_rn(swiglu_mul(__bfloat162float(f_acc[tn][tm][0].y), __bfloat162float(f_acc[tn][tm][1].y)));
                nv_bfloat162 abs = __habs2(f_acc[tn][tm][0]);
                token_max[tm] = __hmax2(abs, token_max[tm]);
            }
        }
        for(int tm = 0; tm < TM; tm++)
        {
            token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 16), token_max[tm]);
            token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 8), token_max[tm]);
            token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 4), token_max[tm]);
            if (lane_id < 4)
            {
                int off = tm*8 + (lane_id)*2;
                reinterpret_cast<nv_bfloat162*>(block_max + off * WARPGROUPS)[warp_id] = token_max[tm];
            }
        }
        consumer_sync();
        constexpr float fp8_max = 448.0;
        constexpr float fp8_min = -448.0;
        float token_scale[TM][2];

        for(int tm = 0; tm < TM; tm++)
        {
            int off = tm*8 + (lane_id%4)*2;
            for(int wg = 0; wg < WARPGROUPS; wg++)
            {
                float4 bmax = block_max[off * WARPGROUPS + wg];
                token_max[tm] = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.x), token_max[tm]);
                token_max[tm] = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.y), token_max[tm]);
                token_max[tm] = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.z), token_max[tm]);
                token_max[tm] = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.w), token_max[tm]);
            }
            token_scale[tm][0] = __bfloat162float(token_max[tm].x) / fp8_max;
            token_scale[tm][1] = __bfloat162float(token_max[tm].y) / fp8_max;
            for (int t = 0; t < 2; t++)
            {
                for(int tn = 0; tn < TN; tn++)
                {
                    int x_row = tm*8 + (lane_id%4)*2 + t;
                    int x_col = (warp_id/4)*(TN*32) + tn*32 + (warp_id%4)*8 + lane_id/4;
                    float val = __bfloat162float(t == 0 ? f_acc[tn][tm][0].x : f_acc[tn][tm][0].y);
                    // TEMPORARY debug probe, 2026-08-24 -- dumps the correctly-laid-out
                    // (already un-swizzled, human/Python-readable) pre-down-proj SiLU-gated
                    // activation at one known coordinate, to compare directly against
                    // reference_pipeline's `h[0,0]` without needing to decode wgmma's raw
                    // per-thread output-register fragment layout. Remove once the bug is found.
                    if (blockIdx.x == 0 && blockIdx.y == 0 && (x_row == 0 || x_col == 0))
                        printf("DEBUGXY row=%d col=%d val=%.9f lane=%d warp=%d tm=%d tn=%d t=%d scale=%.9f\n",
                               x_row, x_col, val, lane_id, warp_id, tm, tn, t, token_scale[tm][t]);
                    float q = val / token_scale[tm][t];
                    val = fminf(fmaxf(q, fp8_min), fp8_max);
                    int i = x_row*BK2 + x_col;
                    int swizzled = swizzle<S_BITS_DOWN>(i);
                    s_d.x[swizzled] = fp8(val);
                }
            }
        }

        for(int t = 0; t < TPT; t += 2)
        {
            int token_idx = (t/2)*8 + (lane_id%4)*2;
            const float2 topk_w = *reinterpret_cast<const float2*>(&topk_scales[token_idx]);
            token_scale[t/2][0] *= topk_w.x * scaling_factor;
            token_scale[t/2][1] *= topk_w.y * scaling_factor;
        }

        consumer_sync();

        for (int compute_stage = 0; compute_stage < n_stages_down; compute_stage += 1)
        {
            if (smem_stage == STAGES)
            {
                p ^= 1;
                smem_stage = 0;
            }
            float s_w[TN2/2];

            float tile_acc[TN2][TM][4];
            memset(tile_acc, 0, sizeof(tile_acc));
            fp8* sx = s_d.x;
            wait(bar + smem_stage, p);
            for (int tn = 0; tn < TN2; tn++)
            {
                for (int tm = 0; tm < TM; tm++)
                {
                    for (int t = 0; t < 4; t++)
                        asm volatile("" : "+f"(tile_acc[tn][tm][t]) :: "memory");
                }
            }
            warpgroup_arrive();
            const int scale_rows_w = N2/block_shape[1];
            const int scale_cols_w = K2/block_shape[0];
            for (int i = 0; i < (TN2/2); i++)
            {
                s_w[i] = scale_w_down[((compute_stage/STAGES)%2)*STAGES*scales_per_stage_down + (compute_stage%STAGES)*scales_per_stage_down + (warp_id/4)*(TN2/2) + i];
            }

            for(int tn2 = 0; tn2 < TN2; tn2++)
            {
                fp8* sw = s_d.w + smem_stage*WS + ((warp_id/4)*TN2 + tn2)*64*BK2;
                uint64_t desc_w = make_smem_descriptor<BK2/2, 1, S_MODE_DOWN>(sw);
                uint64_t desc_x = make_smem_descriptor<BK2/2, 1, S_MODE_DOWN>(sx);
                for(int tk = 0; tk < BK2/32; tk++)
                {
                    wgmma<1,1,1, BM>(tile_acc[tn2], desc_w, desc_x);
                    desc_w += (32>>4);
                    desc_x += (32>>4);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(bar + STAGES + smem_stage);

            constexpr int PAD = BN2+8;
            __nv_bfloat16 out_tile[TN2/2][TM][8];
            for(int tn2 = 0; tn2 < TN2; tn2 += 2)
            {
                for(int tm = 0; tm < TM; tm++)
                {
                    for (int t = 0; t < 8; t++)
                    {
                        int s = t%2;
                        out_tile[tn2/2][tm][t] = __float2bfloat16_rn(token_scale[tm][s]*tile_acc[tn2 + t/4][tm][t%4]*s_w[tn2/2]);
                        // TEMPORARY debug probe, 2026-08-24 -- breaks down the
                        // down-proj output combine (out = token_scale * raw_wgmma
                        // * s_w) into its 3 factors for one fixed coordinate, to
                        // find which factor is responsible for the ~227,000x
                        // final-output magnitude blowup seen even at
                        // n_stages_down=1 (so it's not a multi-stage bug -- the
                        // single-stage case is already wrong). Compare s_w against
                        // dp_scale's expected magnitude and token_scale against
                        // what quantize_activation_fp8_dynamic-style scales should
                        // look like (both should be small, roughly weight/activation
                        // std over 448, NOT anywhere near 1.0). Remove once found.
                        if (blockIdx.x == 0 && blockIdx.y == 0 && lane_id == 0 && warp_id == 0
                            && tn2 == 0 && tm == 0 && t == 0)
                            printf("DEBUGDOWN compute_stage=%d s_w=%.9f token_scale=%.9f "
                                   "raw_tile_acc=%.9f out=%.9f\n",
                                   compute_stage, s_w[tn2/2], token_scale[tm][s],
                                   tile_acc[tn2 + t/4][tm][t%4],
                                   __bfloat162float(out_tile[tn2/2][tm][t]));
                    }
                }
            }
            asm volatile("cp.async.bulk.wait_group 0;");
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
            consumer_sync();
            for(int tn2 = 0; tn2 < TN2; tn2 += 2)
            {
                for(int tm = 0; tm < TM; tm++)
                {
                    int out_row = tm * 8 + lane_id%8;
                    int out_col = (warp_id/4)*TN2*64 + (warp_id%4)*16 + (lane_id&8) + tn2*64 + (lane_id/16)*64;
                    st_matrix_x4_trans(reinterpret_cast<uint32_t*>(out_tile[tn2/2][tm]),
                            __cvta_generic_to_shared(s_d.out + out_row*PAD + out_col));
                }
            }
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
            consumer_sync();
            if(threadIdx.x < PRODUCER_THREADS + BM)
            {
                if (token_src < M)
                {
                    int row = threadIdx.x - PRODUCER_THREADS;
                    // TEMPORARY debug probe, 2026-08-24 -- every VALID
                    // (non-padding) final add into `out`, with which block/
                    // expert produced it and a sample of what's about to be
                    // added, to find why tokens 8-15 end up ~5-7.6 magnitude
                    // (vs ref's ~1e-5) while 0-7 don't -- routing content
                    // (local_slots) showed no clean single-expert
                    // correlation, so this checks whether it's block-
                    // specific, a double-add (same token_src from multiple
                    // blocks/stages unexpectedly), or bad values already in
                    // s_d.out before the add. Remove once found.
                    printf("DEBUGOUT blockIdx.x=%d blockIdx.y=%d exp_idx=%d compute_stage=%d "
                           "row=%d token_src=%d s_d.out[row*PAD]=%.6f s_d.out[row*PAD+1]=%.6f\n",
                           blockIdx.x, blockIdx.y, exp_idx, compute_stage, row, token_src,
                           __bfloat162float(s_d.out[row*PAD]), __bfloat162float(s_d.out[row*PAD+1]));
                    cuda::ptx::cp_reduce_async_bulk(
                            cuda::ptx::space_global,
                            cuda::ptx::space_shared,
                            cuda::ptx::op_add,
                            out + token_src*N2 + compute_stage*BN2,
                            s_d.out + row*PAD,
                            BN2*sizeof(__nv_bfloat16));
                }
                cuda::ptx::cp_async_bulk_commit_group();
            }
            smem_stage++;
        }
    }
}

template<int BM, int BN, int WN, int STAGES>
void launch_fused_moe_kernel_up_down_acc(
        const fp8* x,
        const float* x_scale,
        fp8* w, const float* w_scale,
        fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const float* topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        int num_experts,
        int sorted_num,
        int block_m,
        float scaling_factor,
        cudaStream_t stream
        )
{
    constexpr int BK = 128;
    constexpr int PRODUCER_THREADS = 128;
    dim3 dimBlock(32*WN + PRODUCER_THREADS, 1, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*WN)), std::ceil((float)sorted_num/(block_m)), 1);

    size_t sMemSize = std::max(sizeof(smem_up<STAGES, WN, BM, BK, BN>),
                               sizeof(smem_down<STAGES, WN, BM, BK, BN>));

    gpuErrchk(cudaFuncSetAttribute(
                fused_moe_w8a8_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    CUtensorMap tensor_map_w{};
    {
        constexpr uint32_t rank = 2;
        uint64_t size[rank] = {static_cast<uint64_t>(K), static_cast<uint64_t>(N*num_experts)};
        uint64_t stride[rank] = {static_cast<uint64_t>(K) * sizeof(fp8)};
        uint32_t box_size[rank] = {BK, BN*WN};
        uint32_t elem_stride[rank] = {1, 1};
        auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
        cuErrchk(cuTensorMapEncodeTiled(
                &tensor_map_w,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                rank, w, size, stride, box_size, elem_stride,
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                ));
    }

    constexpr int BK2 = WN*BN/2;
    constexpr int BN2 = BK*2;
    const int K2 = N/2;
    const int N2 = K;
    CUtensorMap tensor_map_w2{};
    {
        constexpr uint32_t rank = 2;
        uint64_t size[rank] = {static_cast<uint64_t>(K2), static_cast<uint64_t>(N2*num_experts)};
        uint64_t stride[rank] = {static_cast<uint64_t>(K2) * sizeof(fp8)};
        uint32_t box_size[rank] = {BK2, BN2};
        uint32_t elem_stride[rank] = {1, 1};
        auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
        cuErrchk(cuTensorMapEncodeTiled(
                &tensor_map_w2,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                rank, w2, size, stride, box_size, elem_stride,
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                WN*BN == 256 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B :
                WN*BN == 128 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B :
                               CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                ));
    }

    fused_moe_w8a8_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS><<<dimGrid, dimBlock, sMemSize, stream>>>(
            x,
            x_scale,
            tensor_map_w,
            w_scale,
            tensor_map_w2,
            w2_scale,
            out,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            top_k,
            M,
            K,
            N,
            scaling_factor
            );
}

template<int BM, int BN, int WN>
void dispatch_stages(
        int stages,
        const fp8* x, const float* x_scale,
        fp8* w, const float* w_scale,
        fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const float* topk_weights,
        const int top_k,
        int M, int K, int N,
        int num_experts,
        int sorted_num, int block_m,
        float scaling_factor,
        cudaStream_t stream)
{
    switch(stages) {
        case 1: launch_fused_moe_kernel_up_down_acc<BM, BN, WN, 1>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 2: launch_fused_moe_kernel_up_down_acc<BM, BN, WN, 2>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 3: launch_fused_moe_kernel_up_down_acc<BM, BN, WN, 3>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 4: launch_fused_moe_kernel_up_down_acc<BM, BN, WN, 4>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 5: launch_fused_moe_kernel_up_down_acc<BM, BN, WN, 5>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        default:
            fprintf(stderr, "Unsupported stages value: %d. Supported values are: 1, 2, 3, 4, 5\n", stages);
    }
}

template<int BM>
void dispatch_bn_wn(
        int bn, int wn, int stages,
        const fp8* x, const float* x_scale,
        fp8* w, const float* w_scale,
        fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const float* topk_weights,
        const int top_k,
        int M, int K, int N,
        int num_experts,
        int sorted_num, int block_m,
        float scaling_factor,
        cudaStream_t stream)
{
    if (bn == 32 && wn == 8) {
        dispatch_stages<BM, 32, 8>(stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream);
    }
    else if (bn == 64 && wn == 4) {
        dispatch_stages<BM, 64, 4>(stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream);
    }
    else {
        fprintf(stderr, "Unsupported BN/WN pair: (%d, %d). Supported pairs are: (32, 8) and (64, 4)\n", bn, wn);
    }
}

void fused_moe_w8a8_wgmma_up_down_acc(
        const fp8* x,
        const float* x_scale,
        fp8* w, const float* w_scale,
        fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const float* topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        int num_experts,
        int sorted_num,
        int block_m,
        int block_n,
        int warp_n,
        int stages,
        float scaling_factor,
        cudaStream_t stream
        )
{
    switch(block_m) {
        case 8:   dispatch_bn_wn<8>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 16:  dispatch_bn_wn<16>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 24:  dispatch_bn_wn<24>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 32:  dispatch_bn_wn<32>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 40:  dispatch_bn_wn<40>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 48:  dispatch_bn_wn<48>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 56:  dispatch_bn_wn<56>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 64:  dispatch_bn_wn<64>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 72:  dispatch_bn_wn<72>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 80:  dispatch_bn_wn<80>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 88:  dispatch_bn_wn<88>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 96:  dispatch_bn_wn<96>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 104: dispatch_bn_wn<104>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 112: dispatch_bn_wn<112>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 120: dispatch_bn_wn<120>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        case 128: dispatch_bn_wn<128>(block_n, warp_n, stages, x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, num_experts, sorted_num, block_m, scaling_factor, stream); break;
        default:
                  fprintf(stderr, "Unsupported block_m value: %d. Supported values are: 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128\n", block_m);
                  break;
    }
}