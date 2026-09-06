#pragma once
#ifdef __SLANG__
#define DSSIM_UINT uint
#else
#include <cstddef>
#include <cstdint>
#include <type_traits>
#define DSSIM_UINT std::uint32_t
#endif
// Shared C++/Slang ABI. Pointers contain GPU virtual addresses.
struct ComputeRoot {
    DSSIM_UINT params[4];
    DSSIM_UINT* buffers[8];
    DSSIM_UINT yIndex;
    DSSIM_UINT uvIndex;
};
#ifndef __SLANG__
static_assert(sizeof(ComputeRoot) == 88);
static_assert(offsetof(ComputeRoot, buffers) == 16);
static_assert(offsetof(ComputeRoot, yIndex) == 80);
static_assert(std::is_trivially_copyable_v<ComputeRoot>);
#endif
#undef DSSIM_UINT
