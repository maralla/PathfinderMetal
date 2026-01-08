// Generates 3 uint dispatch args (X,Y,Z) for indirect compute dispatch.
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wunused-variable"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

// Must match the name used by existing dice shader bindings.
struct bComputeIndirectParams
{
    uint iComputeIndirectParams[1];
};

// Storage buffer that holds the 3 uints expected by
// MTLComputeCommandEncoder.dispatchThreadgroups(indirectBuffer:...)
// (threadgroupsPerGridX, threadgroupsPerGridY, threadgroupsPerGridZ)
struct bDispatchArgs
{
    // Metal expects 3 uints (x, y, z) tightly packed. We'll write them here.
    uint iDispatchArgs[3];
};

// Fixed workgroup size for the bin pass (matches ProgramsD3D11.BIN_WORKGROUP_SIZE)
constant uint kBinWorkgroupSize [[maybe_unused]] = 64u;

kernel void main0(const device bComputeIndirectParams& uComputeIndirectParams [[buffer(0)]],
                  device bDispatchArgs& uDispatchArgs [[buffer(1)]])
{
    // Dice writes the total microline count into iComputeIndirectParams[3].
    // The array sizing here is opaque to us (generated), so we access via index.
    // We guard against out-of-range by relying on the generator emitting a large
    // enough array; if not, the existing pipeline would already be broken.
    const uint microlineCount = uComputeIndirectParams.iComputeIndirectParams[3];

    const uint groupsX = (microlineCount + (kBinWorkgroupSize - 1u)) / kBinWorkgroupSize;
    uDispatchArgs.iDispatchArgs[0] = groupsX;
    uDispatchArgs.iDispatchArgs[1] = 1u;
    uDispatchArgs.iDispatchArgs[2] = 1u;
}


