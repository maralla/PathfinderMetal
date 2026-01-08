// Automatically generated helper shader to finalize alpha tiles without CPU readback.
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wunused-variable"

#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_atomic>

using namespace metal;

struct bZBuffer
{
    int iZBuffer[1];
};

// Output range buffer: [start, count]
struct bAlphaRange
{
    int2 iAlphaRange[1];
};

kernel void main0(const device bZBuffer& uZ [[buffer(0)]],
                  device bAlphaRange& uOut [[buffer(1)]])
{
    // Header layout matches existing pipeline: index 4 holds alpha tile count for this batch.
    const int count = uZ.iZBuffer[4];
    // Caller passes the absolute starting index via first element (already set) or 0.
    // We just patch count here to avoid races; single-thread kernel.
    int2 range = uOut.iAlphaRange[0];
    range.y = count;
    uOut.iAlphaRange[0] = range;
}

