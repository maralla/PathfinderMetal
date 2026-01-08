#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wunused-variable"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct bZBuffer
{
    int iZBuffer[1];
};

struct bDispatchArgs
{
    uint iDispatchArgs[3];
};

kernel void main0(const device bZBuffer& uZ [[buffer(0)]], device bDispatchArgs& uArgs [[buffer(1)]])
{
    const uint count = uint(uZ.iZBuffer[4]);
    const uint maxX = 1u << 15; // matches fill grid split
    const uint x = (count < maxX) ? count : maxX;
    const uint y = (count + (maxX - 1u)) / maxX;
    uArgs.iDispatchArgs[0] = x;
    uArgs.iDispatchArgs[1] = y;
    uArgs.iDispatchArgs[2] = 1u;
}
