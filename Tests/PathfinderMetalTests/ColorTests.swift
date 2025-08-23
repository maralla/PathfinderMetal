import Testing
import simd

@testable import PathfinderMetal

struct ColorUTests {
    @Test("ColorU init and opacity flags")
    func initAndFlags() {
        let opaque = ColorU(r: 1, g: 2, b: 3, a: 255)
        #expect(opaque.isOpaque)
        #expect(!opaque.isFullyTransparent)

        let transparent = ColorU(r: 10, g: 20, b: 30, a: 0)
        #expect(!transparent.isOpaque)
        #expect(transparent.isFullyTransparent)
    }

    @Test("ColorU init from RGBA UInt32 packs channels correctly")
    func initFromRGBA() {
        let c = ColorU(rgba: 0x11223344)
        #expect(c.r == 0x11)
        #expect(c.g == 0x22)
        #expect(c.b == 0x33)
        #expect(c.a == 0x44)
    }

    @Test("ColorU to ColorF conversion divides by 255")
    func f32Conversion() {
        let c = ColorU(r: 255, g: 127, b: 0, a: 255)
        let f = c.f32
        #expect(abs(f.r - 1.0) < 0.0001)
        #expect(abs(f.g - (127.0 / 255.0)) < 0.0001)
        #expect(abs(f.b - 0.0) < 0.0001)
        #expect(abs(f.a - 1.0) < 0.0001)
    }

    @Test("ColorU static constants")
    func staticConstants() {
        #expect(ColorU.black == ColorU(r: 0, g: 0, b: 0, a: 255))
        #expect(ColorU.transparent_black == ColorU(r: 0, g: 0, b: 0, a: 0))
        #expect(ColorU.white == ColorU(r: 255, g: 255, b: 255, a: 255))
    }

    @Test("ColorU.toU8Array flattens bytes in RGBA order")
    func toU8ArrayFlattens() {
        let colors = [
            ColorU(r: 1, g: 2, b: 3, a: 4),
            ColorU(r: 5, g: 6, b: 7, a: 8),
        ]
        let bytes = ColorU.toU8Array(colors)
        #expect(bytes == [1, 2, 3, 4, 5, 6, 7, 8])
    }
}

struct ColorFTests {
    @Test("ColorF init stores channel values")
    func initStoresValues() {
        let c = ColorF(r: 0.1, g: 0.2, b: 0.3, a: 0.4)
        #expect(abs(c.r - 0.1) < 0.0001)
        #expect(abs(c.g - 0.2) < 0.0001)
        #expect(abs(c.b - 0.3) < 0.0001)
        #expect(abs(c.a - 0.4) < 0.0001)
    }

    @Test("ColorF init(simd:) maps elements correctly")
    func simdInit() {
        let v = SIMD4<Float32>(0.1, 0.2, 0.3, 0.4)
        let c = ColorF(simd: v)
        #expect(abs(c.r - 0.1) < 0.0001)
        #expect(abs(c.g - 0.2) < 0.0001)
        #expect(abs(c.b - 0.3) < 0.0001)
        #expect(abs(c.a - 0.4) < 0.0001)
    }

    @Test("ColorF.simd produces matching vector")
    func simdProperty() {
        let c = ColorF(r: 0.25, g: 0.5, b: 0.75, a: 1.0)
        let v = c.simd
        #expect(abs(v.x - 0.25) < 0.0001)
        #expect(abs(v.y - 0.5) < 0.0001)
        #expect(abs(v.z - 0.75) < 0.0001)
        #expect(abs(v.w - 1.0) < 0.0001)
    }

    @Test("ColorF.lerp linearly interpolates components")
    func lerpInterpolates() {
        let a = ColorF(r: 0.0, g: 0.0, b: 0.0, a: 0.0)
        let b = ColorF(r: 1.0, g: 1.0, b: 1.0, a: 1.0)
        let m = a.lerp(other: b, t: 0.25)
        #expect(abs(m.r - 0.25) < 0.0001)
        #expect(abs(m.g - 0.25) < 0.0001)
        #expect(abs(m.b - 0.25) < 0.0001)
        #expect(abs(m.a - 0.25) < 0.0001)
    }

    @Test("ColorF.u8 converts with truncation behavior")
    func u8ConversionTruncates() {
        // Note: Due to implementation, this performs truncation after scaling by 255.
        let c = ColorF(r: 0.5, g: 1.0, b: 0.0, a: 0.999)
        let u = c.u8
        #expect(u.r == 127)  // 0.5 * 255 = 127.5 -> truncates to 127
        #expect(u.g == 255)
        #expect(u.b == 0)
        #expect(u.a == 254)  // 0.999 * 255 ≈ 254.745 -> truncates to 254
    }

    @Test("ColorF.white has all ones")
    func staticWhite() {
        let w = ColorF.white
        #expect(abs(w.r - 1.0) < 0.0001)
        #expect(abs(w.g - 1.0) < 0.0001)
        #expect(abs(w.b - 1.0) < 0.0001)
        #expect(abs(w.a - 1.0) < 0.0001)
    }
}

struct GradientTests {
    @Test("Gradient Hashable equality for identical gradients")
    func hashableEquality() {
        let line = LineSegment(from: .init(0, 0), to: .init(1, 0))
        let stops = [
            Gradient.ColorStop(offset: 0.0, color: .black),
            Gradient.ColorStop(offset: 1.0, color: .white),
        ]
        let g1 = Gradient(geometry: .linear(line), stops: stops, wrap: .clamp)
        let g2 = Gradient(geometry: .linear(line), stops: stops, wrap: .clamp)
        #expect(g1 == g2)
        #expect(g1.hashValue == g2.hashValue)
    }

    @Test("Gradient.isOpaque detects transparency")
    func isOpaqueProperty() {
        let stopsOpaque = [
            Gradient.ColorStop(offset: 0.0, color: .black),
            Gradient.ColorStop(offset: 1.0, color: .white),
        ]
        let gOpaque = Gradient(geometry: .linear(.init(from: .zero, to: .init(1, 0))), stops: stopsOpaque, wrap: .clamp)
        #expect(gOpaque.isOpaque)

        let stopsTranslucent = [
            Gradient.ColorStop(offset: 0.0, color: .transparent_black),
            Gradient.ColorStop(offset: 1.0, color: .white),
        ]
        let gTranslucent = Gradient(
            geometry: .linear(.init(from: .zero, to: .init(1, 0))),
            stops: stopsTranslucent,
            wrap: .clamp
        )
        #expect(!gTranslucent.isOpaque)
    }

    @Test("Gradient.binarySearchBy returns matching index for orderedSame comparator")
    func binarySearchByFindsIndex() {
        let stops = [
            Gradient.ColorStop(offset: 0.0, color: .black),
            Gradient.ColorStop(offset: 0.25, color: .black),
            Gradient.ColorStop(offset: 0.5, color: .black),
            Gradient.ColorStop(offset: 0.75, color: .black),
            Gradient.ColorStop(offset: 1.0, color: .black),
        ]
        let g = Gradient(geometry: .linear(.init(from: .zero, to: .init(1, 0))), stops: stops, wrap: .clamp)
        let target: Float32 = 0.5
        let idx = g.binarySearchBy { stop in
            if stop.offset == target { return .orderedSame }
            return stop.offset < target ? .orderedAscending : .orderedDescending
        }
        #expect(idx == 2)
    }

    @Test("Gradient.sample clamps and interpolates between stops")
    func sampleClampsAndInterpolates() {
        let stops = [
            Gradient.ColorStop(offset: 0.0, color: ColorU(r: 255, g: 0, b: 0, a: 255)),  // red
            Gradient.ColorStop(offset: 1.0, color: ColorU(r: 0, g: 0, b: 255, a: 255)),  // blue
        ]
        let g = Gradient(geometry: .linear(.init(from: .zero, to: .init(1, 0))), stops: stops, wrap: .clamp)

        // Clamp below 0 and above 1
        #expect(g.sample(t: -1.0) == ColorU(r: 255, g: 0, b: 0, a: 255))
        #expect(g.sample(t: 2.0) == ColorU(r: 0, g: 0, b: 255, a: 255))

        // Midpoint interpolation. With truncation, 0.5 -> 127 for r and b.
        #expect(g.sample(t: 0.5) == ColorU(r: 127, g: 0, b: 127, a: 255))
    }

    @Test("Gradient.sample handles duplicate offsets (zero denom)")
    func sampleDuplicateOffsets() {
        let stops = [
            Gradient.ColorStop(offset: 0.0, color: ColorU(r: 10, g: 20, b: 30, a: 40)),
            Gradient.ColorStop(offset: 0.0, color: ColorU(r: 50, g: 60, b: 70, a: 80)),
            Gradient.ColorStop(offset: 1.0, color: ColorU(r: 0, g: 0, b: 0, a: 255)),
        ]
        let g = Gradient(geometry: .linear(.init(from: .zero, to: .init(1, 0))), stops: stops, wrap: .clamp)
        // For t == 0, denominator becomes 0 for the first segment; returns lower stop color.
        #expect(g.sample(t: 0.0) == ColorU(r: 50, g: 60, b: 70, a: 80))
        #expect(g.sample(t: 0.5) == ColorU(r: 25, g: 30, b: 35, a: 167))
    }

    @Test("Gradient.apply_transform updates geometry correctly for linear and radial")
    func applyTransform() {
        // Linear
        var gLinear = Gradient(
            geometry: .linear(.init(from: .init(0, 0), to: .init(1, 0))),
            stops: [Gradient.ColorStop(offset: 0.0, color: .black)],
            wrap: .clamp
        )
        let scale2 = Transform(scale: 2.0)
        gLinear.apply_transform(scale2)
        if case let .linear(line) = gLinear.geometry {
            #expect(abs(line.from.x - 0.0) < 0.0001)
            #expect(abs(line.from.y - 0.0) < 0.0001)
            #expect(abs(line.to.x - 2.0) < 0.0001)
            #expect(abs(line.to.y - 0.0) < 0.0001)
        } else {
            Issue.record("Expected linear geometry after transform")
        }

        // Radial
        let baseT = Transform(scale: SIMD2<Float32>(1.0, 2.0))
        var gRadial = Gradient(
            geometry: .radial(
                line: .init(from: .init(0, 0), to: .init(0, 0)),
                radii: SIMD2<Float32>(1.0, 2.0),
                transform: baseT
            ),
            stops: [Gradient.ColorStop(offset: 0.0, color: .black)],
            wrap: .clamp
        )
        let translate = Transform(translation: SIMD2<Float32>(3, 4))
        gRadial.apply_transform(translate)
        if case let .radial(_, radii, transform) = gRadial.geometry {
            #expect(radii == SIMD2<Float32>(1.0, 2.0))
            // Expect new transform = translate * baseT
            let expected = translate * baseT
            #expect(transform == expected)
        } else {
            Issue.record("Expected radial geometry after transform")
        }
    }
}
