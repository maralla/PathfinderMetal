import Testing
@testable import PathfinderMetal
import simd

struct PFRectInt32Tests {
    @Test("PFRect<Int32> zero and init origin+size")
    func zeroAndInit() {
        let z = PFRect<Int32>.zero
        #expect(z.minX == 0 && z.minY == 0 && z.maxX == 0 && z.maxY == 0)

        let r = PFRect<Int32>(origin: .init(10, 20), size: .init(30, 40))
        #expect(r.origin == SIMD2<Int32>(10, 20))
        #expect(r.size == SIMD2<Int32>(30, 40))
        #expect(r.width == 30)
        #expect(r.height == 40)
        #expect(r.area == 1200)
        #expect(r.lowerRight == SIMD2<Int32>(40, 60))
        #expect(r.upperRight == SIMD2<Int32>(40, 20))
        #expect(r.lowerLeft == SIMD2<Int32>(10, 60))
    }

    @Test("PFRect<Int32> contract and intersects")
    func contractAndIntersects() {
        let r = PFRect<Int32>(origin: .init(0, 0), size: .init(100, 50))
        let c = r.contract(.init(10, 5))
        #expect(c.origin == SIMD2<Int32>(10, 5))
        #expect(c.lowerRight == SIMD2<Int32>(90, 45))

        let a = PFRect<Int32>(origin: .init(0, 0), size: .init(10, 10))
        let b = PFRect<Int32>(origin: .init(5, 5), size: .init(10, 10))
        let disjoint = PFRect<Int32>(origin: .init(20, 20), size: .init(5, 5))
        #expect(a.intersects(b))
        #expect(!a.intersects(disjoint))
    }

    @Test("PFRect<Int32> f32 conversion")
    func f32Conversion() {
        let r = PFRect<Int32>(origin: .init(-1, 2), size: .init(3, 4))
        let f = r.f32
        #expect(abs(f.origin.x - Float(-1)) < 0.0001)
        #expect(abs(f.origin.y - Float(2)) < 0.0001)
        #expect(abs(f.lowerRight.x - Float(2)) < 0.0001)
        #expect(abs(f.lowerRight.y - Float(6)) < 0.0001)
    }
}

struct PFRectFloat32Tests {
    @Test("PFRect<Float32> zero and init origin+size")
    func zeroAndInit() {
        let z = PFRect<Float32>.zero
        #expect(z.minX == 0 && z.minY == 0 && z.maxX == 0 && z.maxY == 0)

        let r = PFRect<Float32>(origin: .init(1.5, 2.5), size: .init(3.25, 4.75))
        #expect(abs(r.origin.x - 1.5) < 0.0001)
        #expect(abs(r.origin.y - 2.5) < 0.0001)
        #expect(abs(r.size.x - 3.25) < 0.0001)
        #expect(abs(r.size.y - 4.75) < 0.0001)
        #expect(abs(r.width - 3.25) < 0.0001)
        #expect(abs(r.height - 4.75) < 0.0001)
        #expect(abs(r.area - (3.25 * 4.75)) < 0.0001)
        #expect(length(r.center - SIMD2<Float32>(1.5 + 1.625, 2.5 + 2.375)) < 0.0001)
    }

    @Test("PFRect<Float32> i32 conversion")
    func i32Conversion() {
        let r = PFRect<Float32>(origin: .init(-1.25, 2.75), size: .init(3.5, 4.25))
        let i = r.i32
        #expect(i.origin == SIMD2<Int32>(Int32(-1.25), Int32(2.75)))
        #expect(i.lowerRight == SIMD2<Int32>(Int32(2.25), Int32(7.0)))
    }

    @Test("PFRect<Float32> contract, dilate, union/round/intersection")
    func contractDilateAndUnions() {
        let r = PFRect<Float32>(origin: .init(0, 0), size: .init(10, 10))
        let c = r.contract(.init(1, 2))
        #expect(abs(c.origin.x - 1) < 0.0001)
        #expect(abs(c.origin.y - 2) < 0.0001)
        #expect(abs(c.lowerRight.x - 9) < 0.0001)
        #expect(abs(c.lowerRight.y - 8) < 0.0001)

        let d1 = r.dilate(.init(2, 3))
        #expect(d1.origin == SIMD2<Float32>(-2, -3))
        #expect(d1.lowerRight == SIMD2<Float32>(12, 13))

        let d2 = r.dilate(1)
        #expect(d2.origin == SIMD2<Float32>(-1, -1))
        #expect(d2.lowerRight == SIMD2<Float32>(11, 11))

        let a = PFRect<Float32>(origin: .init(0, 0), size: .init(5, 5))
        let b = PFRect<Float32>(origin: .init(3, 4), size: .init(10, 10))
        #expect(a.intersects(b))
        let inter = a.intersection(b)
        #expect(inter != nil)
        #expect(inter!.origin == SIMD2<Float32>(3, 4))
        #expect(inter!.lowerRight == SIMD2<Float32>(5, 5))

        let rounded = PFRect<Float32>(origin: .init(0.1, 0.9), size: .init(1.2, 1.2)).round_out()
        #expect(rounded.origin == SIMD2<Float32>(0, 0))
        #expect(rounded.lowerRight == SIMD2<Float32>(2, 3))

        let u = a.unionRect(b)
        #expect(u.origin == SIMD2<Float32>(0, 0))
        #expect(u.lowerRight == SIMD2<Float32>(13, 14))

        var bounds = PFRect<Float32>.zero
        bounds.unionRect(newPoint: SIMD2<Float32>(2, 3), first: true)
        #expect(bounds.origin == SIMD2<Float32>(2, 3))
        #expect(bounds.lowerRight == SIMD2<Float32>(2, 3))
        bounds.unionRect(newPoint: SIMD2<Float32>(-1, 5), first: false)
        #expect(bounds.origin == SIMD2<Float32>(-1, 3))
        #expect(bounds.lowerRight == SIMD2<Float32>(2, 5))
    }

    @Test("PFRect<Float32> scale operator *")
    func scaleOperator() {
        let r = PFRect<Float32>(origin: .init(1, 2), size: .init(3, 4))
        let s = r * SIMD2<Float32>(2, 3)
        #expect(s.origin == SIMD2<Float32>(2, 6))
        #expect(s.lowerRight == SIMD2<Float32>(8, 18))
    }
}


