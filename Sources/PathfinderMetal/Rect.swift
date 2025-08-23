import simd

struct PFRect<T: SIMDScalar> {
    let value: SIMD4<T>
}

extension PFRect<Int32> {
    static let zero = PFRect<Int32>(rawValue: .zero)

    init(origin: SIMD2<Int32>, size: SIMD2<Int32>) {
        value = SIMD4(lowHalf: origin, highHalf: origin &+ size)
    }

    var size: SIMD2<Int32> {
        value.highHalf &- value.lowHalf
    }

    var f32: PFRect<Float32> {
        .init(rawValue: SIMD4<Float32>(value))
    }

    var width: Int32 { value.z - value.x }
    var height: Int32 { value.w - value.y }
    var area: Int32 { width * height }

    func contract(_ amount: SIMD2<Int32>) -> PFRect<Int32> {
        .init(origin: value.lowHalf &+ amount, lower_right: value.highHalf &- amount)
    }

    func intersects(_ other: PFRect<Int32>) -> Bool {
        let leftVec = SIMD4<Int32>(value.x, value.y, other.value.x, other.value.y)
        let rightVec = SIMD4<Int32>(other.value.z, other.value.w, value.z, value.w)

        // Component-wise less-than comparison
        let comparison = leftVec .< rightVec
        return comparison[0] && comparison[1] && comparison[2] && comparison[3]
    }
}

extension PFRect<Float32> {
    static let zero = PFRect<Float32>(rawValue: .zero)

    init(origin: SIMD2<Float32>, size: SIMD2<Float32>) {
        value = SIMD4(lowHalf: origin, highHalf: origin + size)
    }

    var size: SIMD2<Float32> {
        value.highHalf - value.lowHalf
    }

    var width: Float32 { value.z - value.x }
    var height: Float32 { value.w - value.y }
    var area: Float32 { width * height }

    var i32: PFRect<Int32> {
        .init(rawValue: SIMD4<Int32>(value))
    }

    func contract(_ amount: SIMD2<Float32>) -> PFRect<Float32> {
        .init(origin: value.lowHalf + amount, lower_right: value.highHalf - amount)
    }

    var center: SIMD2<Float32> {
        value.lowHalf + self.size * 0.5
    }

    func intersects(_ other: PFRect<Float32>) -> Bool {
        let leftVec = SIMD4<Float32>(value.x, value.y, other.value.x, other.value.y)
        let rightVec = SIMD4<Float32>(other.value.z, other.value.w, value.z, value.w)

        // Component-wise less-than comparison
        let comparison = leftVec .< rightVec
        return comparison[0] && comparison[1] && comparison[2] && comparison[3]
    }

    func intersection(_ other: PFRect<Float32>) -> PFRect<Float32>? {
        if !intersects(other) {
            return nil
        }

        return .init(
            origin: simd.max(origin, other.origin),
            lower_right: simd.min(lowerRight, other.lowerRight)
        )
    }

    func round_out() -> PFRect<Float32> {
        return .init(origin: simd.floor(origin), lower_right: simd.ceil(lowerRight))
    }

    func union_point(_ point: SIMD2<Float32>) -> PFRect<Float32> {
        return .init(origin: min(origin, point), lower_right: max(lowerRight, point))
    }

    func unionRect(_ other: PFRect<Float32>) -> PFRect<Float32> {
        .init(
            origin: min(origin, other.origin),  // Take minimum corner
            lower_right: max(lowerRight, other.lowerRight)  // Take maximum corner
        )
    }

    mutating func unionRect(newPoint: SIMD2<Float32>, first: Bool) {
        if first {
            self = .init(origin: newPoint, lower_right: newPoint)
        } else {
            self = .init(origin: simd.min(origin, newPoint), lower_right: simd.max(lowerRight, newPoint))
        }
    }

    func dilate(_ amount: SIMD2<Float32>) -> PFRect<Float32> {
        return .init(origin: origin - amount, lower_right: lowerRight + amount)
    }

    func dilate(_ amount: Float32) -> PFRect<Float32> {
        return .init(origin: origin - amount, lower_right: lowerRight + amount)
    }

    static func * (rect: PFRect<Float32>, factor: SIMD2<Float32>) -> PFRect<Float32> {
        .init(rawValue: rect.value * SIMD4<Float32>(factor.x, factor.y, factor.x, factor.y))
    }
}

extension PFRect {
    var originY: T { value.y }
    var originX: T { value.x }

    var origin: SIMD2<T> { value.lowHalf }

    var lowerLeft: SIMD2<T> { .init(value.x, value.w) }
    var upperRight: SIMD2<T> { .init(value.z, value.y) }
    var lowerRight: SIMD2<T> { value.highHalf }

    var minX: T { value[0] }
    var minY: T { value[1] }
    var maxX: T { value[2] }
    var maxY: T { value[3] }

    init(rawValue: SIMD4<T>) {
        self.value = rawValue
    }

    init(origin: SIMD2<T>, lower_right: SIMD2<T>) {
        value = .init(lowHalf: origin, highHalf: lower_right)
    }
}
