import simd

struct RectI: Sendable {
    let value: SIMD4<Int32>

    var originY: Int32 { value.y }
    var originX: Int32 { value.x }

    var origin: SIMD2<Int32> { value.lowHalf }

    var lowerLeft: SIMD2<Int32> { .init(value.x, value.w) }
    var upperRight: SIMD2<Int32> { .init(value.z, value.y) }
    var lowerRight: SIMD2<Int32> { value.highHalf }

    var minX: Int32 { value[0] }
    var minY: Int32 { value[1] }
    var maxX: Int32 { value[2] }
    var maxY: Int32 { value[3] }

    init(rawValue: SIMD4<Int32>) {
        self.value = rawValue
    }

    init(origin: SIMD2<Int32>, lower_right: SIMD2<Int32>) {
        value = .init(lowHalf: origin, highHalf: lower_right)
    }

    static let zero = RectI(rawValue: .zero)

    init(origin: SIMD2<Int32>, size: SIMD2<Int32>) {
        value = SIMD4(lowHalf: origin, highHalf: origin &+ size)
    }

    var size: SIMD2<Int32> {
        value.highHalf &- value.lowHalf
    }

    var f32: RectF {
        .init(rawValue: SIMD4<Float32>(value))
    }

    var width: Int32 { value.z - value.x }
    var height: Int32 { value.w - value.y }
    var area: Int32 { width * height }

    func contract(_ amount: SIMD2<Int32>) -> RectI {
        .init(origin: value.lowHalf &+ amount, lower_right: value.highHalf &- amount)
    }

    func intersects(_ other: RectI) -> Bool {
        let leftVec = SIMD4<Int32>(value.x, value.y, other.value.x, other.value.y)
        let rightVec = SIMD4<Int32>(other.value.z, other.value.w, value.z, value.w)

        // Component-wise less-than comparison
        let comparison = leftVec .< rightVec
        return comparison[0] && comparison[1] && comparison[2] && comparison[3]
    }
}

struct RectF {
    let value: SIMD4<Float32>

    var originY: Float32 { value.y }
    var originX: Float32 { value.x }

    var origin: SIMD2<Float32> { value.lowHalf }

    var lowerLeft: SIMD2<Float32> { .init(value.x, value.w) }
    var upperRight: SIMD2<Float32> { .init(value.z, value.y) }
    var lowerRight: SIMD2<Float32> { value.highHalf }

    var minX: Float32 { value[0] }
    var minY: Float32 { value[1] }
    var maxX: Float32 { value[2] }
    var maxY: Float32 { value[3] }

    init(rawValue: SIMD4<Float32>) {
        self.value = rawValue
    }

    init(origin: SIMD2<Float32>, lower_right: SIMD2<Float32>) {
        value = .init(lowHalf: origin, highHalf: lower_right)
    }

    static let zero = RectF(rawValue: .zero)

    init(origin: SIMD2<Float32>, size: SIMD2<Float32>) {
        value = SIMD4(lowHalf: origin, highHalf: origin + size)
    }

    var size: SIMD2<Float32> {
        value.highHalf - value.lowHalf
    }

    var width: Float32 { value.z - value.x }
    var height: Float32 { value.w - value.y }
    var area: Float32 { width * height }

    var i32: RectI {
        .init(rawValue: SIMD4<Int32>(value))
    }

    func contract(_ amount: SIMD2<Float32>) -> RectF {
        .init(origin: value.lowHalf + amount, lower_right: value.highHalf - amount)
    }

    var center: SIMD2<Float32> {
        value.lowHalf + self.size * 0.5
    }

    func intersects(_ other: RectF) -> Bool {
        let leftVec = SIMD4<Float32>(value.x, value.y, other.value.x, other.value.y)
        let rightVec = SIMD4<Float32>(other.value.z, other.value.w, value.z, value.w)

        // Component-wise less-than comparison
        let comparison = leftVec .< rightVec
        return comparison[0] && comparison[1] && comparison[2] && comparison[3]
    }

    func intersection(_ other: RectF) -> RectF? {
        if !intersects(other) {
            return nil
        }

        return .init(
            origin: simd.max(origin, other.origin),
            lower_right: simd.min(lowerRight, other.lowerRight)
        )
    }

    func round_out() -> RectF {
        return .init(origin: simd.floor(origin), lower_right: simd.ceil(lowerRight))
    }

    @inline(__always)
    func union_point(_ point: SIMD2<Float32>) -> RectF {
        return .init(origin: min(origin, point), lower_right: max(lowerRight, point))
    }

    @inline(__always)
    func unionRect(_ other: RectF) -> RectF {
        .init(
            origin: min(origin, other.origin),  // Take minimum corner
            lower_right: max(lowerRight, other.lowerRight)  // Take maximum corner
        )
    }

    @inline(__always)
    mutating func unionRect(newPoint: SIMD2<Float32>, first: Bool) {
        if first {
            self = .init(origin: newPoint, lower_right: newPoint)
        } else {
            self = .init(origin: simd.min(origin, newPoint), lower_right: simd.max(lowerRight, newPoint))
        }
    }

    func dilate(_ amount: SIMD2<Float32>) -> RectF {
        return .init(origin: origin - amount, lower_right: lowerRight + amount)
    }

    func dilate(_ amount: Float32) -> RectF {
        return .init(origin: origin - amount, lower_right: lowerRight + amount)
    }

    static func * (rect: RectF, factor: SIMD2<Float32>) -> RectF {
        .init(rawValue: rect.value * SIMD4<Float32>(factor.x, factor.y, factor.x, factor.y))
    }
}
