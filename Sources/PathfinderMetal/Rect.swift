import simd

struct RectI: Sendable {
    private let value: I4

    var rawValue: SIMD4<Int32> { .init(value.x, value.y, value.z, value.w) }
    var originY: Int32 { value.y }
    var originX: Int32 { value.x }
    var origin: SIMD2<Int32> { .init(x: value.x, y: value.y) }
    var lowerLeft: SIMD2<Int32> { .init(value.x, value.w) }
    var upperRight: SIMD2<Int32> { .init(value.z, value.y) }
    var lowerRight: SIMD2<Int32> { .init(value.z, value.w) }

    var minX: Int32 { value.x }
    var minY: Int32 { value.y }
    var maxX: Int32 { value.z }
    var maxY: Int32 { value.w }

    var size: SIMD2<Int32> { .init(x: value.z - value.x, y: value.w - value.y) }

    var f32: RectF { .init(value) }
    var width: Int32 { value.z - value.x }
    var height: Int32 { value.w - value.y }
    var area: Int32 { width * height }

    init(rawValue: I4) {
        value = rawValue
    }

    init(rawValue: SIMD4<Int32>) {
        self.init(rawValue: I4(rawValue))
    }

    init(origin: SIMD2<Int32>, lower_right: SIMD2<Int32>) {
        value = .init(x: origin.x, y: origin.y, z: lower_right.x, w: lower_right.y)
    }

    init(origin: SIMD2<Int32>, size: SIMD2<Int32>) {
        value = .init(x: origin.x, y: origin.y, z: origin.x + size.x, w: origin.y + size.y)
    }

    static let zero = RectI(rawValue: .zero)

    func contract(_ amount: SIMD2<Int32>) -> RectI {
        return .init(
            rawValue: I4(
                x: value.x + amount.x,
                y: value.y + amount.y,
                z: value.z - amount.x,
                w: value.w - amount.y
            )
        )
    }

    func intersects(_ other: RectI) -> Bool {
        return value.x < other.value.z
            && value.y < other.value.w
            && other.value.x < value.z
            && other.value.y < value.w
    }
}

struct RectF {
    private let value: F4

    var rawValue: SIMD4<Float32> { .init(value.x, value.y, value.z, value.w) }
    var originY: Float32 { value.y }
    var originX: Float32 { value.x }

    var origin: SIMD2<Float32> { .init(value.x, value.y) }
    var lowerLeft: SIMD2<Float32> { .init(value.x, value.w) }
    var upperRight: SIMD2<Float32> { .init(value.z, value.y) }
    var lowerRight: SIMD2<Float32> { .init(value.z, value.w) }

    var minX: Float32 { value.x }
    var minY: Float32 { value.y }
    var maxX: Float32 { value.z }
    var maxY: Float32 { value.w }

    init(rawValue: SIMD4<Float32>) {
        self.value = F4(rawValue)
    }

    init(rawValue: F4) {
        self.value = rawValue
    }

    init(origin: SIMD2<Float32>, lower_right: SIMD2<Float32>) {
        value = .init(lowHalf: F2(origin), highHalf: F2(lower_right))
    }

    init(origin: F2, lowerRight: F2) {
        value = .init(lowHalf: origin, highHalf: lowerRight)
    }

    static let zero = RectF(rawValue: .zero)

    init(origin: SIMD2<Float32>, size: SIMD2<Float32>) {
        value = F4(
            lowHalf: .init(origin.x, origin.y),
            highHalf: .init(origin.x + size.x, origin.y + size.y)
        )
    }

    init(_ i4: I4) {
        self.init(
            origin: .init(x: Float(i4.x), y: Float(i4.y)),
            lower_right: .init(x: Float(i4.z), y: Float(i4.w))
        )
    }

    var size: SIMD2<Float32> { .init(value.z - value.x, value.w - value.y) }
    var width: Float32 { value.z - value.x }
    var height: Float32 { value.w - value.y }
    var area: Float32 { width * height }
    var center: SIMD2<Float32> {
        let width = value.z - value.x
        let height = value.w - value.y
        return .init(value.x + width / 2, value.y + height / 2)
    }

    var i32: RectI {
        .init(rawValue: I4(value))
    }

    func contract(_ amount: SIMD2<Float32>) -> RectF {
        .init(
            rawValue: F4(
                x: value.x + amount.x,
                y: value.y + amount.y,
                z: value.z - amount.x,
                w: value.w - amount.y
            )
        )
    }

    func intersects(_ other: RectF) -> Bool {
        return value.x < other.value.z
            && value.y < other.value.w
            && other.value.x < value.z
            && other.value.y < value.w
    }

    func intersection(_ other: RectF) -> RectF? {
        if !intersects(other) {
            return nil
        }

        let origin = F2(
            x: max(value.x, other.value.x),
            y: max(value.y, other.value.y)
        )

        let lowerRight = F2(
            x: min(value.z, other.value.z),
            y: min(value.w, other.value.w)
        )

        return .init(origin: origin, lowerRight: lowerRight)
    }

    func round_out() -> RectF {
        return .init(
            origin: F2(x: floor(value.x), y: floor(value.y)),
            lowerRight: F2(x: ceil(value.z), y: ceil(value.w))
        )
    }

    @inline(__always)
    func union_point(_ point: SIMD2<Float32>) -> RectF {
        return .init(
            origin: F2(x: min(value.x, point.x), y: min(value.y, point.y)),
            lowerRight: F2(x: max(value.z, point.x), y: max(value.w, point.y))
        )
    }

    func unionPoint(_ point: F2) -> RectF {
        return .init(
            origin: F2(x: min(value.x, point.x), y: min(value.y, point.y)),
            lowerRight: F2(x: max(value.z, point.x), y: max(value.w, point.y))
        )
    }

    @inline(__always)
    func unionRect(_ other: RectF) -> RectF {
        .init(
            origin: F2(x: min(value.x, other.value.x), y: min(value.y, other.value.y)),
            lowerRight: F2(x: max(value.z, other.value.z), y: max(value.w, other.value.w))
        )
    }

    @inline(__always)
    mutating func unionRect(newPoint: SIMD2<Float32>, first: Bool) {
        if first {
            self = .init(
                origin: F2(newPoint.x, newPoint.y),
                lowerRight: F2(newPoint.x, newPoint.y)
            )
        } else {
            self = unionPoint(F2(newPoint))
        }
    }

    func dilate(_ amount: SIMD2<Float32>) -> RectF {
        return .init(origin: origin - amount, lower_right: lowerRight + amount)
    }

    func dilate(_ amount: Float32) -> RectF {
        return .init(origin: origin - amount, lower_right: lowerRight + amount)
    }

    static func * (rect: RectF, factor: SIMD2<Float32>) -> RectF {
        .init(rawValue: rect.value * F4(factor.x, factor.y, factor.x, factor.y))
    }
}
