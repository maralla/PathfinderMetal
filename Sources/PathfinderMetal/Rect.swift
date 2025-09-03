import Foundation

struct RectI: Sendable {
    private let value: I4

    var rawValue: I4 { value }
    var originY: Int32 { value.y }
    var originX: Int32 { value.x }
    var origin: I2 { .init(x: value.x, y: value.y) }
    var lowerLeft: I2 { .init(value.x, value.w) }
    var upperRight: I2 { .init(value.z, value.y) }
    var lowerRight: I2 { .init(value.z, value.w) }

    var minX: Int32 { value.x }
    var minY: Int32 { value.y }
    var maxX: Int32 { value.z }
    var maxY: Int32 { value.w }

    var size: I2 { .init(x: value.z - value.x, y: value.w - value.y) }

    var f32: RectF { .init(value) }
    var width: Int32 { value.z - value.x }
    var height: Int32 { value.w - value.y }
    var area: Int32 { width * height }

    init(rawValue: I4) {
        value = rawValue
    }

    init(origin: I2, lower_right: I2) {
        value = .init(x: origin.x, y: origin.y, z: lower_right.x, w: lower_right.y)
    }

    init(origin: I2, size: I2) {
        value = .init(x: origin.x, y: origin.y, z: origin.x + size.x, w: origin.y + size.y)
    }

    static let zero = RectI(rawValue: .zero)

    func contract(_ amount: I2) -> RectI {
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

    var rawValue: F4 { value }
    var originY: Float32 { value.y }
    var originX: Float32 { value.x }

    var origin: F2 { .init(value.x, value.y) }
    var lowerLeft: F2 { .init(value.x, value.w) }
    var upperRight: F2 { .init(value.z, value.y) }
    var lowerRight: F2 { .init(value.z, value.w) }

    var minX: Float32 { value.x }
    var minY: Float32 { value.y }
    var maxX: Float32 { value.z }
    var maxY: Float32 { value.w }
    var size: F2 { .init(value.z - value.x, value.w - value.y) }
    var width: Float32 { value.z - value.x }
    var height: Float32 { value.w - value.y }
    var area: Float32 { width * height }
    var center: F2 {
        let width = value.z - value.x
        let height = value.w - value.y
        return .init(value.x + width / 2, value.y + height / 2)
    }

    var i32: RectI {
        .init(rawValue: I4(value))
    }

    init(rawValue: F4) {
        self.value = rawValue
    }

    init(origin: F2, lowerRight: F2) {
        value = .init(lowHalf: origin, highHalf: lowerRight)
    }

    static let zero = RectF(rawValue: .zero)

    init(origin: F2, size: F2) {
        value = F4(
            lowHalf: .init(origin.x, origin.y),
            highHalf: .init(origin.x + size.x, origin.y + size.y)
        )
    }

    init(_ i4: I4) {
        self.init(
            origin: .init(x: Float(i4.x), y: Float(i4.y)),
            lowerRight: .init(x: Float(i4.z), y: Float(i4.w))
        )
    }

    func contract(_ amount: F2) -> RectF {
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

    func unionPoint(_ point: F2) -> RectF {
        return .init(
            origin: F2(x: min(value.x, point.x), y: min(value.y, point.y)),
            lowerRight: F2(x: max(value.z, point.x), y: max(value.w, point.y))
        )
    }

    func unionRect(_ other: RectF) -> RectF {
        .init(
            origin: F2(x: min(value.x, other.value.x), y: min(value.y, other.value.y)),
            lowerRight: F2(x: max(value.z, other.value.z), y: max(value.w, other.value.w))
        )
    }

    func unionRect(newPoint: F2, first: Bool) -> RectF {
        if first {
            return .init(origin: newPoint, lowerRight: newPoint)
        }

        return unionPoint(newPoint)
    }

    func dilate(_ amount: F2) -> RectF {
        return .init(
            rawValue: .init(
                value.x - amount.x,
                value.y - amount.y,
                value.z + amount.x,
                value.w + amount.y
            )
        )
    }

    func dilate(_ amount: Float32) -> RectF {
        return dilate(.init(amount, amount))
    }

    static func * (rect: RectF, factor: F2) -> RectF {
        .init(rawValue: rect.value * F4(factor.x, factor.y, factor.x, factor.y))
    }
}
