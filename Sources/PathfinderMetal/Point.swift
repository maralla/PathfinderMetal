import Foundation

struct UnitVector {
    private var value: F2

    var x: Float32 { value.x }
    var y: Float32 { value.y }

    init(rawValue: SIMD2<Float32>) {
        self.value = F2(rawValue)
    }

    init(rawValue: F2) {
        self.value = rawValue
    }

    init(theta: Float) {
        self.init(rawValue: .init(cos(theta), sin(theta)))
    }

    /// Angle addition formula.
    func rotate_by(_ other: UnitVector) -> UnitVector {
        let xyyx = F4(value.x, value.y, value.y, value.x)
        let xyxy = F4(other.value.x, other.value.y, other.value.x, other.value.y)
        let products = xyyx * xyxy
        return .init(rawValue: F2(products.x - products.y, products.z + products.w))
    }

    /// Angle subtraction formula.
    func rev_rotate_by(_ other: UnitVector) -> UnitVector {
        let xyyx = F4(value.x, value.y, value.y, value.x)
        let xyxy = F4(other.value.x, other.value.y, other.value.x, other.value.y)
        let products = xyyx * xyxy
        return .init(rawValue: F2(products.x + products.y, products.z - products.w))
    }

    /// Half angle formula.
    func halve_angle() -> UnitVector {
        let x = self.value.x
        let term = F2(x, -x)

        let intermediate = F2(repeating: 0.5) * (F2(repeating: 1.0) + term)
        let maxed = F2(x: max(intermediate.x, 0), y: max(intermediate.y, 0))
        return UnitVector(rawValue: F2(sqrt(maxed.x), sqrt(maxed.y)))
    }
}

public struct F2: Hashable {
    public var x: Float32
    public var y: Float32

    nonisolated(unsafe) public static let zero = F2(x: 0, y: 0)

    public init(x: Float32, y: Float32) {
        self.x = x
        self.y = y
    }

    init(_ x: Float32, _ y: Float32) {
        self.x = x
        self.y = y
    }

    init(_ i2: I2) {
        self.x = Float(i2.x)
        self.y = Float(i2.y)
    }

    init(_ simd: SIMD2<Float32>) {
        self.init(simd.x, simd.y)
    }

    init(repeating v: Float32) {
        self.x = v
        self.y = v
    }

    var lengthSquared: Float32 {
        x * x + y * y
    }

    var normalized: F2 {
        let len = sqrt(lengthSquared)
        if len == 0 || len.isNaN {
            return self
        }

        return .init(x / len, y / len)
    }

    var simd: SIMD2<Float32> {
        .init(x, y)
    }

    func min() -> Float32 {
        Swift.min(x, y)
    }

    static func + (lhs: F2, rhs: F2) -> F2 {
        return .init(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
    }

    static func * (lhs: F2, rhs: F2) -> F2 {
        return .init(x: lhs.x * rhs.x, y: lhs.y * rhs.y)
    }

    static func / (lhs: F2, rhs: F2) -> F2 {
        return .init(x: lhs.x / rhs.x, y: lhs.y / rhs.y)
    }

    static func * (lhs: F2, rhs: Double) -> F2 {
        let v = Float32(rhs)
        return .init(x: lhs.x * v, y: lhs.y * v)
    }

    static func * (lhs: F2, rhs: Float32) -> F2 {
        let v = rhs
        return .init(x: lhs.x * v, y: lhs.y * v)
    }

    static func + (lhs: F2, rhs: Float32) -> F2 {
        let v = rhs
        return .init(x: lhs.x + v, y: lhs.y + v)
    }

    static func - (lhs: F2, rhs: F2) -> F2 {
        return .init(x: lhs.x - rhs.x, y: lhs.y - rhs.y)
    }

    static prefix func - (operand: F2) -> F2 {
        return .init(x: -operand.x, y: -operand.y)
    }
}

struct F4: Equatable, Hashable {
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    var lowHalf: F2 { .init(x, y) }
    var highHalf: F2 { .init(z, w) }

    var simd: SIMD4<Float32> {
        return .init(x, y, z, w)
    }

    static let zero = F4(x: 0, y: 0, z: 0, w: 0)

    init(lowHalf: F2, highHalf: F2) {
        x = lowHalf.x
        y = lowHalf.y
        z = highHalf.x
        w = highHalf.y
    }

    init(x: Float32, y: Float32, z: Float32, w: Float32) {
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    }

    init(_ x: Float32, _ y: Float32, _ z: Float32, _ w: Float32) {
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    }

    init(_ simd: SIMD4<Float32>) {
        self.init(simd.x, simd.y, simd.z, simd.w)
    }

    init(repeating v: Float32) {
        self.init(v, v, v, v)
    }

    static func * (lhs: F4, rhs: F4) -> F4 {
        return .init(x: lhs.x * rhs.x, y: lhs.y * rhs.y, z: lhs.z * rhs.z, w: lhs.w * rhs.w)
    }

    static func / (lhs: F4, rhs: F4) -> F4 {
        return .init(x: lhs.x / rhs.x, y: lhs.y / rhs.y, z: lhs.z / rhs.z, w: lhs.w / rhs.w)
    }

    static func - (lhs: F4, rhs: F4) -> F4 {
        return .init(x: lhs.x - rhs.x, y: lhs.y - rhs.y, z: lhs.z - rhs.z, w: lhs.w - rhs.w)
    }

    static func + (lhs: F4, rhs: F4) -> F4 {
        return .init(x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z, w: lhs.w + rhs.w)
    }
}

struct I2: Equatable, Hashable {
    var x: Int32
    var y: Int32

    var simd: SIMD2<Int32> {
        .init(x, y)
    }

    init(x: Int32, y: Int32) {
        self.x = x
        self.y = y
    }

    init(_ x: Int32, _ y: Int32) {
        self.x = x
        self.y = y
    }

    init(_ f2: F2) {
        self.x = Int32(f2.x)
        self.y = Int32(f2.y)
    }

    init(_ f2: SIMD2<Int32>) {
        self.x = Int32(f2.x)
        self.y = Int32(f2.y)
    }

    init(repeating v: Int32) {
        self.x = v
        self.y = v
    }

    static let zero = I2(x: 0, y: 0)

    static func + (lhs: I2, rhs: I2) -> I2 {
        return .init(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
    }

    static func * (lhs: I2, rhs: Int) -> I2 {
        let v = Int32(rhs)
        return .init(x: lhs.x * v, y: lhs.y * v)
    }

}

struct I4 {
    var x: Int32
    var y: Int32
    var z: Int32
    var w: Int32

    init(x: Int32, y: Int32, z: Int32, w: Int32) {
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    }

    init(_ simd: SIMD4<Int32>) {
        self.init(x: simd.x, y: simd.y, z: simd.z, w: simd.w)
    }

    init(_ f4: F4) {
        self.init(x: Int32(f4.x), y: Int32(f4.y), z: Int32(f4.z), w: Int32(f4.w))
    }

    static let zero = I4(x: 0, y: 0, z: 0, w: 0)
}
