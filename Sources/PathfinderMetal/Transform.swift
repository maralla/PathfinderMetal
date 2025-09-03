import Foundation

public struct Transform: Hashable {
    private var _matrix: F4
    private var _vector: F2

    var matrix: F4 {
        .init(_matrix.x, _matrix.y, _matrix.z, _matrix.w)
    }

    var vector: F2 {
        .init(_vector.x, _vector.y)
    }

    var m11: Float32 {
        matrix.x
    }

    var m21: Float32 {
        matrix.y
    }

    var m12: Float32 {
        matrix.z
    }

    var m22: Float32 {
        matrix.w
    }

    var m13: Float32 {
        vector.x
    }

    var m23: Float32 {
        vector.y
    }

    var isIdentity: Bool {
        let d = Transform()
        return self == d
    }

    public init() {
        self.init(scale: .init(repeating: 1.0))
    }

    public init(scale: Float32) {
        _matrix = .init(scale, 0.0, 0.0, scale)
        _vector = .zero
    }

    public init(scale: F2) {
        _matrix = .init(scale.x, 0.0, 0.0, scale.y)
        _vector = .zero
    }

    init(matrix: F4, vector: F2) {
        self._matrix = matrix
        self._vector = vector
    }

    init(translation: F2) {
        _matrix = .init(1, 0, 0, 1)
        _vector = translation
    }

    init(rotation theta: Float) {
        self.init(rotation_vector: UnitVector(theta: theta))
    }

    init(rotation_vector vector: UnitVector) {
        let m = F4(
            x: vector.x,
            y: vector.y,
            z: -vector.y,
            w: vector.x
        )

        _matrix = m
        self._vector = .zero
    }

    init(m11: Float32, m12: Float32, m13: Float32, m21: Float32, m22: Float32, m23: Float32) {
        _matrix = .init(m11, m21, m12, m22)
        _vector = .init(x: m13, y: m23)
    }

    public func extract_scale() -> F2 {
        let squared = _matrix * _matrix
        let value = squared.lowHalf + squared.highHalf
        return .init(sqrt(value.x), sqrt(value.y))
    }

    func inverse() -> Transform {
        let det = _matrix.x * _matrix.w - _matrix.z * _matrix.y
        let adjugate = _matrix * F4(1.0, -1.0, -1.0, 1.0)
        let matrix_inv = F4(repeating: 1.0 / det) * adjugate

        let halves = matrix_inv * F4(x: _vector.x, y: _vector.x, z: _vector.y, w: _vector.y)
        let vector_inv = -F2(x: halves.x + halves.z, y: halves.y + halves.w)

        return .init(matrix: matrix_inv, vector: vector_inv)
    }

    func translate(_ vector: F2) -> Transform {
        Transform(translation: vector) * self
    }

    func rotate(_ theta: Float) -> Transform {
        return Transform(rotation: theta) * self
    }

    static func * (transform: Transform, other: Transform) -> Transform {
        let xyxy = F4(
            x: transform._matrix.x,
            y: transform._matrix.y,
            z: transform._matrix.x,
            w: transform._matrix.y
        )
        let xxzz = F4(
            x: other._matrix.x,
            y: other._matrix.x,
            z: other._matrix.z,
            w: other._matrix.z
        )
        let zwzw = F4(
            x: transform._matrix.z,
            y: transform._matrix.w,
            z: transform._matrix.z,
            w: transform._matrix.w
        )
        let yyww = F4(
            x: other._matrix.y,
            y: other._matrix.y,
            z: other._matrix.w,
            w: other._matrix.w
        )

        let halves =
            transform._matrix
            * F4(other._vector.x, other._vector.x, other._vector.y, other._vector.y)

        return Transform(
            matrix: xyxy * xxzz + zwzw * yyww,
            vector: halves.lowHalf + halves.highHalf + transform._vector
        )
    }

    static func * (transform: Transform, other: F2) -> F2 {
        let xxyy = F4(other.x, other.x, other.y, other.y)
        let halves = transform._matrix * xxyy
        return (halves.lowHalf + halves.highHalf + transform._vector)
    }

    static func * (transform: Transform, rect: RectF) -> RectF {
        let (upper_left, upper_right) = (transform * rect.origin, transform * rect.upperRight)
        let (lower_left, lower_right) = (transform * rect.lowerLeft, transform * rect.lowerRight)

        let minPoint = F2(
            x: min(upper_left.x, min(upper_right.x, min(lower_left.x, lower_right.x))),
            y: min(upper_left.y, min(upper_right.y, min(lower_left.y, lower_right.y)))
        )

        let maxPoint = F2(
            x: max(upper_left.x, max(upper_right.x, max(lower_left.x, lower_right.x))),
            y: max(upper_left.y, max(upper_right.y, max(lower_left.y, lower_right.y)))
        )

        return .init(origin: minPoint, lowerRight: maxPoint)
    }

    static func * (lhs: Transform, rhs: LineSegment) -> LineSegment {
        return .init(from: (lhs * F2(rhs.from)).simd, to: (lhs * F2(rhs.to)).simd)
    }
}
