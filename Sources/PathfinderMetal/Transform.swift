import simd

public struct Transform: Hashable {
  var matrix: SIMD4<Float32>
  var vector: SIMD2<Float32>

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
    matrix = .init(scale, 0.0, 0.0, scale)
    vector = .zero
  }

  public init(scale: SIMD2<Float32>) {
    matrix = .init(scale.x, 0.0, 0.0, scale.y)
    vector = .zero
  }

  init(matrix: SIMD4<Float32>, vector: SIMD2<Float32>) {
    self.matrix = matrix
    self.vector = vector
  }

  init(translation: SIMD2<Float32>) {
    matrix = .init(1, 0, 0, 1)
    vector = translation
  }

  init(rotation theta: Float) {
    self.init(rotation_vector: UnitVector(theta: theta))
  }

  init(rotation_vector vector: UnitVector) {
    let v = SIMD4<Float32>(vector.value.x, vector.value.y, vector.value.y, vector.value.x)
    let m = v * SIMD4<Float32>(1.0, 1.0, -1.0, 1.0)

    matrix = m
    self.vector = .zero
  }

  init(m11: Float32, m12: Float32, m13: Float32, m21: Float32, m22: Float32, m23: Float32) {
    matrix = .init(m11, m21, m12, m22)
    vector = .init(x: m13, y: m23)
  }

  public func extract_scale() -> SIMD2<Float32> {
    let squared = matrix * matrix
    let value = squared.lowHalf + squared.highHalf
    return .init(sqrt(value.x), sqrt(value.y))
  }

  func inverse() -> Transform {
    let det = matrix.x * matrix.w - matrix.z * matrix.y
    let adjugate = matrix * SIMD4<Float32>(1.0, -1.0, -1.0, 1.0)
    let matrix_inv = SIMD4<Float32>(repeating: 1.0 / det) * adjugate

    let halves = matrix_inv * SIMD4<Float32>(x: vector.x, y: vector.x, z: vector.y, w: vector.y)
    let vector_inv = -SIMD2<Float32>(x: halves.x + halves.z, y: halves.y + halves.w)

    return .init(matrix: matrix_inv, vector: vector_inv)
  }

  func translate(_ vector: SIMD2<Float32>) -> Transform {
    Transform(translation: vector) * self
  }

  func rotate(_ theta: Float) -> Transform {
    return Transform(rotation: theta) * self
  }

  static func * (transform: Transform, other: Transform) -> Transform {
    let lowHalf = transform.matrix.lowHalf
    let highHalf = transform.matrix.highHalf

    let xyxy = SIMD4<Float32>(
      x: transform.matrix.x, y: transform.matrix.y, z: transform.matrix.x, w: transform.matrix.y)
    let xxzz = SIMD4<Float32>(
      x: other.matrix.x, y: other.matrix.x, z: other.matrix.z, w: other.matrix.z)
    let zwzw = SIMD4<Float32>(
      x: transform.matrix.z, y: transform.matrix.w, z: transform.matrix.z, w: transform.matrix.w)
    let yyww = SIMD4<Float32>(
      x: other.matrix.y, y: other.matrix.y, z: other.matrix.w, w: other.matrix.w)

    let halves =
      transform.matrix
      * SIMD4<Float32>(other.vector.x, other.vector.x, other.vector.y, other.vector.y)

    return Transform(
      matrix: xyxy * xxzz + zwzw * yyww,
      vector: halves.lowHalf + halves.highHalf + transform.vector
    )
  }

  static func * (transform: Transform, other: SIMD2<Float32>) -> SIMD2<Float32> {
    let xxyy = SIMD4<Float32>(other.x, other.x, other.y, other.y)
    let halves = transform.matrix * xxyy
    return halves.lowHalf + halves.highHalf + transform.vector
  }

  static func * (transform: Transform, rect: PFRect<Float32>) -> PFRect<Float32> {
    let (upper_left, upper_right) = (transform * rect.origin, transform * rect.upperRight)
    let (lower_left, lower_right) = (transform * rect.lowerLeft, transform * rect.lowerRight)
    let min_point = simd.min(upper_left, simd.min(upper_right, simd.min(lower_left, lower_right)))
    let max_point = simd.max(upper_left, simd.max(upper_right, simd.max(lower_left, lower_right)))
    return .init(origin: min_point, lower_right: max_point)
  }

  static func * (lhs: Transform, rhs: LineSegment) -> LineSegment {
    return .init(from: lhs * rhs.from, to: lhs * rhs.to)
  }
}
