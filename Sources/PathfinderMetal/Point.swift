import Foundation
import simd

struct UnitVector {
  var value: SIMD2<Float32>

  init(rawValue: SIMD2<Float32>) {
    self.value = rawValue
  }

  init(theta: Float) {
    self.init(rawValue: .init(cos(theta), sin(theta)))
  }

  /// Angle addition formula.
  func rotate_by(_ other: UnitVector) -> UnitVector {
    let xyyx = SIMD4<Float32>(value.x, value.y, value.y, value.x)
    let xyxy = SIMD4<Float32>(other.value.x, other.value.y, other.value.x, other.value.y)
    let products = xyyx * xyxy
    return .init(rawValue: .init(products[0] - products[1], products[2] + products[3]))
  }

  /// Angle subtraction formula.
  func rev_rotate_by(_ other: UnitVector) -> UnitVector {
    let xyyx = SIMD4<Float32>(value.x, value.y, value.y, value.x)
    let xyxy = SIMD4<Float32>(other.value.x, other.value.y, other.value.x, other.value.y)
    let products = xyyx * xyxy
    return .init(rawValue: .init(products[0] + products[1], products[2] - products[3]))
  }

  /// Half angle formula.
  func halve_angle() -> UnitVector {
    let x = self.value.x
    let term = SIMD2<Float32>(x, -x)

    let intermediate = SIMD2<Float32>(repeating: 0.5) * (SIMD2<Float32>(repeating: 1.0) + term)
    let maxed = max(intermediate, SIMD2<Float32>())
    return UnitVector(rawValue: .init(SIMD2<Float32>(sqrt(maxed.x), sqrt(maxed.y))))
  }
}
