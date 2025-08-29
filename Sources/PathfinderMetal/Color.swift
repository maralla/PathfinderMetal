import Foundation

public struct Color<T: SIMDScalar>: Hashable {
    private var value: SIMD4<T>

    var r: T { value.x }
    var g: T { value.y }
    var b: T { value.z }
    var a: T {
        get { value.w }
        set { value.w = newValue }
    }
}

extension Color<UInt8>: Sendable {
    var isOpaque: Bool {
        value.w == 255
    }

    var isFullyTransparent: Bool {
        value.w == 0
    }

    public init(r: UInt8, g: UInt8, b: UInt8, a: UInt8) {
        value = .init(r, g, b, a)
    }

    init(rgba: UInt32) {
        self.init(
            r: UInt8(rgba >> 24),
            g: UInt8((rgba >> 16) & 0xff),
            b: UInt8((rgba >> 8) & 0xff),
            a: UInt8(rgba & 0xff)
        )
    }

    var f32: Color<Float> {
        let v = SIMD4<Float32>(value)
        let c = SIMD4<Float32>(repeating: 1.0 / 255.0)
        let result = v * c
        return .init(r: result.x, g: result.y, b: result.z, a: result.w)
    }

    public static let black = Color<UInt8>(r: 0, g: 0, b: 0, a: 255)
    public static let transparent_black = Color<UInt8>(rgba: 0)
    public static let white = Color<UInt8>(r: 255, g: 255, b: 255, a: 255)

    static func toU8Array(_ slice: [Color<UInt8>]) -> [UInt8] {
        return slice.withUnsafeBytes { bytes in
            Array(bytes)
        }
    }
}

extension Color<Float> {
    public init(r: Float, g: Float, b: Float, a: Float) {
        value = .init(r, g, b, a)
    }

    init(simd: SIMD4<Float32>) {
        self.init(r: simd.x, g: simd.y, b: simd.z, a: simd.w)
    }

    func lerp(other: Color<Float>, t: Float32) -> Color<Float> {
        .init(simd: value + (other.simd - value) * SIMD4<Float32>(repeating: t))
    }

    var simd: SIMD4<Float32> { value }

    var u8: Color<UInt8> {
        let v = SIMD4<Int32>(simd * SIMD4<Float32>(repeating: 255.0).rounded(.toNearestOrAwayFromZero))
        return .init(r: UInt8(v.x), g: UInt8(v.y), b: UInt8(v.z), a: UInt8(v.w))
    }

    var isFullyTransparent: Bool {
        value.w == 0
    }

    var isOpaque: Bool {
        value.w == 1
    }

    static var white: Color<Float> { .init(r: 1.0, g: 1.0, b: 1.0, a: 1.0) }
    static var black: Color<Float> { .init(r: 0.0, g: 0.0, b: 0.0, a: 1.0) }
    static let transparent_black = Color<Float>(r: 0.0, g: 0.0, b: 0.0, a: 0.0)
}

struct PFColorMatrix {
    var f1: SIMD4<Float32>
    var f2: SIMD4<Float32>
    var f3: SIMD4<Float32>
    var f4: SIMD4<Float32>
    var f5: SIMD4<Float32>
}

public struct Gradient {
    enum GradientGeometry {
        /// A linear gradient that follows a line.
        ///
        /// The line is in scene coordinates, not relative to the bounding box of the path.
        case linear(LineSegment)
        /// A radial gradient that radiates outward from a line connecting two circles (or from one
        /// circle).
        case radial(
            /// The line that connects the centers of the two circles. For single-circle radial
            /// gradients (the common case), this line has zero length, with start point and endpoint
            /// both at the circle's center point.
            ///
            /// This is in scene coordinates, not relative to the bounding box of the path.
            line: LineSegment,
            /// The radii of the two circles. The first value may be zero to start the gradient at the
            /// center of the circle.
            radii: SIMD2<Float32>,
            /// Transform from radial gradient space into screen space.
            ///
            /// Like `gradientTransform` in SVG. Note that this is the inverse of Cairo's gradient
            /// transform.
            transform: Transform
        )
    }

    enum GradientWrap {
        /// The area before the gradient is filled with the color of the first stop, and the area after
        /// the gradient is filled with the color of the last stop.
        case clamp
        /// The gradient repeats indefinitely.
        case `repeat`
    }

    struct ColorStop: Hashable {
        /// The offset of the color stop, between 0.0 and 1.0 inclusive. The value 0.0 represents the
        /// start of the gradient, and 1.0 represents the end.
        var offset: Float32
        /// The color of the gradient stop.
        var color: Color<Float32>
    }

    static let GRADIENT_TILE_LENGTH: UInt32 = 256

    /// Information specific to the type of gradient (linear or radial).
    var geometry: GradientGeometry
    var stops: [ColorStop]
    /// What should be rendered upon reaching the end of the color stops.
    var wrap: GradientWrap
}

extension Gradient: Hashable {
    public static func == (lhs: Gradient, rhs: Gradient) -> Bool {
        lhs.hashValue == rhs.hashValue
    }

    public func hash(into hasher: inout Hasher) {
        switch geometry {
        case .linear(let line):
            (0).hash(into: &hasher)
            line.hash(into: &hasher)
        case .radial(let line, let radii, let transform):
            (1).hash(into: &hasher)
            line.hash(into: &hasher)
            radii.hash(into: &hasher)
            transform.hash(into: &hasher)
        }

        for stop in self.stops {
            stop.hash(into: &hasher)
        }
    }
}

extension Gradient {
    var isOpaque: Bool {
        stops.allSatisfy({ $0.color.isOpaque })
    }

    func binarySearchBy(_ compare: (ColorStop) -> ComparisonResult) -> Int {
        var left = 0
        var right = stops.count

        while left < right {
            let mid = left + (right - left) / 2
            let result = compare(stops[mid])

            switch result {
            case .orderedSame:
                return mid
            case .orderedAscending:
                left = mid + 1
            case .orderedDescending:
                right = mid
            }
        }

        return left
    }

    func sample(t: Float32) -> Color<Float32> {
        if self.stops.isEmpty {
            return .transparent_black
        }

        let t = min(1.0, max(0.0, t))

        let last_index = self.stops.count - 1

        let position = binarySearchBy({ stop in
            if stop.offset < t || stop.offset == 0.0 {
                return .orderedAscending
            }

            return .orderedDescending
        })

        let upper_index = min(position, last_index)
        let lower_index = if upper_index > 0 { upper_index - 1 } else { upper_index }

        let lower_stop = self.stops[lower_index]
        let upper_stop = self.stops[upper_index]

        let denom = upper_stop.offset - lower_stop.offset
        if denom == 0.0 {
            return lower_stop.color
        }

        let ratio = min(((t - lower_stop.offset) / denom), 1.0)
        return lower_stop.color.lerp(other: upper_stop.color, t: ratio)
    }

    mutating func apply_transform(_ new_transform: Transform) {
        if new_transform.isIdentity {
            return
        }

        switch geometry {
        case .linear(var line):
            line = new_transform * line
            geometry = .linear(line)
        case .radial(let line, let radii, var transform):
            transform = new_transform * transform
            geometry = .radial(line: line, radii: radii, transform: transform)
        }
    }
}
