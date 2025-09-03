import Foundation

struct Segment {
    enum SegmentKind: UInt8 {
        /// An invalid segment.
        case none
        /// A line segment.
        case line
        /// A quadratic Bézier curve.
        case quadratic
        /// A cubic Bézier curve.
        case cubic
    }

    struct SegmentFlags: OptionSet {
        let rawValue: UInt8

        /// This segment is the first one in the contour.
        static let FIRST_IN_SUBPATH = SegmentFlags(rawValue: 0x01)
        /// This segment is the closing segment of the contour (i.e. it returns back to the
        /// starting point).
        static let CLOSES_SUBPATH = SegmentFlags(rawValue: 0x02)
    }

    /// The start and end points of the curve.
    var baseline: LineSegment
    /// The control point or points.
    ///
    /// If this is a line (which can be determined by examining the segment kind), this field is
    /// ignored. If this is a quadratic Bézier curve, the start point of this line represents the
    /// control point, and the endpoint of this line is ignored. Otherwise, if this is a cubic
    /// Bézier curve, both the start and endpoints are used.
    var ctrl: LineSegment
    /// The type of segment this is: invalid, line, quadratic, or cubic Bézier curve.
    var kind: SegmentKind
    /// Various flags that describe information about this segment in a path.
    var flags: SegmentFlags

    var isNone: Bool {
        return kind == .none
    }

    var isLine: Bool {
        return kind == .line
    }

    var isQuadratic: Bool {
        return kind == .quadratic
    }
}

struct LineSegment: Hashable {
    var value: F4

    var from: F2 { .init(x: value.x, y: value.y) }
    var to: F2 { .init(x: value.z, y: value.w) }

    init() {
        self.init(rawValue: .zero)
    }

    init(rawValue: F4) {
        value = rawValue
    }

    init(from: F2, to: F2) {
        value = .init(lowHalf: from, highHalf: to)
    }

    var minX: Float {
        return min(from.x, to.x)
    }

    var maxX: Float {
        return max(from.x, to.x)
    }

    var minY: Float {
        return min(from.y, to.y)
    }

    var maxY: Float {
        return max(from.y, to.y)
    }

    var vector: F2 {
        return to - from
    }

    var square_length: Float32 {
        let (dx, dy) = (value.z - value.x, value.w - value.y)
        return dx * dx + dy * dy
    }

    var reversed: LineSegment {
        return .init(rawValue: .init(value.z, value.w, value.x, value.y))
    }

    func split(_ t: Float32) -> (LineSegment, LineSegment) {
        assert(t >= 0.0 && t <= 1.0)

        let from_from = F4(value.x, value.y, value.x, value.y)
        let to_to = F4(value.z, value.w, value.z, value.w)

        let d_d = to_to - from_from
        let mid_mid = from_from + d_d * F4(repeating: t)

        let left = F4(from_from.x, from_from.y, mid_mid.x, mid_mid.y)
        let right = F4(mid_mid.x, mid_mid.y, to_to.x, to_to.y)

        return (
            LineSegment(rawValue: left),
            LineSegment(rawValue: right)
        )
    }

    // Returns the left segment first, followed by the right segment.
    func split_at_x(_ x: Float32) -> (LineSegment, LineSegment) {
        let (min_part, max_part) = split(solve_t_for_x(x))
        if min_part.from.x < max_part.from.x {
            return (min_part, max_part)
        } else {
            return (max_part, min_part)
        }
    }

    // Returns the upper segment first, followed by the lower segment.
    func split_at_y(_ y: Float) -> (LineSegment, LineSegment) {
        let (min_part, max_part) = split(solve_t_for_y(y))

        // Make sure we compare `from_y` and `to_y` to properly handle the case in which one of the
        // two segments is zero-length.
        if min_part.from.y < max_part.to.y {
            return (min_part, max_part)
        } else {
            return (max_part, min_part)
        }
    }

    func solve_t_for_x(_ x: Float) -> Float {
        return (x - from.x) / (to.x - from.x)
    }

    func solve_t_for_y(_ y: Float) -> Float {
        return (y - from.y) / (to.y - from.y)
    }

    func solve_x_for_y(_ y: Float) -> Float {
        return lerp(from.x, to.x, solve_t_for_y(y))
    }

    func solve_y_for_x(_ x: Float) -> Float {
        return lerp(from.y, to.y, solve_t_for_x(x))
    }

    func sample(_ t: Float) -> F2 {
        return from + vector * t
    }

    /// Linear interpolation.
    func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float {
        return a + (b - a) * t
    }

    func is_zero_length() -> Bool {
        return vector == .zero
    }

    static func * (lhs: LineSegment, rhs: F2) -> LineSegment {
        return .init(rawValue: lhs.value * .init(rhs.x, rhs.y, rhs.x, rhs.y))
    }

    static func + (lhs: LineSegment, rhs: F2) -> LineSegment {
        return .init(rawValue: lhs.value + .init(rhs.x, rhs.y, rhs.x, rhs.y))
    }

    static func * (lhs: LineSegment, rhs: Float32) -> LineSegment {
        return .init(rawValue: lhs.value * F4(repeating: rhs))
    }

    static func *= (lhs: inout LineSegment, rhs: F2) {
        lhs = lhs * rhs
    }
}

extension Segment {
    init() {
        self.baseline = LineSegment()
        self.ctrl = LineSegment()
        self.kind = .none
        self.flags = []
    }

    /// Returns a segment representing a straight line.
    init(line: LineSegment) {
        self.baseline = line
        self.ctrl = LineSegment()
        self.kind = .line
        self.flags = []
    }

    /// Returns a segment representing a quadratic Bézier curve.
    init(quadratic baseline: LineSegment, ctrl: F2) {
        self.baseline = baseline
        self.ctrl = LineSegment(from: ctrl, to: .zero)
        self.kind = .quadratic
        self.flags = []
    }

    /// Returns a segment representing a cubic Bézier curve.
    init(cubic baseline: LineSegment, ctrl: LineSegment) {
        self.baseline = baseline
        self.ctrl = ctrl
        self.kind = .cubic
        self.flags = []
    }

    /// Approximates an unit-length arc with a cubic Bézier curve.
    ///
    /// The maximum supported sweep angle is π/2 (i.e. 90°).
    init(arc sweep_angle: Float) {
        self.init(arcFromCos: cos(sweep_angle))
    }

    /// Approximates an unit-length arc with a cubic Bézier curve, given the cosine of the sweep
    /// angle.
    ///
    /// The maximum supported sweep angle is π/2 (i.e. 90°).
    init(arcFromCos cos_sweep_angle: Float) {
        // Richard A. DeVeneza, "How to determine the control points of a Bézier curve that
        // approximates a small arc", 2004.
        //
        // https://www.tinaja.com/glib/bezcirc2.pdf
        if cos_sweep_angle >= 1.0 - Contour.EPSILON {
            self.init(line: LineSegment(from: F2(1.0, 0.0), to: F2(1.0, 0.0)))
            return
        }

        let term = F4(cos_sweep_angle, -cos_sweep_angle, cos_sweep_angle, -cos_sweep_angle)
        let signs = F4(1.0, -1.0, 1.0, 1.0)
        let intermediate = (F4(repeating: 1.0) + term) * F4(repeating: 0.5)
        let p3p0 =
            F4(
                sqrt(intermediate.x),
                sqrt(intermediate.y),
                sqrt(intermediate.z),
                sqrt(intermediate.w)
            )
            * signs
        let (p0x, p0y) = (p3p0.z, p3p0.w)
        let (p1x, p1y) = (4.0 - p0x, (1.0 - p0x) * (3.0 - p0x) / p0y)
        let p2p1 = F4(p1x, -p1y, p1x, p1y) * F4(repeating: 1.0 / 3.0)
        self.init(cubic: LineSegment(rawValue: p3p0), ctrl: LineSegment(rawValue: p2p1))
    }

    /// Returns a cubic Bézier segment that approximates a quarter of an arc, centered on the +x
    /// axis.
    static var quarter_circle_arc: Segment {
        let p0 = F2(repeating: Contour.SQRT_2 * 0.5)
        let p1 = F2(-Contour.SQRT_2 / 6.0 + 4.0 / 3.0, 7.0 * Contour.SQRT_2 / 6.0 - 4.0 / 3.0)
        let flip = F2(1.0, -1.0)
        let (p2, p3) = (p1 * flip, p0 * flip)
        return .init(cubic: LineSegment(from: p3, to: p0), ctrl: LineSegment(from: p2, to: p1))
    }

    /// If this segment is a line, returns it. In debug builds, panics otherwise.
    func as_line_segment() -> LineSegment {
        assert(is_line())
        return baseline
    }

    /// Returns true if this segment is invalid.
    func is_none() -> Bool {
        return kind == .none
    }

    /// Returns true if this segment represents a straight line.
    func is_line() -> Bool {
        return kind == .line
    }

    /// Returns true if this segment represents a quadratic Bézier curve.
    func is_quadratic() -> Bool {
        return kind == .quadratic
    }

    /// Returns true if this segment represents a cubic Bézier curve.
    func is_cubic() -> Bool {
        return kind == .cubic
    }

    /// If this segment is a cubic Bézier curve, returns it. In debug builds, panics otherwise.
    func as_cubic_segment() -> CubicSegment {
        assert(is_cubic())
        return CubicSegment(self)
    }

    /// If this segment is a quadratic Bézier curve, elevates it to a cubic Bézier curve and
    /// returns it. If this segment is a cubic Bézier curve, this method simply returns it.
    ///
    /// If this segment is neither a quadratic Bézier curve nor a cubic Bézier curve, this method
    /// returns an unspecified result.
    ///
    /// FIXME(pcwalton): Handle lines!
    // FIXME(pcwalton): We should basically never use this function.
    func to_cubic() -> Segment {
        if is_cubic() {
            return self
        }

        var new_segment = self
        let p1_2 = ctrl.from + ctrl.from
        new_segment.ctrl = LineSegment(from: baseline.from + p1_2, to: p1_2 + baseline.to) * (1.0 / 3.0)
        new_segment.kind = .cubic
        return new_segment
    }

    /// Returns this segment with endpoints and control points reversed.
    func reversed() -> Segment {
        return .init(
            baseline: baseline.reversed,
            ctrl: is_quadratic() ? ctrl : ctrl.reversed,
            kind: kind,
            flags: flags
        )
    }

    /// Returns true if this segment is smaller than an implementation-defined epsilon value.
    func is_tiny() -> Bool {
        let EPSILON: Float = 0.0001
        return baseline.square_length < EPSILON
    }

    /// Divides this segment into two at the given parametric t value, which must range from 0.0 to
    /// 1.0.
    ///
    /// This uses de Casteljau subdivision.
    func split(_ t: Float) -> (Segment, Segment) {
        // FIXME(pcwalton): Don't degree elevate!
        if is_line() {
            let (before, after) = as_line_segment().split(t)
            return (Segment(line: before), Segment(line: after))
        } else {
            return to_cubic().as_cubic_segment().split(t)
        }
    }

    /// Returns the position of the point on this line or curve with the given parametric t value,
    /// which must range from 0.0 to 1.0.
    ///
    /// If called on an invalid segment (`None` type), the result is unspecified.
    func sample(_ t: Float) -> F2 {
        // FIXME(pcwalton): Don't degree elevate!
        if is_line() {
            return as_line_segment().sample(t)
        } else {
            return to_cubic().as_cubic_segment().sample(t)
        }
    }

    /// Applies the given affine transform to this segment and returns it.
    func transform(_ transform: Transform) -> Segment {
        return .init(
            baseline: transform * baseline,
            ctrl: transform * ctrl,
            kind: kind,
            flags: flags
        )
    }

    /// Treats this point as a vector and calculates its length.
    func length(of value: F2) -> Float {
        let squared = value * value
        return sqrt(squared.x + squared.y)
    }

    func arc_length() -> Float {
        // FIXME(pcwalton)
        return length(of: baseline.vector)
    }

    func time_for_distance(_ distance: Float) -> Float {
        // FIXME(pcwalton)
        return distance / arc_length()
    }
}

/// A wrapper for a `Segment` that contains method specific to cubic Bézier curves.
struct CubicSegment {
    let segment: Segment

    init(_ segment: Segment) {
        self.segment = segment
    }

    /// Returns true if the maximum deviation of this curve from the straight line connecting its
    /// endpoints is less than `tolerance`.
    ///
    /// See Kaspar Fischer, "Piecewise Linear Approximation of Bézier Curves", 2000.
    func is_flat(_ tolerance: Float) -> Bool {
        var uv =
            F4(repeating: 3.0) * segment.ctrl.value
            - (segment.baseline.value + segment.baseline.value + segment.baseline.reversed.value)
        uv = uv * uv
        uv = F4(
            max(uv.x, uv.z),
            max(uv.y, uv.w),
            max(uv.z, uv.x),
            max(uv.w, uv.y)
        )

        return uv.x + uv.y <= 16.0 * tolerance * tolerance
    }

    /// Splits this cubic Bézier curve into two at the given parametric t value, which will be
    /// clamped to the range 0.0 to 1.0.
    ///
    /// This uses de Casteljau subdivision.
    func split(_ t: Float) -> (Segment, Segment) {
        let baseline0: LineSegment
        let ctrl0: LineSegment
        let baseline1: LineSegment
        let ctrl1: LineSegment

        if t <= 0.0 {
            let from = segment.baseline.from
            baseline0 = LineSegment(from: from, to: from)
            ctrl0 = LineSegment(from: from, to: from)
            baseline1 = segment.baseline
            ctrl1 = segment.ctrl
        } else if t >= 1.0 {
            let to = segment.baseline.to
            baseline0 = segment.baseline
            ctrl0 = segment.ctrl
            baseline1 = LineSegment(from: to, to: to)
            ctrl1 = LineSegment(from: to, to: to)
        } else {
            let tttt = F4(repeating: t)

            let (p0p3, p1p2) = (segment.baseline.value, segment.ctrl.value)
            let p0p1 = F4(p0p3.x, p0p3.y, p1p2.x, p1p2.y)

            // p01 = lerp(p0, p1, t), p12 = lerp(p1, p2, t), p23 = lerp(p2, p3, t)
            let p01p12 = p0p1 + tttt * (p1p2 - p0p1)
            let pxxp23 = p1p2 + tttt * (p0p3 - p1p2)
            let p12p23 = F4(p01p12.z, p01p12.w, pxxp23.z, pxxp23.w)

            // p012 = lerp(p01, p12, t), p123 = lerp(p12, p23, t)
            let p012p123 = p01p12 + tttt * (p12p23 - p01p12)
            let p123 = F4(p012p123.z, p012p123.w, p012p123.z, p012p123.w)

            // p0123 = lerp(p012, p123, t)
            let p0123 = p012p123 + tttt * (p123 - p012p123)

            baseline0 = LineSegment(rawValue: .init(p0p3.x, p0p3.y, p0123.x, p0123.y))
            ctrl0 = LineSegment(rawValue: .init(p01p12.x, p01p12.y, p012p123.x, p012p123.y))
            baseline1 = LineSegment(rawValue: .init(p0123.x, p0123.y, p0p3.z, p0p3.w))
            ctrl1 = LineSegment(rawValue: .init(p012p123.z, p012p123.w, p12p23.z, p12p23.w))
        }

        return (
            Segment(
                baseline: baseline0,
                ctrl: ctrl0,
                kind: .cubic,
                flags: segment.flags.intersection(.FIRST_IN_SUBPATH)
            ),
            Segment(
                baseline: baseline1,
                ctrl: ctrl1,
                kind: .cubic,
                flags: segment.flags.intersection(.CLOSES_SUBPATH)
            )
        )
    }

    /// A convenience method equivalent to `segment.split(t).0`.
    func split_before(_ t: Float) -> Segment {
        return split(t).0
    }

    /// A convenience method equivalent to `segment.split(t).1`.
    func split_after(_ t: Float) -> Segment {
        return split(t).1
    }

    /// Returns the position of the point on this curve at parametric time `t`, which will be
    /// clamped between 0.0 and 1.0.
    ///
    /// FIXME(pcwalton): Use Horner's method!
    func sample(_ t: Float) -> F2 {
        return split(t).0.baseline.to
    }

    /// Returns the left extent of this curve's axis-aligned bounding box.
    func min_x() -> Float {
        return min(segment.baseline.minX, segment.ctrl.minX)
    }

    /// Returns the top extent of this curve's axis-aligned bounding box.
    func min_y() -> Float {
        return min(segment.baseline.minY, segment.ctrl.minY)
    }

    /// Returns the right extent of this curve's axis-aligned bounding box.
    func max_x() -> Float {
        return max(segment.baseline.maxX, segment.ctrl.maxX)
    }

    /// Returns the bottom extent of this curve's axis-aligned bounding box.
    func max_y() -> Float {
        return max(segment.baseline.maxY, segment.ctrl.maxY)
    }
}
