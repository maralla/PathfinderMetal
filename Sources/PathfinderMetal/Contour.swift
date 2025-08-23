import simd

struct Contour {
    static let EPSILON: Float = 0.001
    static let SQRT_2: Float = 1.4142135623730951

    /// Flags that each point can have, indicating whether it is on-curve or whether it's a control
    /// point.
    struct PointFlags: OptionSet {
        let rawValue: UInt8

        /// This point is the first control point of a cubic Bézier curve or the only control point
        /// of a quadratic Bézier curve.
        static let controlPoint0 = PointFlags(rawValue: 0x1)

        /// This point is the second point of a quadratic Bézier curve.
        static let controlPoint1 = PointFlags(rawValue: 0x2)
    }

    /// Flags specifying what actions to take when pushing a segment onto a contour.
    struct PushSegmentFlags: OptionSet {
        let rawValue: UInt8

        /// The bounds should be updated.
        static let UPDATE_BOUNDS = PushSegmentFlags(rawValue: 0x01)
        /// The "from" point of the segment
        static let INCLUDE_FROM_POINT = PushSegmentFlags(rawValue: 0x02)
    }

    /// Flags that control the behavior of `Contour::iter()`.
    struct ContourIterFlags: OptionSet {
        let rawValue: UInt8

        /// Set to true to avoid iterating over the implicit line segment that joins the last point
        /// to the first point for closed contours.
        static let IGNORE_CLOSE_SEGMENT = ContourIterFlags(rawValue: 1)
    }

    struct ContourStrokeToFill {
        let input: Contour
        var output: Contour
        let radius: Float
        let join: Canvas.StrokeStyle.LineJoin
    }

    var points: [SIMD2<Float32>] = []
    var flags: [PointFlags] = []
    var bounds: PFRect<Float32> = .zero
    var closed: Bool = false

    init() {}

    init(capacity: Int) {
        points.reserveCapacity(capacity)
        flags.reserveCapacity(capacity)
    }

    init(rect: PFRect<Float32>) {
        self.init(capacity: 4)
        pushPoint(rect.origin, flags: PointFlags(), updateBounds: false)
        pushPoint(rect.upperRight, flags: PointFlags(), updateBounds: false)
        pushPoint(rect.lowerRight, flags: PointFlags(), updateBounds: false)
        pushPoint(rect.lowerLeft, flags: PointFlags(), updateBounds: false)
        close()
        bounds = rect
    }

    init(rect: PFRect<Float32>, radius: SIMD2<Float32>) {
        let sqrt2 = Float32(2.squareRoot())
        let QUARTER_ARC_CP_FROM_OUTSIDE: Float32 = (3.0 - 4.0 * (sqrt2 - 1.0)) / 3.0

        if radius == .zero {
            self.init(rect: rect)
            return
        }

        let radius = (rect.size * 0.5).min()
        let control_point_offset = radius * QUARTER_ARC_CP_FROM_OUTSIDE

        self.init(capacity: 8)

        // upper left corner
        let upperLeftP0 = rect.origin
        let upperLeftP1 = upperLeftP0 + control_point_offset
        let upperLeftP2 = upperLeftP0 + radius
        pushEndpoint(to: SIMD2<Float32>(upperLeftP0.x, upperLeftP2.y))
        pushCubic(
            ctrl0: SIMD2<Float32>(upperLeftP0.x, upperLeftP1.y),
            ctrl1: SIMD2<Float32>(upperLeftP1.x, upperLeftP0.y),
            to: SIMD2<Float32>(upperLeftP2.x, upperLeftP0.y)
        )

        // upper right
        let upperRightP0 = rect.upperRight
        let upperRightP1 = upperRightP0 + control_point_offset * SIMD2<Float32>(-1.0, 1.0)
        let upperRightP2 = upperRightP0 + radius * SIMD2<Float32>(-1.0, 1.0)
        pushEndpoint(to: SIMD2<Float32>(upperRightP2.x, upperRightP0.y))
        pushCubic(
            ctrl0: SIMD2<Float32>(upperRightP1.x, upperRightP0.y),
            ctrl1: SIMD2<Float32>(upperRightP0.x, upperRightP1.y),
            to: SIMD2<Float32>(upperRightP0.x, upperRightP2.y)
        )

        // lower right
        let lowerRightP0 = rect.lowerRight
        let lowerRightP1 = lowerRightP0 + control_point_offset * SIMD2<Float32>(-1.0, -1.0)
        let lowerRightP2 = lowerRightP0 + radius * SIMD2<Float32>(-1.0, -1.0)
        pushEndpoint(to: SIMD2<Float32>(lowerRightP0.x, lowerRightP2.y))
        pushCubic(
            ctrl0: SIMD2<Float32>(lowerRightP0.x, lowerRightP1.y),
            ctrl1: SIMD2<Float32>(lowerRightP1.x, lowerRightP0.y),
            to: SIMD2<Float32>(lowerRightP2.x, lowerRightP0.y)
        )

        // lower left
        let lowerLeftP0 = rect.lowerLeft
        let lowerLeftP1 = lowerLeftP0 + control_point_offset * SIMD2<Float32>(1.0, -1.0)
        let lowerLeftP2 = lowerLeftP0 + radius * SIMD2<Float32>(1.0, -1.0)
        pushEndpoint(to: SIMD2<Float32>(lowerLeftP2.x, lowerLeftP0.y))
        pushCubic(
            ctrl0: SIMD2<Float32>(lowerLeftP1.x, lowerLeftP0.y),
            ctrl1: SIMD2<Float32>(lowerLeftP0.x, lowerLeftP1.y),
            to: SIMD2<Float32>(lowerLeftP0.x, lowerLeftP2.y)
        )

        close()
    }

    var isEmpty: Bool {
        points.isEmpty
    }

    func last_position() -> SIMD2<Float32>? {
        return points.last
    }

    mutating func close() {
        closed = true
    }

    mutating func pushPoint(_ point: SIMD2<Float32>, flags: PointFlags, updateBounds: Bool) {
        if updateBounds {
            let first = isEmpty
            self.bounds.unionRect(newPoint: point, first: first)
        }

        self.points.append(point)
        self.flags.append(flags)
    }

    mutating func pushEndpoint(to: SIMD2<Float32>) {
        pushPoint(to, flags: PointFlags(), updateBounds: true)
    }

    mutating func pushQuadratic(ctrl: SIMD2<Float32>, to: SIMD2<Float32>) {
        pushPoint(ctrl, flags: .controlPoint0, updateBounds: true)
        pushPoint(to, flags: [], updateBounds: true)
    }

    mutating func pushCubic(ctrl0: SIMD2<Float32>, ctrl1: SIMD2<Float32>, to: SIMD2<Float32>) {
        self.pushPoint(ctrl0, flags: .controlPoint0, updateBounds: true)
        self.pushPoint(ctrl1, flags: .controlPoint1, updateBounds: true)
        self.pushPoint(to, flags: PointFlags(), updateBounds: true)
    }

    mutating func push_arc(
        _ transform: Transform,
        _ start_angle: Float,
        _ end_angle: Float,
        _ direction: PFPath.ArcDirection
    ) {
        if end_angle - start_angle >= Float.pi * 2.0 {
            push_ellipse(transform)
        } else {
            let start = SIMD2<Float>(cos(start_angle), sin(start_angle))
            let end = SIMD2<Float>(cos(end_angle), sin(end_angle))
            push_arc_from_unit_chord(transform, LineSegment(from: start, to: end), direction)
        }
    }

    mutating func push_arc_from_unit_chord(
        _ transform: Transform,
        _ chord: LineSegment,
        _ direction: PFPath.ArcDirection
    ) {
        var chord = chord
        var direction_transform = Transform()
        if direction == .ccw {
            chord *= SIMD2<Float>(1.0, -1.0)
            direction_transform = Transform(scale: SIMD2<Float>(1.0, -1.0))
        }

        var vector = UnitVector(rawValue: chord.from)
        let end_vector = UnitVector(rawValue: chord.to)

        for segment_index in 0..<4 {
            var sweep_vector = end_vector.rev_rotate_by(vector)
            let last = sweep_vector.value.x >= -Self.EPSILON && sweep_vector.value.y >= -Self.EPSILON

            var segment: Segment
            if !last {
                sweep_vector = UnitVector(rawValue: SIMD2<Float>(0.0, 1.0))
                segment = .quarter_circle_arc
            } else {
                segment = .init(arcFromCos: sweep_vector.value.x)
            }

            let half_sweep_vector = sweep_vector.halve_angle()
            let rotation = Transform(rotation_vector: half_sweep_vector.rotate_by(vector))
            segment = segment.transform(transform * direction_transform * rotation)

            var push_segment_flags: PushSegmentFlags = .UPDATE_BOUNDS
            if segment_index == 0 {
                push_segment_flags.insert(.INCLUDE_FROM_POINT)
            }
            push_segment(segment, push_segment_flags)

            if last {
                break
            }

            vector = vector.rotate_by(sweep_vector)
        }
    }

    mutating func push_ellipse(_ transform: Transform) {
        let segment = Segment.quarter_circle_arc
        var rotation: Transform

        push_segment(segment.transform(transform), [.UPDATE_BOUNDS, .INCLUDE_FROM_POINT])

        rotation = Transform(rotation_vector: UnitVector(rawValue: SIMD2<Float>(0.0, 1.0)))
        push_segment(segment.transform(transform * rotation), .UPDATE_BOUNDS)

        rotation = Transform(rotation_vector: UnitVector(rawValue: SIMD2<Float>(-1.0, 0.0)))
        push_segment(segment.transform(transform * rotation), .UPDATE_BOUNDS)

        rotation = Transform(rotation_vector: UnitVector(rawValue: SIMD2<Float>(0.0, -1.0)))
        push_segment(segment.transform(transform * rotation), .UPDATE_BOUNDS)
    }

    mutating func push_segment(_ segment: Segment, _ flags: PushSegmentFlags) {
        if segment.isNone {
            return
        }

        let update_bounds = flags.contains(.UPDATE_BOUNDS)
        pushPoint(segment.baseline.from, flags: [], updateBounds: update_bounds)

        if !segment.isLine {
            pushPoint(
                segment.ctrl.from,
                flags: .controlPoint0,
                updateBounds: update_bounds
            )
            if !segment.isQuadratic {
                pushPoint(
                    segment.ctrl.to,
                    flags: .controlPoint1,
                    updateBounds: update_bounds
                )
            }
        }

        pushPoint(segment.baseline.to, flags: [], updateBounds: update_bounds)
    }

    // Use this function to keep bounds up to date when mutating paths. See `Outline::transform()`
    // for an example of use.
    func updateBounds(bounds: inout PFRect<Float32>?) {
        bounds = bounds?.unionRect(self.bounds) ?? self.bounds
    }

    func flags_of(_ index: Int) -> PointFlags {
        flags[index]
    }

    func position_of(_ index: Int) -> SIMD2<Float32> {
        points[index]
    }

    func position_of_last(_ index: Int) -> SIMD2<Float32> {
        return points[points.count - index]
    }

    static func union_rect(
        _ bounds: inout PFRect<Float32>,
        _ new_point: SIMD2<Float32>,
        _ first: Bool
    ) {
        if first {
            bounds = .init(origin: new_point, lower_right: new_point)
        } else {
            bounds = bounds.union_point(new_point)
        }
    }

    func len() -> Int {
        return points.count
    }

    func iter(_ flags: ContourIterFlags) -> ContourIter {
        return ContourIter(
            contour: self,
            index: 1,
            flags: flags
        )
    }

    func point_is_endpoint(_ point_index: Int) -> Bool {
        return flags[point_index].intersection([.controlPoint0, .controlPoint1]).isEmpty
    }

    mutating func transform(_ transform: Transform) {
        if transform.isIdentity {
            return
        }

        for (point_index, point) in points.enumerated() {
            points[point_index] = transform * point
            Self.union_rect(&bounds, points[point_index], point_index == 0)
        }
    }

    func might_need_join(_ join: Canvas.StrokeStyle.LineJoin) -> Bool {
        if self.len() < 2 {
            return false
        } else {
            switch join {
            case .miter(_), .round:
                return true
            case .bevel:
                return false
            }
        }
    }

    mutating func add_join(
        _ distance: Float,
        _ join: Canvas.StrokeStyle.LineJoin,
        _ join_point: SIMD2<Float32>,
        _ next_tangent: LineSegment
    ) {
        let (p0, p1) = (self.position_of_last(2), self.position_of_last(1))
        let prev_tangent = LineSegment(from: p0, to: p1)

        if prev_tangent.square_length < Self.EPSILON || next_tangent.square_length < Self.EPSILON {
            return
        }

        switch join {
        case .bevel:
            break
        case .miter(let miter_limit):
            if let prev_tangent_t = prev_tangent.intersection_t(next_tangent) {
                if prev_tangent_t < -Self.EPSILON {
                    return
                }
                let miter_endpoint = prev_tangent.sample(prev_tangent_t)
                let threshold = miter_limit * distance

                if length_squared(miter_endpoint - join_point) > threshold * threshold {
                    return
                }

                self.pushEndpoint(to: miter_endpoint)
            }
        case .round:
            let scale = abs(distance)
            let transform = Transform(scale: scale).translate(join_point)
            let chord_from = normalize(prev_tangent.to - join_point)
            let chord_to = normalize(next_tangent.to - join_point)
            let chord = LineSegment(from: chord_from, to: chord_to)
            self.push_arc_from_unit_chord(transform, chord, .cw)
        }
    }
}

struct ContourIter {
    private let contour: Contour
    private var index: Int
    private let flags: Contour.ContourIterFlags

    init(contour: Contour, index: Int, flags: Contour.ContourIterFlags) {
        self.contour = contour
        self.index = index
        self.flags = flags
    }
}

extension ContourIter: IteratorProtocol {
    typealias Element = Segment

    mutating func next() -> Segment? {
        let include_close_segment = contour.closed && !flags.contains(.IGNORE_CLOSE_SEGMENT)
        if (index == contour.len() && !include_close_segment) || index == contour.len() + 1 {
            return nil
        }

        let point0_index = index - 1
        let point0 = contour.position_of(point0_index)
        if index == contour.len() {
            let point1 = contour.position_of(0)
            index += 1
            return Segment(line: .init(from: point0, to: point1))
        }

        let point1_index = index
        index += 1
        let point1 = contour.position_of(point1_index)
        if contour.point_is_endpoint(point1_index) {
            return Segment(line: .init(from: point0, to: point1))
        }

        let point2_index = index
        let point2 = contour.position_of(point2_index)
        index += 1
        if contour.point_is_endpoint(point2_index) {
            return Segment(quadratic: .init(from: point0, to: point2), ctrl: point1)
        }

        let point3_index = index
        let point3 = contour.position_of(point3_index)
        index += 1
        assert(contour.point_is_endpoint(point3_index))
        return Segment(cubic: .init(from: point0, to: point3), ctrl: .init(from: point1, to: point2))
    }
}

struct ContourDash {
    static let EPSILON: Float32 = 0.0001

    let input: Contour
    var output: Outline
    var state: DashState

    init(input: Contour, output: Outline, state: DashState) {
        self.input = input
        self.output = output
        self.state = state
    }

    mutating func dash() {
        var iterator = input.iter([])
        var queued_segment: Segment? = nil

        while true {
            if queued_segment == nil {
                guard let segment = iterator.next() else { break }
                queued_segment = segment
            }

            var current_segment = queued_segment!
            queued_segment = nil
            var distance = state.distance_left

            let t = current_segment.time_for_distance(distance)
            if t < 1.0 {
                let (prev_segment, next_segment) = current_segment.split(t)
                current_segment = prev_segment
                queued_segment = next_segment
            } else {
                distance = current_segment.arc_length()
            }

            if state.is_on() {
                state.output.push_segment(current_segment, [])
            }

            state.distance_left -= distance
            if state.distance_left < Self.EPSILON {
                if state.is_on() {
                    output.pushContour(state.output)
                    state.output = Contour()
                }

                state.current_dash_index += 1
                if state.current_dash_index == state.dashes.count {
                    state.current_dash_index = 0
                }

                state.distance_left = state.dashes[state.current_dash_index]
            }
        }
    }
}

extension Contour.ContourStrokeToFill {
    init(_ input: Contour, _ output: Contour, _ radius: Float, _ join: Canvas.StrokeStyle.LineJoin) {
        self.input = input
        self.output = output
        self.radius = radius
        self.join = join
    }

    mutating func offset_forward() {
        var index = 0

        var iterator = input.iter([])
        while true {
            guard let segment = iterator.next() else { break }

            // FIXME(pcwalton): We negate the radius here so that round end caps can be drawn
            // clockwise. Of course, we should just implement anticlockwise arcs to begin with...
            let join = index == 0 ? .bevel : self.join
            segment.offset(-self.radius, join, &self.output)

            index += 1
        }
    }

    mutating func offset_backward() {
        var segments: [Segment] = []
        var iterator = input.iter([])
        while true {
            guard let segment = iterator.next() else { break }
            segments.append(segment.reversed())
        }

        segments.reverse()
        for (segment_index, segment) in segments.enumerated() {
            // FIXME(pcwalton): We negate the radius here so that round end caps can be drawn
            // clockwise. Of course, we should just implement anticlockwise arcs to begin with...
            let join = segment_index == 0 ? .bevel : self.join
            segment.offset(-self.radius, join, &self.output)
        }
    }
}
