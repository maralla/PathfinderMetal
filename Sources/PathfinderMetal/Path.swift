import simd

struct PFPath {
    enum ArcDirection {
        /// Clockwise, starting from the +x axis.
        case cw
        /// Counterclockwise, starting from the +x axis.
        case ccw
    }

    var outline: Outline
    var current_contour: Contour

    init() {
        outline = Outline()
        current_contour = Contour()
    }

    mutating func close_path() {
        current_contour.close()
    }

    mutating func move_to(_ to: SIMD2<Float32>) {
        flush_current_contour()
        current_contour.pushEndpoint(to: to)
    }

    mutating func line_to(_ to: SIMD2<Float32>) {
        current_contour.pushEndpoint(to: to)
    }

    mutating func quadratic_curve_to(_ ctrl: SIMD2<Float32>, _ to: SIMD2<Float32>) {
        current_contour.pushQuadratic(ctrl: ctrl, to: to)
    }

    mutating func bezier_curve_to(
        _ ctrl0: SIMD2<Float32>,
        _ ctrl1: SIMD2<Float32>,
        _ to: SIMD2<Float32>
    ) {
        current_contour.pushCubic(ctrl0: ctrl0, ctrl1: ctrl1, to: to)
    }

    mutating func arc(
        _ center: SIMD2<Float32>,
        _ radius: Float,
        _ start_angle: Float,
        _ end_angle: Float,
        _ direction: ArcDirection
    ) {
        let transform = Transform(scale: radius).translate(center)
        current_contour.push_arc(transform, start_angle, end_angle, direction)
    }

    mutating func arc_to(_ ctrl: SIMD2<Float32>, _ to: SIMD2<Float32>, _ radius: Float) {
        // FIXME(pcwalton): What should we do if there's no initial point?
        let from = current_contour.last_position() ?? .zero
        let v0 = from - ctrl
        let v1 = to - ctrl
        let vu0 = simd.normalize(v0)
        let vu1 = simd.normalize(v1)
        let hypot = radius / sqrt(0.5 * (1.0 - simd.dot(vu0, vu1)))
        let bisector = vu0 + vu1
        let center = ctrl + bisector * (hypot / simd.length(bisector))

        let transform = Transform(scale: radius).translate(center)
        let chord = LineSegment(
            from: SIMD2<Float32>(vu0.y, vu0.x) * SIMD2<Float>(-1.0, 1.0),
            to: SIMD2<Float32>(vu1.y, vu1.x) * SIMD2<Float>(1.0, -1.0)
        )

        // FIXME(pcwalton): Is clockwise direction correct?
        current_contour.push_arc_from_unit_chord(transform, chord, .cw)
    }

    mutating func rect(_ rect: RectF) {
        flush_current_contour()
        current_contour.pushEndpoint(to: rect.origin.simd)
        current_contour.pushEndpoint(to: rect.upperRight.simd)
        current_contour.pushEndpoint(to: rect.lowerRight.simd)
        current_contour.pushEndpoint(to: rect.lowerLeft.simd)
        current_contour.close()
    }

    mutating func ellipse(
        _ center: SIMD2<Float32>,
        _ axes: SIMD2<Float32>,
        _ rotation: Float,
        _ start_angle: Float,
        _ end_angle: Float
    ) {
        flush_current_contour()

        let transform = Transform(scale: axes).rotate(rotation).translate(center)
        current_contour.push_arc(transform, start_angle, end_angle, .cw)

        if end_angle - start_angle >= 2.0 * Float.pi {
            current_contour.close()
        }
    }

    // https://html.spec.whatwg.org/multipage/canvas.html#dom-path2d-addpath
    mutating func add_path(_ path: PFPath, _ transform: Transform) {
        var path = path

        flush_current_contour()
        path.flush_current_contour()
        path.outline.transform(transform)
        let last_contour = path.outline.popContour()
        for contour in path.outline.contours {
            outline.pushContour(contour)
        }

        current_contour = last_contour ?? .init()
    }

    mutating func into_outline() -> Outline {
        flush_current_contour()
        return outline
    }

    private mutating func flush_current_contour() {
        if !current_contour.isEmpty {
            outline.pushContour(current_contour)
            current_contour = .init()
        }
    }
}
