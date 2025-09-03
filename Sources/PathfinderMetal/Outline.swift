import simd

struct Outline {
    struct OutlineStrokeToFill {
        let input: Outline
        var output: Outline
        let style: Canvas.StrokeStyle
    }

    var contours: [Contour] = []
    var bounds: RectF = .zero

    init() {
        contours = []
        bounds = .zero
    }

    init(capacity: Int) {
        contours.reserveCapacity(capacity)
    }

    init(segments: [Segment]) {
        var currentContour = Contour()

        for segment in segments {
            if segment.flags.contains(.FIRST_IN_SUBPATH) {
                if !currentContour.isEmpty {
                    let contour = currentContour
                    currentContour = .init()
                    contours.append(contour)
                }

                currentContour.pushPoint(segment.baseline.from, flags: .init(), updateBounds: true)
            }

            if segment.flags.contains(.CLOSES_SUBPATH) {
                if !currentContour.isEmpty {
                    currentContour.close()
                    let contour = currentContour
                    currentContour = .init()
                    pushContour(contour)
                }

                continue
            }

            if segment.isNone {
                continue
            }

            if !segment.isLine {
                currentContour.pushPoint(segment.ctrl.from, flags: .controlPoint0, updateBounds: true)
                if !segment.isQuadratic {
                    currentContour.pushPoint(
                        segment.ctrl.to,
                        flags: .controlPoint1,
                        updateBounds: true
                    )
                }
            }

            currentContour.pushPoint(segment.baseline.to, flags: .init(), updateBounds: true)
        }

        pushContour(currentContour)
    }

    init(rect: RectF) {
        pushContour(Contour(rect: rect))
    }

    init(rect: RectF, radius: SIMD2<Float32>) {
        pushContour(Contour(rect: rect, radius: radius))
    }

    /// Adds a new subpath to this outline.
    mutating func pushContour(_ contour: Contour) {
        if contour.isEmpty {
            return
        }

        if self.contours.isEmpty {
            self.bounds = contour.bounds
        } else {
            self.bounds = self.bounds.unionRect(contour.bounds)
        }

        self.contours.append(contour)
    }

    /// Removes the last subpath from this outline and returns it.
    mutating func popContour() -> Contour? {
        let lastContour = self.contours.popLast()

        var newBounds: RectF? = nil

        for contour in self.contours {
            contour.updateBounds(bounds: &newBounds)
        }

        self.bounds = newBounds ?? .zero

        return lastContour
    }

    mutating func transform(_ transform: Transform) {
        if transform.isIdentity {
            return
        }

        var new_bounds: RectF? = nil
        for i in 0..<contours.count {
            contours[i].transform(transform)
            contours[i].updateBounds(bounds: &new_bounds)
        }

        bounds = new_bounds ?? .zero
    }
}

struct OutlineDash {
    let input: Outline
    var output: Outline
    var state: DashState

    /// Creates a new outline dasher for the given stroke.
    ///
    /// Arguments:
    ///
    /// * `input`: The input stroke to be dashed. This must not yet been converted to a fill; i.e.
    ///   it is assumed that the stroke-to-fill conversion happens *after* this dashing process.
    ///
    /// * `dashes`: The list of dashes, specified as alternating pixel lengths of lines and gaps
    ///   that describe the pattern. See
    ///   <https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/setLineDash>.
    ///
    /// * `offset`: The line dash offset, or "phase". See
    ///   <https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/lineDashOffset>.
    init(_ input: Outline, _ dashes: [Float], _ offset: Float) {
        self.input = input
        self.output = Outline()
        self.state = DashState(dashes: dashes, offset: offset)
    }

    /// Performs the dashing operation.
    ///
    /// The results can be retrieved with the `into_outline()` method.
    mutating func dash() {
        for contour in input.contours {
            var dasher = ContourDash(input: contour, output: output, state: state)
            dasher.dash()
            self.output = dasher.output
            self.state = dasher.state
        }
    }

    /// Returns the resulting dashed outline.
    mutating func into_outline() -> Outline {
        if state.is_on() {
            output.pushContour(state.output)
        }

        return output
    }
}

struct DashState {
    var output: Contour
    let dashes: [Float]
    var current_dash_index: Int
    var distance_left: Float

    init(dashes: [Float], offset: Float) {
        self.output = Contour()
        self.dashes = dashes

        let total = dashes.reduce(0, +)
        var offset = offset.truncatingRemainder(dividingBy: total)

        var current_dash_index = 0
        while current_dash_index < dashes.count {
            let dash = dashes[current_dash_index]
            if offset < dash {
                break
            }
            offset -= dash
            current_dash_index += 1
        }

        self.current_dash_index = current_dash_index
        self.distance_left = offset
    }

    func is_on() -> Bool {
        return current_dash_index % 2 == 0
    }
}

extension Outline.OutlineStrokeToFill {
    /// Creates a new `OutlineStrokeToFill` object that will stroke the given outline with the
    /// given stroke style.
    init(_ input: Outline, style: Canvas.StrokeStyle) {
        self.input = input
        self.output = Outline()
        self.style = style
    }

    /// Performs the stroke operation.
    mutating func offset() {
        var new_contours: [Contour] = []
        for input in self.input.contours {
            let closed = input.closed
            var stroker = Contour.ContourStrokeToFill(
                input,
                Contour(),
                self.style.line_width * 0.5,
                self.style.line_join
            )

            stroker.offset_forward()
            if closed {
                self.push_stroked_contour(&new_contours, &stroker, true)
                stroker = Contour.ContourStrokeToFill(
                    input,
                    Contour(),
                    self.style.line_width * 0.5,
                    self.style.line_join
                )
            } else {
                self.add_cap(&stroker.output)
            }

            stroker.offset_backward()
            if !closed {
                self.add_cap(&stroker.output)
            }

            self.push_stroked_contour(&new_contours, &stroker, closed)
        }

        var new_bounds: RectF? = nil
        new_contours.forEach { contour in contour.updateBounds(bounds: &new_bounds) }

        self.output.contours = new_contours
        self.output.bounds = new_bounds ?? .init(rawValue: .zero)
    }

    /// Returns the resulting stroked outline. This should be called after `offset()`.
    func into_outline() -> Outline {
        return output
    }

    private mutating func push_stroked_contour(
        _ new_contours: inout [Contour],
        _ stroker: inout Contour.ContourStrokeToFill,
        _ closed: Bool
    ) {
        // Add join if necessary.
        if closed && stroker.output.might_need_join(self.style.line_join) {
            let (p1, p0) = (stroker.output.position_of(1), stroker.output.position_of(0))
            let final_segment = LineSegment(from: p1, to: p0)
            stroker.output.add_join(
                self.style.line_width * 0.5,
                self.style.line_join,
                stroker.input.position_of(0),
                final_segment
            )
        }

        stroker.output.closed = true
        new_contours.append(stroker.output)
    }

    private mutating func add_cap(_ contour: inout Contour) {
        if self.style.line_cap == .butt || contour.len() < 2 {
            return
        }

        let width = self.style.line_width
        let p1 = contour.position_of_last(1)

        // Determine the ending gradient.
        var p0: SIMD2<Float32>
        var p0_index = contour.len() - 2
        repeat {
            p0 = contour.position_of(p0_index)
            if simd.length_squared(p1 - p0) > Contour.EPSILON {
                break
            }
            if p0_index == 0 {
                return
            }
            p0_index -= 1
        } while true
        let gradient = simd.normalize(p1 - p0)

        switch self.style.line_cap {
        case .butt:
            fatalError("unreachable")

        case .square:
            let offset = gradient * (width * 0.5)

            let p2 = p1 + offset
            let p3 = p2 + SIMD2<Float32>(gradient.y, gradient.x) * SIMD2<Float32>(-width, width)
            let p4 = p3 - offset

            contour.pushEndpoint(to: p2)
            contour.pushEndpoint(to: p3)
            contour.pushEndpoint(to: p4)

        case .round:
            let scale = width * 0.5
            let offset = SIMD2<Float32>(gradient.y, gradient.x) * SIMD2<Float32>(-1.0, 1.0)
            let translation = p1 + offset * (width * 0.5)
            let transform = Transform(scale: scale).translate(F2(translation))
            let chord = LineSegment(from: -offset, to: offset)
            contour.push_arc_from_unit_chord(transform, chord, .cw)
        }
    }
}
