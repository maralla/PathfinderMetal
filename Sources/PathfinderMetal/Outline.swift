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
