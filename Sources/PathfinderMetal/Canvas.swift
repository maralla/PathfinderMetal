import Foundation
import Metal
import QuartzCore

public class Canvas {
    static let HAIRLINE_STROKE_WIDTH: Float32 = 0.0333

    enum PathOp {
        case fill
        case stroke
    }

    public enum LineCap {
        case butt
        case square
        case round
    }

    enum LineJoin {
        case miter, bevel, round
    }

    enum ImageSmoothingQuality {
        case low
        case medium
        case high
    }

    enum FillStyle {
        case color(ColorU)
        case gradient(Gradient)
        case pattern(Pattern)

        func into_paint() -> Paint {
            switch self {
            case .color(let color):
                return Paint(color: color)
            case .gradient(let gradient):
                return Paint(
                    baseColor: .white,
                    overlay: .init(
                        compositeOp: .srcIn,
                        contents: .gradient(gradient)
                    )
                )
            case .pattern(let pattern):
                return Paint(pattern: pattern)
            }
        }
    }

    struct StrokeStyle {
        public enum LineJoin {
            /// Connected segments are joined by extending their outside edges to connect at a single
            /// point, with the effect of filling an additional lozenge-shaped area. The Float value
            /// specifies the miter limit ratio.
            case miter(Float)
            /// Fills an additional triangular area between the common endpoint of connected segments and
            /// the separate outside rectangular corners of each segment.
            case bevel
            /// Rounds off the corners of a shape by filling an additional sector of disc centered at the
            /// common endpoint of connected segments. The radius for these rounded corners is equal to the
            /// line width.
            case round
        }

        /// The width of the stroke in scene units.
        var line_width: Float
        /// The shape of the ends of the stroke.
        var line_cap: LineCap
        /// The shape used to join two line segments where they meet.
        var line_join: LineJoin
    }

    enum CompositeOperation {
        case sourceOver, sourceIn, sourceOut, sourceAtop
        case destinationOver, destinationIn, destinationOut, destinationAtop
        case lighter, copy, xor, multiply, screen, overlay
        case darken, lighten, colorDodge, colorBurn, hardLight, softLight
        case difference, exclusion, hue, saturation, color, luminosity
    }

    struct State {
        var transform: Transform
        var line_width: Float
        var line_cap: LineCap
        var line_join: LineJoin
        var miter_limit: Float
        var line_dash: [Float]
        var line_dash_offset: Float
        var fill_paint: Paint
        var stroke_paint: Paint
        var shadow_color: ColorU
        var shadow_blur: Float
        var shadow_offset: SIMD2<Float32>
        var image_smoothing_enabled: Bool
        var image_smoothing_quality: ImageSmoothingQuality
        var global_alpha: Float
        var global_composite_operation: CompositeOperation
        var clip_path: UInt32?

        func resolve_stroke_style() -> StrokeStyle {
            return StrokeStyle(
                line_width: line_width,
                line_cap: line_cap,
                line_join: {
                    switch line_join {
                    case .miter:
                        return .miter(miter_limit)
                    case .bevel:
                        return .bevel
                    case .round:
                        return .round
                    }
                }()
            )
        }
    }

    private var scene: Scene
    private var current_state: State
    private var stateStack: [State]

    public init(size: SIMD2<Float32>) {
        scene = .init()
        scene.set_view_box(.init(origin: .zero, size: size))
        current_state = .init()
        stateStack = []
    }

    private var device: PFDevice!

    public func demo(device: MTLDevice, drawable: CAMetalDrawable, size: SIMD2<Int32>) {
        self.device = PFDevice(device, texture: drawable)
        //        var path = PFPath()
        //        path.rect(.init(origin: .init(50, 50), size: .init(500, 500)))
        //        fill_rect(.init(origin: .init(50, 50), size: .init(100, 100)))
        set_line_width(2)
        //        set_stroke_style(.color(.init(r: 255, g: 0, b: 0, a: 255)))
        //        stroke_rect(.init(origin: .init(50, 50), size: .init(100, 100)))

        var path = PFPath()
        path.move_to(.init(x: 520, y: 520))
        path.bezier_curve_to(.init(x: 560, y: 260), .init(x: 680, y: 260), .init(x: 720, y: 320))
        path.bezier_curve_to(.init(x: 760, y: 380), .init(x: 640, y: 420), .init(x: 600, y: 360))
        path.bezier_curve_to(.init(x: 560, y: 320), .init(x: 540, y: 360), .init(x: 520, y: 320))

        path.close_path()

        func addEllipse(_ cx: Float, _ cy: Float, _ rx: Float, _ ry: Float) -> PFPath {
            let kappa: Float = 0.5522847498307936  // 4*(sqrt(2)-1)/3
            let ox = rx * kappa
            let oy = ry * kappa
            var path = PFPath()
            path.move_to(.init(x: cx - rx, y: cy))
            path.bezier_curve_to(
                .init(x: cx - rx, y: cy - oy),
                .init(x: cx - ox, y: cy - ry),
                .init(x: cx, y: cy - ry)
            )
            path.bezier_curve_to(
                .init(x: cx + ox, y: cy - ry),
                .init(x: cx + rx, y: cy - oy),
                .init(x: cx + rx, y: cy)
            )
            path.bezier_curve_to(
                .init(x: cx + rx, y: cy + oy),
                .init(x: cx + ox, y: cy + ry),
                .init(x: cx, y: cy + ry)
            )
            path.bezier_curve_to(
                .init(x: cx - ox, y: cy + ry),
                .init(x: cx - rx, y: cy + oy),
                .init(x: cx - rx, y: cy)
            )
            path.close_path()
            return path
        }

        func curve1() -> PFPath {
            var path = PFPath()
            path.move_to(.init(x: 520, y: 140))
            path.quadratic_curve_to(.init(x: 640, y: 60), .init(x: 760, y: 140))
            path.quadratic_curve_to(.init(x: 640, y: 220), .init(x: 520, y: 140))
            return path
        }

        func makeStar(cx: Float, cy: Float, rOuter: Float, rInner: Float, points: Int) -> PFPath {
            var p = PFPath()
            let startAngle: Float = -.pi / 2
            let step = (.pi * 2) / Float(points)
            for i in 0..<(points * 2) {
                let r = (i % 2 == 0) ? rOuter : rInner
                let ang = startAngle + Float(i) * (step / 2)
                let x = cx + cos(ang) * r
                let y = cy + sin(ang) * r
                if i == 0 { p.move_to(.init(x, y)) } else { p.line_to(.init(x, y)) }
            }
            p.close_path()
            return p
        }

        func makeEllipseOutline(
            cx: Float,
            cy: Float,
            rx: Float,
            ry: Float,
            reverse: Bool = false
        )
            -> PFPath
        {
            var p = PFPath()
            let kappa: Float = 0.5522847498307936
            let ox = rx * kappa
            let oy = ry * kappa
            p.move_to(.init(cx - rx, cy))
            if !reverse {
                p.bezier_curve_to(.init(cx - rx, cy - oy), .init(cx - ox, cy - ry), .init(cx, cy - ry))
                p.bezier_curve_to(.init(cx + ox, cy - ry), .init(cx + rx, cy - oy), .init(cx + rx, cy))
                p.bezier_curve_to(.init(cx + rx, cy + oy), .init(cx + ox, cy + ry), .init(cx, cy + ry))
                p.bezier_curve_to(.init(cx - ox, cy + ry), .init(cx - rx, cy + oy), .init(cx - rx, cy))
            } else {
                // Trace the ellipse in the opposite winding direction
                p.bezier_curve_to(.init(cx - rx, cy + oy), .init(cx - ox, cy + ry), .init(cx, cy + ry))
                p.bezier_curve_to(.init(cx + ox, cy + ry), .init(cx + rx, cy + oy), .init(cx + rx, cy))
                p.bezier_curve_to(.init(cx + rx, cy - oy), .init(cx + ox, cy - ry), .init(cx, cy - ry))
                p.bezier_curve_to(.init(cx - ox, cy - ry), .init(cx - rx, cy - oy), .init(cx - rx, cy))
            }
            p.close_path()
            return p
        }

        func makeRose(cx: Float, cy: Float, a: Float, k: Float, steps: Int) -> PFPath {
            var p = PFPath()
            for i in 0...steps {
                let t = Float(i) / Float(steps) * (.pi * 2)
                let r = a * cos(k * t)
                let x = cx + r * cos(t)
                let y = cy + r * sin(t)
                if i == 0 { p.move_to(.init(x, y)) } else { p.line_to(.init(x, y)) }
            }
            p.close_path()
            return p
        }

        func hachureRect(
            x: Float,
            y: Float,
            w: Float,
            h: Float,
            gap: Float = 10,
            angle: Float = .pi / 12,
            strokeWidth: Float = 3
        ) -> (PFPath, PFPath, PFPath) {
            // Draw angled hatch lines first
            let diag = hypotf(w, h) * 1.2
            let cosA = cos(angle)
            let sinA = sin(angle)
            let count = Int(ceil((w + h) / gap)) + 2

            var fillPath = PFPath()
            for i in -1..<count {
                let offset = (Float(i) * gap) - h
                let cx = x + w * 0.5
                let cy = y + h * 0.5
                let dx = cosA
                let dy = sinA
                let px = cx + offset * (-dy)
                let py = cy + offset * (dx)
                let x0 = px - dx * diag
                let y0 = py - dy * diag
                let x1 = px + dx * diag
                let y1 = py + dy * diag
                fillPath.move_to(.init(x0, y0))
                fillPath.line_to(.init(x1, y1))
            }

            // Draw border last so it's not covered by hatches
            var border = PFPath()
            border.rect(.init(origin: .init(x, y), size: .init(w, h)))

            var hachureClip = PFPath()
            hachureClip.rect(.init(origin: .init(x, y), size: .init(w, h)))

            return (fillPath, border, hachureClip)
        }

        var outer = makeEllipseOutline(cx: 860, cy: 250, rx: 70, ry: 48, reverse: false)
        var inner = makeEllipseOutline(cx: 860, cy: 250, rx: 35, ry: 24, reverse: true)

        var donut = PFPath()
        donut.add_path(&outer, .init())
        donut.add_path(&inner, .init())

        stroke_path(path)
        stroke_path(addEllipse(360, 420, 90, 60))
        stroke_path(curve1())
        stroke_path(makeStar(cx: 620, cy: 620, rOuter: 70, rInner: 30, points: 5))
        fill_path(donut, .evenOdd)
        var rose = makeRose(cx: 960, cy: 360, a: 70, k: 4, steps: 400)

        stroke_path(rose)

        draw {
            setFillStyle(.color(.init(r: 60, g: 200, b: 255, a: 220)))
            fill_path(rose, .winding)
        }

        var (hachurePath, border, hachureClip) = hachureRect(
            x: 120,
            y: 800,
            w: 240,
            h: 190,
            gap: 10,
            angle: .pi / 6,
            strokeWidth: 4
        )
        clip_path(&hachureClip, .winding)
        stroke_path(hachurePath)
        current_state.clip_path = nil
        stroke_path(border)

        var rect = PFPath()
        rect.rect(.init(origin: .init(60, 70), size: .init(240, 190)))

        let origin = SIMD2<Float>(60, 70)
        let center = origin + SIMD2<Float>(240, 190) * 0.5
        set_transform(Transform(translation: -center).rotate(.pi / 4).translate(center))

        stroke_path(rect)
        reset_transform()
        stroke_path(rect)

        var dashRect = PFPath()
        dashRect.rect(.init(origin: .init(520, 800), size: .init(300, 250)))
        setLineDash([5, 6])
        stroke_path(dashRect)
        resetLineDash()

        var dashRect2 = PFPath()
        dashRect2.rect(.init(origin: .init(620, 800), size: .init(300, 250)))
        stroke_path(dashRect2)

        draw() {
            set_stroke_style(.color(.init(r: 255, g: 0, b: 0, a: 255)))
            setLineJoin(.round)
            set_line_width(5)

            var redRect = PFPath()
            redRect.rect(.init(origin: .init(990, 800), size: .init(300, 250)))
            stroke_path(redRect)
        }

        var blackRect = PFPath()
        blackRect.rect(.init(origin: .init(990, 500), size: .init(300, 250)))
        stroke_path(blackRect)

        draw {
            set_stroke_style(.color(.init(r: 0, g: 255, b: 0, a: 255)))
            set_line_width(5)
            setLineJoin(.round)

            current_state.line_cap = .square

            var line = PFPath()
            line.move_to(.init(1000, 200))
            line.line_to(.init(1400, 300))
            line.line_to(.init(1500, 800))

            stroke_path(line)
        }

        render(
            device: self.device,
            options: .init(),
            size: size,
            backgroundColor: .init(r: 1, g: 1, b: 1, a: 1.0)
        )

        self.device.present_drawable(drawable)
    }

    var renderer: Renderer!

    public func render(
        device: PFDevice,
        options: Scene.BuildOptions,
        size: SIMD2<Int32>,
        backgroundColor: ColorF?
    ) {
        if renderer == nil {
            renderer = Renderer(
                device: device,
                options: .init(
                    dest: .full_window(size),
                    background_color: backgroundColor
                )
            )
        }

        let listener = SceneBuilder.RenderCommandListener(send_fn: { cmd in
            self.renderer.render_command(command: cmd)
        })

        var sink = SceneBuilder.SceneSink(listener)

        renderer.begin_scene()
        scene.build(options: options, sink: &sink)
        renderer.end_scene()
    }

    func draw(f: () -> ()) {
        let state = current_state
        stateStack.append(state)
        f()
        current_state = stateStack.popLast()!
    }

    func set_line_width(_ new_line_width: Float32) {
        self.current_state.line_width = new_line_width
    }

    func setLineJoin(_ new_line_join: LineJoin) {
        self.current_state.line_join = new_line_join
    }

    func setFillStyle(_ new_fill_style: FillStyle) {
        self.current_state.fill_paint = new_fill_style.into_paint()
    }

    public func setLineDash(_ newLineDash: [Float]) {
        var lineDash = newLineDash
        // Duplicate and concatenate if an odd number of dashes are present.
        if lineDash.count % 2 == 1 {
            let realLineDash = lineDash
            lineDash.append(contentsOf: realLineDash)
        }

        self.current_state.line_dash = lineDash
    }

    func resetLineDash() {
        current_state.line_dash = []
    }

    public func setLineDashOffset(_ newLineDashOffset: Float) {
        self.current_state.line_dash_offset = newLineDashOffset
    }

    func set_stroke_style(_ new_stroke_style: FillStyle) {
        self.current_state.stroke_paint = new_stroke_style.into_paint()
    }

    func set_transform(_ new_transform: Transform) {
        current_state.transform = new_transform
    }

    func reset_transform() {
        current_state.transform = Transform()
    }

    func fill_rect(_ rect: PFRect<Float32>) {
        var path = PFPath()
        path.rect(rect)
        fill_path(path, .winding)
    }

    func stroke_rect(_ rect: PFRect<Float32>) {
        var path = PFPath()
        path.rect(rect)
        stroke_path(path)
    }

    func clear_rect(_ rect: PFRect<Float32>) {
        var path = PFPath()
        path.rect(rect)

        let paint = Paint.transparent_black
        let resolved_paint = current_state.resolve_paint(paint)
        let paint_id = scene.push_paint(resolved_paint)

        var outline = path.into_outline()
        outline.transform(current_state.transform)

        var draw_path = Scene.DrawPath(outline, paint_id)
        draw_path.blend_mode = .clear
        _ = scene.push_draw_path(draw_path)
    }

    func fill_path(_ path: PFPath, _ fill_rule: Scene.FillRule) {
        var path = path
        var outline = path.into_outline()
        push_path(&outline, .fill, fill_rule)
    }

    func stroke_path(_ path: PFPath) {
        var path = path
        var stroke_style = current_state.resolve_stroke_style()

        // The smaller scale is relevant here, as we multiply by it and want to ensure it is always
        // bigger than `HAIRLINE_STROKE_WIDTH`.
        let transform_scales = current_state.transform.extract_scale()
        let transform_scale = min(transform_scales.x, transform_scales.y)

        // Avoid the division in the normal case of sufficient thickness.
        if stroke_style.line_width * transform_scale < Self.HAIRLINE_STROKE_WIDTH {
            stroke_style.line_width = Self.HAIRLINE_STROKE_WIDTH / transform_scale
        }

        var outline = path.into_outline()
        if !current_state.line_dash.isEmpty {
            var dash = OutlineDash(outline, current_state.line_dash, current_state.line_dash_offset)
            dash.dash()
            outline = dash.into_outline()
        }

        var stroke_to_fill = Outline.OutlineStrokeToFill(outline, style: stroke_style)
        stroke_to_fill.offset()
        outline = stroke_to_fill.into_outline()

        push_path(&outline, .stroke, .winding)
    }

    func clip_path(_ path: inout PFPath, _ fill_rule: Scene.FillRule) {
        var outline = path.into_outline()
        outline.transform(current_state.transform)

        var clip_path = Scene.ClipPath(outline)
        clip_path.fill_rule = fill_rule
        if let existing_clip_path = current_state.clip_path {
            clip_path.clip_path = existing_clip_path
            current_state.clip_path = nil
        }

        let clip_path_id = scene.push_clip_path(clip_path)
        current_state.clip_path = clip_path_id
    }

    func push_path(_ outline: inout Outline, _ path_op: PathOp, _ fill_rule: Scene.FillRule) {
        let paint = current_state.resolve_paint(
            path_op == .fill ? current_state.fill_paint : current_state.stroke_paint
        )
        let paint_id = scene.push_paint(paint)

        let transform = current_state.transform
        let clip_path = current_state.clip_path
        let blend_mode = current_state.global_composite_operation.to_blend_mode()

        outline.transform(transform)

        if !current_state.shadow_color.isFullyTransparent {
            var shadow_outline = outline
            shadow_outline.transform(.init(translation: current_state.shadow_offset))

            let shadow_blur_info = Self.push_shadow_blur_render_targets_if_needed(
                &scene,
                current_state,
                shadow_outline.bounds
            )

            if let shadow_blur_info = shadow_blur_info {
                shadow_outline.transform(Transform(translation: -shadow_blur_info.bounds.f32.origin))
            }

            // Per spec the shadow must respect the alpha of the shadowed path, but otherwise have
            // the color of the shadow paint.
            var shadow_paint = paint
            let shadow_base_alpha = shadow_paint.baseColor.a
            var shadow_color = current_state.shadow_color.f32
            shadow_color.a = (shadow_color.a * Float(shadow_base_alpha) / 255.0)
            shadow_paint.baseColor = shadow_color.u8

            shadow_paint.overlay?.compositeOp = .destIn

            let shadow_paint_id = scene.push_paint(shadow_paint)

            var shadow_path = Scene.DrawPath(shadow_outline, shadow_paint_id)
            if shadow_blur_info == nil {
                shadow_path.clip_path = clip_path
            }
            shadow_path.fill_rule = fill_rule
            shadow_path.blend_mode = blend_mode
            _ = scene.push_draw_path(shadow_path)

            Self.composite_shadow_blur_render_targets_if_needed(&scene, shadow_blur_info, clip_path)
        }

        var path = Scene.DrawPath(outline, paint_id)
        path.clip_path = clip_path
        path.fill_rule = fill_rule
        path.blend_mode = blend_mode
        _ = scene.push_draw_path(path)
    }

    static func composite_shadow_blur_render_targets_if_needed(
        _ scene: inout Scene,
        _ info: ShadowBlurRenderTargetInfo?,
        _ clip_path: UInt32?
    ) {
        guard let info = info else { return }

        var paint_x = Pattern(id: info.id_x, size: info.bounds.size)
        var paint_y = Pattern(id: info.id_y, size: info.bounds.size)
        paint_y.apply_transform(Transform(translation: info.bounds.f32.origin))

        let sigma = info.sigma
        paint_x.filter = .blur(direction: .x, sigma: sigma)
        paint_y.filter = .blur(direction: .y, sigma: sigma)

        let paint_id_x = scene.push_paint(Paint(pattern: paint_x))
        let paint_id_y = scene.push_paint(Paint(pattern: paint_y))

        // TODO(pcwalton): Apply clip as necessary.
        let outline_x = Outline(
            rect: .init(origin: SIMD2<Float>(0.0, 0.0), size: info.bounds.f32.size)
        )
        let path_x = Scene.DrawPath(outline_x, paint_id_x)
        let outline_y = Outline(rect: info.bounds.f32)
        var path_y = Scene.DrawPath(outline_y, paint_id_y)
        path_y.clip_path = clip_path

        scene.pop_render_target()
        _ = scene.push_draw_path(path_x)
        scene.pop_render_target()
        _ = scene.push_draw_path(path_y)
    }

    static func push_shadow_blur_render_targets_if_needed(
        _ scene: inout Scene,
        _ current_state: State,
        _ outline_bounds: PFRect<Float32>
    ) -> ShadowBlurRenderTargetInfo? {
        if current_state.shadow_blur == 0.0 {
            return nil
        }

        let sigma = current_state.shadow_blur * 0.5
        let bounds = outline_bounds.dilate(sigma * 3.0).round_out().i32

        let render_target_y = Scene.RenderTarget(size: bounds.size, name: "")
        let render_target_id_y = scene.push_render_target(render_target_y)
        let render_target_x = Scene.RenderTarget(size: bounds.size, name: "")
        let render_target_id_x = scene.push_render_target(render_target_x)

        return ShadowBlurRenderTargetInfo(
            id_x: render_target_id_x,
            id_y: render_target_id_y,
            bounds: bounds,
            sigma: sigma
        )
    }

}

extension Canvas.State {
    init() {
        self.transform = Transform()
        self.line_width = 1.0
        self.line_cap = .butt
        self.line_join = .miter
        self.miter_limit = 10.0
        self.line_dash = []
        self.line_dash_offset = 0.0
        self.fill_paint = .black
        self.stroke_paint = .black
        self.shadow_color = .transparent_black
        self.shadow_blur = 0.0
        self.shadow_offset = .zero
        self.image_smoothing_enabled = true
        self.image_smoothing_quality = .low
        self.global_alpha = 1.0
        self.global_composite_operation = .sourceOver
        self.clip_path = nil
    }

    func resolve_paint(_ paint: Paint) -> Paint {
        var must_copy = !transform.isIdentity || global_alpha < 1.0
        if !must_copy {
            if let pattern = paint.pattern {
                must_copy = image_smoothing_enabled != pattern.smoothingEnabled
            }
        }

        if !must_copy {
            return paint
        }

        var resolved_paint = paint
        resolved_paint.apply_transform(transform)

        var base_color = resolved_paint.baseColor.f32
        base_color.a = base_color.a * global_alpha
        resolved_paint.baseColor = base_color.u8

        if var pattern = resolved_paint.pattern {
            pattern.smoothingEnabled = image_smoothing_enabled
            resolved_paint.pattern = pattern
        }

        return resolved_paint
    }
}

struct ShadowBlurRenderTargetInfo {
    var id_x: Scene.RenderTargetId
    var id_y: Scene.RenderTargetId
    var bounds: PFRect<Int32>
    var sigma: Float
}
