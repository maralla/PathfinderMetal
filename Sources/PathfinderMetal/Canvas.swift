import CoreImage
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
        case color(Color<Float>)
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
        var transform: Transform = .init()
        var line_width: Float = 1.0
        var line_cap: Canvas.LineCap = .butt
        var line_join: Canvas.LineJoin = .miter
        var miter_limit: Float = 10.0
        var line_dash: [Float] = []
        var line_dash_offset: Float = 0.0
        var fill_paint: Paint = .black
        var stroke_paint: Paint = .black
        var shadow_color: Color<Float> = .transparent_black
        var shadow_blur: Float = 0.0
        var shadow_offset: SIMD2<Float32> = .zero
        var image_smoothing_enabled: Bool = true
        var image_smoothing_quality: Canvas.ImageSmoothingQuality = .low

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
    fileprivate var current_state: State
    private var stateStack: [State]

    public init(size: SIMD2<Float32>) {
        scene = .init()
        scene.set_view_box(.init(origin: .zero, size: F2(size)))
        current_state = .init()
        stateStack = []
    }

    public func render(
        on drawable: CAMetalDrawable,
        device: Device,
        options: Scene.BuildOptions
    ) {
        var renderer = Renderer(
            device: PFDevice(device, texture: drawable),
            options: .init(
                dest: .full_window(.init(scene.view_box.size.simd))
            )
        )

        let listener = SceneBuilder.RenderCommandListener(send_fn: { cmd in
            renderer.render_command(command: cmd)
        })

        var sink = SceneBuilder.SceneSink(listener)

        renderer.begin_scene()
        scene.build(options: options, sink: &sink)
        renderer.end_scene()
        renderer.present(drawable: drawable)
    }

    public func draw(f: (inout DrawContext) -> ()) {
        let state = current_state
        stateStack.append(state)
        var ctx = DrawContext(canvas: self)
        f(&ctx)
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

    func fill_rect(_ rect: RectF) {
        var path = PFPath()
        path.rect(rect)
        fill_path(path, .winding)
    }

    func stroke_rect(_ rect: RectF) {
        var path = PFPath()
        path.rect(rect)
        stroke_path(path)
    }

    func clear_rect(_ rect: RectF) {
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
            shadow_outline.transform(.init(translation: F2(current_state.shadow_offset)))

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
            var shadow_color = current_state.shadow_color
            shadow_color.a = shadow_color.a * shadow_base_alpha
            shadow_paint.baseColor = shadow_color

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

        var paint_x = Pattern(id: info.id_x, size: info.bounds.size.simd)
        var paint_y = Pattern(id: info.id_y, size: info.bounds.size.simd)
        paint_y.apply_transform(Transform(translation: info.bounds.f32.origin))

        let sigma = info.sigma
        paint_x.filter = .blur(direction: .x, sigma: sigma)
        paint_y.filter = .blur(direction: .y, sigma: sigma)

        let paint_id_x = scene.push_paint(Paint(pattern: paint_x))
        let paint_id_y = scene.push_paint(Paint(pattern: paint_y))

        // TODO(pcwalton): Apply clip as necessary.
        let outline_x = Outline(
            rect: .init(origin: .zero, size: F2(info.bounds.f32.size.simd))
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
        _ outline_bounds: RectF
    ) -> ShadowBlurRenderTargetInfo? {
        if current_state.shadow_blur == 0.0 {
            return nil
        }

        let sigma = current_state.shadow_blur * 0.5
        let bounds = outline_bounds.dilate(sigma * 3.0).round_out().i32

        let render_target_y = Scene.RenderTarget(size: bounds.size.simd, name: "")
        let render_target_id_y = scene.push_render_target(render_target_y)
        let render_target_x = Scene.RenderTarget(size: bounds.size.simd, name: "")
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

        var base_color = resolved_paint.baseColor
        base_color.a = base_color.a * global_alpha
        resolved_paint.baseColor = base_color

        if var pattern = resolved_paint.pattern {
            pattern.smoothingEnabled = image_smoothing_enabled
            resolved_paint.pattern = pattern
        }

        return resolved_paint
    }
}

public struct DrawContext {
    public typealias LineCap = Canvas.LineCap

    private var canvas: Canvas!

    private var _fillStyle: Style = .color(.black)
    private var _strokeStyle: Style = .color(.black)

    /// CGAffineTransform: [a b c d tx ty] represents the matrix:
    /// [a  c  tx]
    /// [b  d  ty]
    /// [0  0  1 ]
    public var transform: CGAffineTransform {
        get {
            let value = canvas.current_state.transform
            return CGAffineTransform(
                a: CGFloat(value.m11),  // m11
                b: CGFloat(value.m21),  // m21
                c: CGFloat(value.m12),  // m12
                d: CGFloat(value.m22),  // m22
                tx: CGFloat(value.m13),  // tx
                ty: CGFloat(value.m23)  // ty
            )
        }
        set {
            let cg = newValue
            let a = Float32(cg.a)
            let b = Float32(cg.b)
            let c = Float32(cg.c)
            let d = Float32(cg.d)
            let tx = Float32(cg.tx)
            let ty = Float32(cg.ty)

            let t = Transform(m11: a, m12: c, m13: tx, m21: b, m22: d, m23: ty)
            let new = canvas.current_state.transform * t

            canvas.current_state.transform = new
        }
    }

    public var lineWidth: Float {
        get { canvas.current_state.line_width }
        set { canvas.current_state.line_width = newValue }
    }

    public var lineCap: LineCap {
        get { canvas.current_state.line_cap }
        set { canvas.current_state.line_cap = newValue }
    }

    public var lineDash: [Float] {
        get { canvas.current_state.line_dash }
        set { canvas.current_state.line_dash = newValue }
    }

    public var fillStyle: Style {
        get { _fillStyle }
        set {
            _fillStyle = newValue
            canvas.current_state.fill_paint = newValue.toFillStyle().into_paint()
        }
    }

    public var strokeStyle: Style {
        get { _strokeStyle }
        set {
            _strokeStyle = newValue
            canvas.current_state.stroke_paint = newValue.toFillStyle().into_paint()
        }
    }

    public var lineJoin: LineJoin {
        get {
            switch canvas.current_state.line_join {
            case .miter: return .miter(canvas.current_state.miter_limit)
            case .bevel: return .bevel
            case .round: return .round
            }
        }
        set {
            switch newValue {
            case .bevel: canvas.current_state.line_join = .bevel
            case .round: canvas.current_state.line_join = .round
            case .miter(let limit):
                canvas.current_state.line_join = .miter
                canvas.current_state.miter_limit = limit
            }
        }
    }

    public func mask(path: CGPath, fillRule: FillRule = .winding) {
        var pfPath = toPath(path: path)
        canvas.clip_path(&pfPath, fillRule.sceneFillRule)
    }

    public func stroke(_ path: CGPath) {
        let stroked = path.copy(
            strokingWithWidth: CGFloat(lineWidth),
            lineCap: .round,
            lineJoin: .round,
            miterLimit: 10,
            transform: .identity
        )

        let pfPath = toPath(path: stroked)
        canvas.setFillStyle(strokeStyle.toFillStyle())
        canvas.fill_path(pfPath, .winding)
    }

    public func fill(_ path: CGPath, rule: FillRule = .winding) {
        let pfPath = toPath(path: path)
        canvas.fill_path(pfPath, rule.sceneFillRule)
    }

    init(canvas: Canvas) {
        self.canvas = canvas
    }

    private func toPath(path: CGPath) -> PFPath {
        var pfPath = PFPath()

        path.applyWithBlock { elementPtr in
            let element = elementPtr.pointee
            let points = element.points

            switch element.type {
            case .moveToPoint:
                let point = points[0]
                pfPath.move_to(.init(x: Float(point.x), y: Float(point.y)))

            case .addLineToPoint:
                let point = points[0]
                pfPath.line_to(.init(x: Float(point.x), y: Float(point.y)))

            case .addQuadCurveToPoint:
                let controlPoint = points[0]
                let endPoint = points[1]
                pfPath.quadratic_curve_to(
                    .init(x: Float(controlPoint.x), y: Float(controlPoint.y)),
                    .init(x: Float(endPoint.x), y: Float(endPoint.y))
                )

            case .addCurveToPoint:
                let controlPoint1 = points[0]
                let controlPoint2 = points[1]
                let endPoint = points[2]
                pfPath.bezier_curve_to(
                    .init(x: Float(controlPoint1.x), y: Float(controlPoint1.y)),
                    .init(x: Float(controlPoint2.x), y: Float(controlPoint2.y)),
                    .init(x: Float(endPoint.x), y: Float(endPoint.y))
                )

            case .closeSubpath:
                pfPath.close_path()

            @unknown default:
                break
            }
        }

        return pfPath
    }
}

struct ShadowBlurRenderTargetInfo {
    var id_x: Scene.RenderTargetId
    var id_y: Scene.RenderTargetId
    var bounds: RectI
    var sigma: Float
}

extension DrawContext {
    public enum LineJoin {
        case miter(Float), bevel, round
    }

    public enum FillRule {
        case winding, evenOdd

        var sceneFillRule: Scene.FillRule {
            switch self {
            case .winding: .winding
            case .evenOdd: .evenOdd
            }
        }
    }

    public enum Style {
        case color(CGColor)
        case gradient(Gradient)
        case pattern(Pattern)
    }

    public struct Gradient {
        public enum Geometry {
            case linear(from: CGPoint, to: CGPoint)
            case radial(line: (from: CGPoint, to: CGPoint), radii: CGPoint, transform: CGAffineTransform)
        }

        public enum Wrap {
            case clamp, `repeat`
        }

        public struct ColorStop {
            var offset: Double
            var color: CGColor
        }

        public var geometry: Geometry
        public var stops: [ColorStop]
        public var wrap: Wrap
    }

    public struct ColorMatrix {
        var rVector: CIVector  // Red channel transform
        var gVector: CIVector  // Green channel transform
        var bVector: CIVector  // Blue channel transform
        var aVector: CIVector  // Alpha channel transform
        var biasVector: CIVector  // Bias/offset vector

        public init(
            r: CIVector = CIVector(x: 1, y: 0, z: 0, w: 0),
            g: CIVector = CIVector(x: 0, y: 1, z: 0, w: 0),
            b: CIVector = CIVector(x: 0, y: 0, z: 1, w: 0),
            a: CIVector = CIVector(x: 0, y: 0, z: 0, w: 1),
            bias: CIVector = CIVector(x: 0, y: 0, z: 0, w: 0)
        ) {
            self.rVector = r
            self.gVector = g
            self.bVector = b
            self.aVector = a
            self.biasVector = bias
        }

        var asSIMD4Vectors: PFColorMatrix {
            return .init(
                f1: SIMD4<Float32>(Float(rVector.x), Float(rVector.y), Float(rVector.z), Float(rVector.w)),
                f2: SIMD4<Float32>(Float(gVector.x), Float(gVector.y), Float(gVector.z), Float(gVector.w)),
                f3: SIMD4<Float32>(Float(bVector.x), Float(bVector.y), Float(bVector.z), Float(bVector.w)),
                f4: SIMD4<Float32>(Float(aVector.x), Float(aVector.y), Float(aVector.z), Float(aVector.w)),
                f5: SIMD4<Float32>(Float(biasVector.x), Float(biasVector.y), Float(biasVector.z), Float(biasVector.w))
            )
        }
    }

    public struct Pattern {
        public enum BlurDirection {
            case x, y
        }

        public enum PatternFilter {
            case blur(direction: BlurDirection, angle: Double)
            case colorMatrix(ColorMatrix)
        }

        public enum PatternSource {
            case image(CGImage)
        }

        public struct PatternFlags: OptionSet, Sendable {
            public let rawValue: UInt8

            public init(rawValue: UInt8) {
                self.rawValue = rawValue
            }

            public static let repeatX = PatternFlags(rawValue: 0x01)
            public static let repeatY = PatternFlags(rawValue: 0x02)
            public static let noSmoothing = PatternFlags(rawValue: 0x04)
        }

        public var source: PatternSource
        public var transform: CGAffineTransform
        public var filter: PatternFilter?
        public var flags: PatternFlags
    }
}

extension DrawContext.Style {
    func toFillStyle() -> Canvas.FillStyle {
        switch self {
        case .color(let cgColor):
            return .color(Self.convertCGColorToU8(cgColor))
        case .gradient(let gradient):
            return .gradient(Self.convertGradient(gradient))
        case .pattern(let pattern):
            return .pattern(Self.convertPattern(pattern))
        }
    }

    // MARK: - Conversions

    private static func convertCGColorToU8(_ color: CGColor) -> Color<Float> {
        // Convert any incoming color (P3, sRGB, grayscale, etc.) to linear sRGB
        let linearSRGB = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
        let c = color.converted(to: linearSRGB, intent: .defaultIntent, options: nil) ?? color
        let comps = c.components ?? []
        // linear sRGB always has 4 comps (r,g,b,a); fall back for grayscale
        let r = Float(comps.count >= 3 ? comps[0] : comps.first ?? 0)
        let g = Float(comps.count >= 3 ? comps[1] : comps.first ?? 0)
        let b = Float(comps.count >= 3 ? comps[2] : comps.first ?? 0)
        let a = Float(comps.count >= 4 ? comps[3] : c.alpha)
        return Color<Float>(r: r, g: g, b: b, a: a)
    }

    private static func convertGradient(_ gradient: DrawContext.Gradient) -> Gradient {
        // Geometry
        let geometry: Gradient.GradientGeometry = {
            switch gradient.geometry {
            case .linear(from: let from, to: let to):
                return .linear(
                    LineSegment(
                        from: .init(Float(from.x), Float(from.y)),
                        to: .init(Float(to.x), Float(to.y))
                    )
                )
            case .radial((let from, let to), let radii, let transform):
                let ls = LineSegment(
                    from: .init(Float(from.x), Float(from.y)),
                    to: .init(Float(to.x), Float(to.y))
                )
                let t = Self.convertTransform(transform)
                return .radial(line: ls, radii: .init(Float(radii.x), Float(radii.y)), transform: t)
            }
        }()

        // Stops (sorted by offset)
        var stops = gradient.stops.map { stop in
            Gradient.ColorStop(
                offset: Float32(max(0.0, min(1.0, stop.offset))),
                color: convertCGColorToU8(stop.color)
            )
        }
        stops.sort { $0.offset < $1.offset }

        // Wrap
        let wrap: Gradient.GradientWrap =
            switch gradient.wrap {
            case .clamp: .clamp
            case .repeat: .repeat
            }

        return Gradient(geometry: geometry, stops: stops, wrap: wrap)
    }

    static func convertTransform(_ transform: CGAffineTransform) -> Transform {
        let cg = transform

        let a = Float32(cg.a)
        let b = Float32(cg.b)
        let c = Float32(cg.c)
        let d = Float32(cg.d)
        let tx = Float32(cg.tx)
        let ty = Float32(cg.ty)

        return .init(m11: a, m12: c, m13: tx, m21: b, m22: d, m23: ty)
    }

    private static func convertPattern(_ pattern: DrawContext.Pattern) -> Pattern {
        let source: Pattern.PatternSource = {
            switch pattern.source {
            case .image(let cgImage):
                let decoded = decodeImage(cgImage)
                let image = Pattern.Image(
                    size: decoded.size,
                    pixels: decoded.pixels,
                    pixels_hash: decoded.hash,
                    isOpaque: decoded.isOpaque
                )
                return .image(image)
            }
        }()

        var result = Pattern(source: source)

        // Transform
        result.transform = convertTransform(pattern.transform)

        // Filter
        if let filter = pattern.filter {
            switch filter {
            case .blur(direction: let dir, angle: let angle):
                let sigma = Float32(angle)
                let d: Pattern.BlurDirection = (dir == .x) ? .x : .y
                result.filter = .blur(direction: d, sigma: sigma)
            case .colorMatrix(let matrix):
                result.filter = .colorMatrix(matrix.asSIMD4Vectors)
            }
        }

        // Flags
        var flags = Pattern.PatternFlags()
        if pattern.flags.contains(.repeatX) { flags.formUnion(.REPEAT_X) }
        if pattern.flags.contains(.repeatY) { flags.formUnion(.REPEAT_Y) }
        if pattern.flags.contains(.noSmoothing) { flags.formUnion(.NO_SMOOTHING) }
        result.flags = flags

        return result
    }

    private static func decodeImage(
        _ image: CGImage
    ) -> (pixels: [Color<UInt8>], size: SIMD2<Int32>, isOpaque: Bool, hash: UInt64) {
        let width = image.width
        let height = image.height
        let size = SIMD2<Int32>(Int32(width), Int32(height))

        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let bitsPerComponent = 8

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        )

        var pixels: [UInt8] = Array(repeating: 0, count: Int(bytesPerRow * height))
        pixels.withUnsafeMutableBytes { buffer in
            if let context = CGContext(
                data: buffer.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: bitsPerComponent,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue
            ) {
                let rect = CGRect(x: 0, y: 0, width: width, height: height)
                context.draw(image, in: rect)
            }
        }

        var out: [Color<UInt8>] = []
        out.reserveCapacity(Int(width * height))
        var isOpaque = true

        for i in stride(from: 0, to: pixels.count, by: 4) {
            let r = pixels[i]
            let g = pixels[i + 1]
            let b = pixels[i + 2]
            let a = pixels[i + 3]
            if a != 255 { isOpaque = false }
            out.append(Color<UInt8>(r: r, g: g, b: b, a: a))
        }

        // FNV-1a 64-bit
        var hash: UInt64 = 1469598103934665603
        let prime: UInt64 = 1099511628211
        for byte in pixels { hash ^= UInt64(byte); hash &*= prime }

        return (out, size, isOpaque, hash)
    }
}
