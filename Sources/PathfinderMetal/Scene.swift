import Foundation

public struct Scene {
    struct DrawPathId: Comparable {
        let value: UInt32

        static func < (lhs: DrawPathId, rhs: DrawPathId) -> Bool {
            lhs.value < rhs.value
        }
    }

    struct ClipPathId {
        let value: UInt32
    }

    struct RenderTarget {
        var size: SIMD2<Int32>
        var name: String
    }

    struct RenderTargetId: Hashable {
        /// The ID of the scene that this render target ID belongs to.
        var scene: UInt32
        /// The ID of the render target within this scene.
        var render_target: UInt32
    }

    enum DisplayItem {
        /// Draws paths to the render target on top of the stack.
        case drawPaths(Range<UInt32>)

        /// Pushes a render target onto the top of the stack.
        case pushRenderTarget(RenderTargetId)

        /// Pops a render target from the stack.
        case popRenderTarget
    }

    enum FillRule {
        /// The nonzero rule: <https://en.wikipedia.org/wiki/Nonzero-rule>
        case winding
        /// The even-odd rule: <https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule>
        case evenOdd
    }

    public enum BlendMode {
        case clear, copy, srcIn
        case srcOut, srcOver, srcAtop, destIn, destOut, destOver, destAtop, xor
        case lighter, darken, lighten, multiply, screen, hardLight, overlay, colorDodge, colorBurn,
            softLight
        case difference, exclusion, hue, saturation, color, luminosity

        public func occludesBackdrop() -> Bool {
            switch self {
            case .clear, .copy, .srcOver: return true
            default: return false
            }
        }

        public func needsReadableFramebuffer() -> Bool {
            switch self {
            case .lighten, .darken, .multiply, .screen, .hardLight, .overlay, .colorDodge, .colorBurn,
                .softLight,
                .difference, .exclusion, .hue, .saturation, .color, .luminosity:
                return true
            default:
                return false
            }
        }
    }

    struct DrawPath {
        /// The actual vector path outline.
        var outline: Outline
        /// The ID of the paint that specifies how to fill the interior of this outline.
        var paint: UInt16
        /// The ID of an optional clip path that will be used to clip this path.
        var clip_path: UInt32?
        /// How to fill this path (winding or even-odd).
        var fill_rule: FillRule
        /// How to blend this path with everything below it.
        var blend_mode: BlendMode
        /// The name of this path, for debugging.
        ///
        /// Pass the empty string (which does not allocate) if debugging is not needed.
        var name: String

        init(_ outline: Outline, _ paint: UInt16) {
            self.outline = outline
            self.paint = paint
            self.clip_path = nil
            self.fill_rule = .winding
            self.blend_mode = .srcOver
            self.name = ""
        }
    }

    struct ClipPath {
        /// The actual vector path outline.
        var outline: Outline
        /// The ID of another, previously-defined, clip path that clips this one.
        ///
        /// Nested clips can be achieved by clipping clip paths with other clip paths.
        var clip_path: UInt32?
        /// How to fill this path (winding or even-odd).
        var fill_rule: FillRule
        /// The name of this clip path, for debugging.
        ///
        /// Pass the empty string (which does not allocate) if debugging is not needed.
        var name: String

        init(_ outline: Outline) {
            self.outline = outline
            self.clip_path = nil
            self.fill_rule = .winding
            self.name = ""
        }
    }

    struct SceneEpoch {
        var hi: UInt64
        var lo: UInt64

        func equals(_ other: SceneEpoch) -> Bool {
            return self.hi == other.hi && self.lo == other.lo
        }

        init(_ hi: UInt64, _ lo: UInt64) {
            self.hi = hi
            self.lo = lo
        }

        func successor() -> SceneEpoch {
            if lo == UInt64.max {
                return SceneEpoch(hi + 1, 0)
            } else {
                return SceneEpoch(hi, lo + 1)
            }
        }

        mutating func next() {
            self = successor()
        }
    }

    public struct BuildOptions {
        /// A global transform to be applied to the scene.
        var transform: Transform = .init()
        /// Expands outlines by the given number of device pixels. This is useful to perform *stem
        /// darkening* for fonts, to mitigate the thinness of gamma-corrected fonts.
        var dilation: SIMD2<Float32> = .zero
        /// True if subpixel antialiasing for LCD screens is to be performed.
        var subpixel_aa_enabled: Bool = false

        func prepare(_ bounds: PFRect<Float32>) -> SceneBuilder.PreparedBuildOptions {
            let renderTransform: SceneBuilder.PreparedRenderTransform
            if transform.isIdentity {
                renderTransform = .none
            } else {
                renderTransform = .transform(transform)
            }

            return .init(
                transform: renderTransform,
                dilation: self.dilation,
                subpixel_aa_enabled: self.subpixel_aa_enabled
            )
        }
    }

    var display_list: [DisplayItem]
    var draw_paths: [DrawPath]
    var clip_paths: [ClipPath]
    var palette: Palette1
    var bounds: PFRect<Float32>
    var view_box: PFRect<Float32>
    var id: UInt32
    var epoch: SceneEpoch
}

struct SceneBuilder {
    static let TILE_CTRL_MASK_WINDING: Int32 = 0x1
    static let TILE_CTRL_MASK_EVEN_ODD: Int32 = 0x2

    static let TILE_CTRL_MASK_0_SHIFT: Int32 = 0

    static let TILE_WIDTH: UInt32 = 16
    static let TILE_HEIGHT: UInt32 = 16

    enum PreparedRenderTransform {
        case none
        case transform(Transform)
    }

    struct BoundingQuad {
        var f1: SIMD4<Float32> = .zero
        var f2: SIMD4<Float32> = .zero
        var f3: SIMD4<Float32> = .zero
        var f4: SIMD4<Float32> = .zero

        var data: [SIMD4<Float32>] { [f1, f2, f3, f4] }
    }

    struct PreparedBuildOptions {
        var transform: PreparedRenderTransform
        var dilation: SIMD2<Float32>
        var subpixel_aa_enabled: Bool

        func boundingQuad() -> BoundingQuad {
            return BoundingQuad()
        }
    }

    typealias RenderCommandSendFunction = (RenderCommand) -> Void

    struct RenderCommandListener {
        var send_fn: RenderCommandSendFunction

        func send(_ render_command: RenderCommand) {
            send_fn(render_command)
        }
    }

    struct LastSceneInfo {
        var scene_id: UInt32
        var scene_epoch: Scene.SceneEpoch
        var draw_segment_ranges: [Range<UInt32>]
        var clip_segment_ranges: [Range<UInt32>]
    }

    struct TextureLocation {
        var page: UInt32
        var rect: PFRect<Int32>
    }

    struct PaintTextureManager {
        var allocator: TextureAllocator
        var cached_images: [UInt64: TextureLocation]

        init() {
            self.allocator = TextureAllocator()
            self.cached_images = [:]
        }
    }

    struct SceneSink {
        var listener: RenderCommandListener
        var last_scene: LastSceneInfo?
        var paint_texture_manager: PaintTextureManager

        init(_ listener: RenderCommandListener) {
            self.listener = listener
            self.last_scene = nil
            self.paint_texture_manager = PaintTextureManager()
        }
    }

    struct BuiltSegments {
        var draw_segments: RenderCommand.SegmentsD3D11
        var clip_segments: RenderCommand.SegmentsD3D11
        var draw_segment_ranges: [Range<UInt32>]
        var clip_segment_ranges: [Range<UInt32>]
    }

    struct ClipBatchesD3D11 {
        // Will be submitted in reverse (LIFO) order.
        var prepare_batches: [RenderCommand.TileBatchDataD3D11]
        var clip_id_to_path_batch_index: [UInt32: UInt32]
    }

    struct TileBatchBuilder {
        static let MAX_CLIP_BATCHES: UInt32 = 32

        var prepare_commands: [RenderCommand]
        var draw_commands: [RenderCommand]
        var clip_batches_d3d11: ClipBatchesD3D11?
        var next_batch_id: UInt32

        init() {
            prepare_commands = []
            draw_commands = []
            next_batch_id = TileBatchBuilder.MAX_CLIP_BATCHES
            clip_batches_d3d11 = ClipBatchesD3D11(prepare_batches: [], clip_id_to_path_batch_index: [:])
        }

        func send_to(sink: SceneBuilder.SceneSink) {
            if let clip_batches_d3d11 = self.clip_batches_d3d11 {
                for prepare_batch in clip_batches_d3d11.prepare_batches.reversed() {
                    if prepare_batch.path_count > 0 {
                        sink.listener.send(.prepareClipTilesD3D11(prepare_batch))
                    }
                }
            }

            for command in self.prepare_commands {
                sink.listener.send(command)
            }

            for command in self.draw_commands {
                sink.listener.send(command)
            }
        }
    }

    struct BuiltPath {
        var tile_bounds: PFRect<Int32>
        var fill_rule: Scene.FillRule
        var clip_path_id: UInt32?
        var ctrl_byte: UInt8
        var paint_id: UInt16
    }

    struct BuiltDrawPath {
        var path: BuiltPath
        var clip_path_id: UInt32?
        var blend_mode: Scene.BlendMode
        var filter: Filter
        var color_texture: RenderCommand.TileBatchTexture?
        var sampling_flags_1: RenderCommand.TextureSamplingFlags
        var mask_0_fill_rule: Scene.FillRule
        var occludes: Bool

        init(
            _ built_path: BuiltPath,
            _ path_object: Scene.DrawPath,
            _ paint_metadata: Paint.PaintMetadata
        ) {
            let blend_mode = path_object.blend_mode
            let occludes = paint_metadata.is_opaque && blend_mode.occludes_backdrop()
            self.path = built_path
            self.clip_path_id = path_object.clip_path
            self.filter = paint_metadata.filter
            self.color_texture = paint_metadata.tile_batch_texture
            self.sampling_flags_1 = .init()
            self.mask_0_fill_rule = path_object.fill_rule
            self.blend_mode = blend_mode
            self.occludes = occludes
        }
    }

    struct PrepareMode {
        var transform: Transform
    }

    struct DrawTilingPathInfo {
        var paint_id: UInt16
        var blend_mode: Scene.BlendMode
        var fill_rule: Scene.FillRule
    }

    enum TilingPathInfo {
        case clip
        case draw(DrawTilingPathInfo)

        var ctrl: UInt8 {
            var ctrl: UInt8 = 0

            switch self {
            case .draw(let draw_tiling_path_info):
                switch draw_tiling_path_info.fill_rule {
                case .evenOdd:
                    ctrl |= UInt8(
                        SceneBuilder.TILE_CTRL_MASK_EVEN_ODD << SceneBuilder.TILE_CTRL_MASK_0_SHIFT
                    )
                case .winding:
                    ctrl |= UInt8(
                        SceneBuilder.TILE_CTRL_MASK_WINDING << SceneBuilder.TILE_CTRL_MASK_0_SHIFT
                    )
                }
            case .clip:
                break
            }

            return ctrl
        }

        var has_destructive_blend_mode: Bool {
            switch self {
            case .draw(let draw_tiling_path_info):
                return draw_tiling_path_info.blend_mode.is_destructive()
            case .clip:
                return false
            }
        }
    }

    struct GlobalPathId {
        var batch_id: UInt32
        var path_index: UInt32
    }

    struct PreparedClipPath {
        var built_path: SceneBuilder.BuiltPath
        var subclip_id: GlobalPathId?
    }

    var scene: Scene
    var built_options: SceneBuilder.PreparedBuildOptions
    var next_alpha_tile_indices: [Int]  // length 2
    var sink: SceneSink
}

extension SceneBuilder.BuiltPath {
    static func round_rect_out_to_tile_bounds(_ rect: PFRect<Float32>) -> PFRect<Int32> {
        (rect
            * SIMD2<Float32>(
                1.0 / Float32(SceneBuilder.TILE_WIDTH),
                1.0 / Float32(SceneBuilder.TILE_HEIGHT)
            ))
            .round_out().i32
    }

    init(
        _ path_id: UInt32,
        _ path_bounds: PFRect<Float32>,
        _ view_box_bounds: PFRect<Float32>,
        _ fill_rule: Scene.FillRule,
        _ prepare_mode: SceneBuilder.PrepareMode,
        _ clip_path_id: UInt32?,
        _ tiling_path_info: SceneBuilder.TilingPathInfo
    ) {

        let paint_id: UInt16
        switch tiling_path_info {
        case .draw(let draw_tiling_path_info):
            paint_id = draw_tiling_path_info.paint_id
        case .clip:
            paint_id = 0
        }

        let ctrl_byte = tiling_path_info.ctrl

        let tile_map_bounds =
            if tiling_path_info.has_destructive_blend_mode {
                view_box_bounds
            } else {
                path_bounds
            }

        let tile_bounds = Self.round_rect_out_to_tile_bounds(tile_map_bounds)

        self.tile_bounds = tile_bounds
        self.clip_path_id = clip_path_id
        self.fill_rule = fill_rule
        self.ctrl_byte = ctrl_byte
        self.paint_id = paint_id
    }
}

extension SceneBuilder.TileBatchBuilder {
    func prepare_draw_path_for_gpu_binning(
        _ scene: Scene,
        _ built_options: SceneBuilder.PreparedBuildOptions,
        _ draw_path_id: UInt32,
        _ prepare_mode: SceneBuilder.PrepareMode,
        _ paint_metadata: [Paint.PaintMetadata]
    ) -> SceneBuilder.BuiltDrawPath? {
        let transform = prepare_mode.transform

        let effective_view_box = scene.effective_view_box(built_options)
        let draw_path = scene.get_draw_path(draw_path_id)

        var path_bounds = transform * draw_path.outline.bounds

        guard let intersection = path_bounds.intersection(effective_view_box) else { return nil }
        path_bounds = intersection

        let paint_id = draw_path.paint
        let paint_metadata = paint_metadata[Int(paint_id)]

        let built_path = SceneBuilder.BuiltPath(
            draw_path_id,
            path_bounds,
            effective_view_box,
            draw_path.fill_rule,
            prepare_mode,
            draw_path.clip_path,
            .draw(
                .init(
                    paint_id: paint_id,
                    blend_mode: draw_path.blend_mode,
                    fill_rule: draw_path.fill_rule
                )
            )
        )
        return .init(built_path, draw_path, paint_metadata)
    }

    private func fixup_batch_for_new_path_if_possible(
        _ batch_color_texture: inout RenderCommand.TileBatchTexture?,
        _ draw_path: SceneBuilder.BuiltDrawPath
    ) -> Bool {
        if draw_path.color_texture != nil {
            if batch_color_texture == nil {
                batch_color_texture = draw_path.color_texture
                return true
            }

            if draw_path.color_texture != batch_color_texture {
                return false
            }
        }

        return true
    }

    mutating func build_tile_batches_for_draw_path_display_item(
        _ scene: Scene,
        _ sink: SceneBuilder.SceneSink,
        _ built_options: SceneBuilder.PreparedBuildOptions,
        _ draw_path_id_range: Range<UInt32>,
        _ paint_metadata: [Paint.PaintMetadata],
        _ prepare_mode: SceneBuilder.PrepareMode
    ) {
        var draw_tile_batch: RenderCommand.DrawTileBatchD3D11? = nil
        for draw_path_id in draw_path_id_range {
            guard
                let draw_path = self.prepare_draw_path_for_gpu_binning(
                    scene,
                    built_options,
                    draw_path_id,
                    prepare_mode,
                    paint_metadata
                )
            else { continue }

            // Try to reuse the current batch if we can.
            var flush_needed = false
            if var existing_batch = draw_tile_batch {
                flush_needed = !fixup_batch_for_new_path_if_possible(
                    &existing_batch.color_texture,
                    draw_path
                )
                draw_tile_batch = existing_batch
            }

            // If we couldn't reuse the batch, flush it.
            if flush_needed {
                let batch = draw_tile_batch
                draw_tile_batch = nil

                if let batch {
                    self.draw_commands.append(.drawTilesD3D11(batch))
                }
            }

            // Create a new batch if necessary.
            if draw_tile_batch == nil {
                draw_tile_batch = .init(
                    tile_batch_data: .init(self.next_batch_id, prepare_mode, .draw),
                    color_texture: draw_path.color_texture
                )

                self.next_batch_id += 1
            }

            // Add clip path if necessary.
            let clip_path: SceneBuilder.GlobalPathId?
            if var clip_batches_d3d11 = self.clip_batches_d3d11 {
                clip_path = add_clip_path_to_batch(
                    scene,
                    sink,
                    built_options,
                    draw_path.clip_path_id,
                    prepare_mode,
                    0,
                    &clip_batches_d3d11
                )
                self.clip_batches_d3d11 = clip_batches_d3d11
            } else {
                clip_path = nil
            }

            _ = draw_tile_batch?.tile_batch_data.push(
                draw_path.path,
                draw_path_id,
                clip_path,
                draw_path.occludes,
                sink
            )
        }

        if let draw_tile_batch {
            self.draw_commands.append(.drawTilesD3D11(draw_tile_batch))
        }
    }

    func add_clip_path_to_batch(
        _ scene: Scene,
        _ sink: SceneBuilder.SceneSink,
        _ built_options: SceneBuilder.PreparedBuildOptions,
        _ clip_path_id: UInt32?,
        _ prepare_mode: SceneBuilder.PrepareMode,
        _ clip_level: Int,
        _ clip_batches_d3d11: inout SceneBuilder.ClipBatchesD3D11
    ) -> SceneBuilder.GlobalPathId? {
        guard let clip_path_id else { return nil }

        if let clip_path_batch_index = clip_batches_d3d11.clip_id_to_path_batch_index[clip_path_id] {
            return .init(
                batch_id: UInt32(clip_level),
                path_index: clip_path_batch_index
            )
        }

        let preparedClipPath = prepare_clip_path_for_gpu_binning(
            scene,
            sink,
            built_options,
            clip_path_id,
            prepare_mode,
            clip_level,
            &clip_batches_d3d11
        )
        let clip_path = preparedClipPath.built_path
        let subclip_id = preparedClipPath.subclip_id

        while clip_level >= clip_batches_d3d11.prepare_batches.count {
            let clip_tile_batch_id = UInt32(clip_batches_d3d11.prepare_batches.count)
            clip_batches_d3d11.prepare_batches.append(.init(clip_tile_batch_id, prepare_mode, .clip))
        }

        let clip_path_batch_index = clip_batches_d3d11.prepare_batches[clip_level].push(
            clip_path,
            clip_path_id,
            subclip_id,
            true,
            sink
        )

        clip_batches_d3d11.clip_id_to_path_batch_index[clip_path_id] = clip_path_batch_index

        return .init(
            batch_id: UInt32(clip_level),
            path_index: clip_path_batch_index
        )
    }

    func prepare_clip_path_for_gpu_binning(
        _ scene: Scene,
        _ sink: SceneBuilder.SceneSink,
        _ built_options: SceneBuilder.PreparedBuildOptions,
        _ clip_path_id: UInt32,
        _ prepare_mode: SceneBuilder.PrepareMode,
        _ clip_level: Int,
        _ clip_batches_d3d11: inout SceneBuilder.ClipBatchesD3D11
    ) -> SceneBuilder.PreparedClipPath {
        let transform = prepare_mode.transform

        let effective_view_box = scene.effective_view_box(built_options)
        let clip_path = scene.get_clip_path(clip_path_id)

        // Add subclip path if necessary.
        let subclip_id = add_clip_path_to_batch(
            scene,
            sink,
            built_options,
            clip_path.clip_path,
            prepare_mode,
            clip_level + 1,
            &clip_batches_d3d11
        )

        let path_bounds = transform * clip_path.outline.bounds

        // TODO(pcwalton): Clip to view box!

        let built_path = SceneBuilder.BuiltPath(
            clip_path_id,
            path_bounds,
            effective_view_box,
            clip_path.fill_rule,
            prepare_mode,
            clip_path.clip_path,
            .clip
        )

        return .init(built_path: built_path, subclip_id: subclip_id)
    }
}

extension Scene {
    private static var _nextSceneID: UInt32 = 0
    private static let _nextSceneIDLock = NSLock()

    private static func getNextSceneID() -> UInt32 {
        _nextSceneIDLock.lock()
        defer { _nextSceneIDLock.unlock() }

        let current = _nextSceneID
        _nextSceneID += 1
        return current
    }

    init() {
        let scene_id = Self.getNextSceneID()

        self.display_list = []
        self.draw_paths = []
        self.clip_paths = []
        self.palette = Palette1(scene_id)
        self.bounds = .zero
        self.view_box = .zero
        self.id = scene_id
        self.epoch = SceneEpoch(0, 1)
    }

    mutating func set_view_box(_ new_view_box: PFRect<Float32>) {
        view_box = new_view_box
        epoch.next()
    }

    mutating func build(options: BuildOptions, sink: inout SceneBuilder.SceneSink) {
        let prepared_options = options.prepare(self.bounds)

        var builder = SceneBuilder(self, prepared_options, sink)
        builder.build()

        self = builder.scene
        sink = builder.sink
    }

    func build_paint_info(
        _ texture_manager: inout SceneBuilder.PaintTextureManager,
        _ render_transform: Transform
    ) -> Paint.PaintInfo {
        palette.build_paint_info(texture_manager: &texture_manager, render_transform: render_transform)
    }

    func effective_view_box(_ render_options: SceneBuilder.PreparedBuildOptions) -> PFRect<Float32> {
        if render_options.subpixel_aa_enabled {
            self.view_box * SIMD2<Float32>(3.0, 1.0)
        } else {
            self.view_box
        }
    }

    func get_draw_path(_ draw_path_id: UInt32) -> DrawPath {
        self.draw_paths[Int(draw_path_id)]
    }

    func get_clip_path(_ clip_path_id: UInt32) -> ClipPath {
        self.clip_paths[Int(clip_path_id)]
    }

    mutating func push_paint(_ paint: Paint) -> UInt16 {
        let paint_id = palette.push_paint(paint)
        epoch.next()
        return paint_id
    }

    mutating func push_clip_path(_ clip_path: ClipPath) -> UInt32 {
        bounds = bounds.unionRect(clip_path.outline.bounds)
        let clip_path_id = UInt32(clip_paths.count)
        clip_paths.append(clip_path)
        epoch.next()
        return clip_path_id
    }

    mutating func push_draw_path(_ draw_path: DrawPath) -> UInt32 {
        let draw_path_index = UInt32(draw_paths.count)
        draw_paths.append(draw_path)
        push_draw_path_with_index(draw_path_index)
        return draw_path_index
    }

    mutating func pop_render_target() {
        display_list.append(.popRenderTarget)
    }

    mutating func push_draw_path_with_index(_ draw_path_id: UInt32) {
        let new_path_bounds = draw_paths[Int(draw_path_id)].outline.bounds
        bounds = bounds.unionRect(new_path_bounds)

        let end_path_id = draw_path_id + 1
        if case .drawPaths(var range) = display_list.last {
            range = range.lowerBound..<end_path_id
            display_list[display_list.count - 1] = .drawPaths(range)
        } else {
            display_list.append(.drawPaths(draw_path_id..<end_path_id))
        }

        epoch.next()
    }

    mutating func push_render_target(_ render_target: RenderTarget) -> Scene.RenderTargetId {
        let render_target_id = palette.push_render_target(render_target)
        display_list.append(.pushRenderTarget(render_target_id))
        epoch.next()
        return render_target_id
    }
}

extension SceneBuilder.BuiltSegments {
    init(scene: Scene) {
        draw_segments = RenderCommand.SegmentsD3D11()
        clip_segments = RenderCommand.SegmentsD3D11()
        draw_segment_ranges = []
        clip_segment_ranges = []
        draw_segment_ranges.reserveCapacity(scene.draw_paths.count)
        clip_segment_ranges.reserveCapacity(scene.clip_paths.count)

        for clip_path in scene.clip_paths {
            let range = clip_segments.add_path(outline: clip_path.outline)
            clip_segment_ranges.append(range)
        }

        for draw_path in scene.draw_paths {
            let range = draw_segments.add_path(outline: draw_path.outline)
            draw_segment_ranges.append(range)
        }
    }
}

extension SceneBuilder {
    init(
        _ scene: Scene,
        _ built_options: PreparedBuildOptions,
        _ sink: SceneSink
    ) {
        self.scene = scene
        self.built_options = built_options
        self.next_alpha_tile_indices = [0, 0]
        self.sink = sink
    }

    mutating func build() {
        let start_time = DispatchTime.now()

        // Send the start rendering command.
        let bounding_quad = self.built_options.boundingQuad()

        let clip_path_count = self.scene.clip_paths.count
        let draw_path_count = self.scene.draw_paths.count
        let total_path_count = clip_path_count + draw_path_count

        let needs_readable_framebuffer = self.needs_readable_framebuffer()

        self.sink.listener.send(
            .start(
                path_count: total_path_count,
                bounding_quad: bounding_quad.data,
                needs_readable_framebuffer: needs_readable_framebuffer
            )
        )

        let prepareMode: PrepareMode

        let render_transform: Transform
        switch self.built_options.transform {
        case .transform(let transform):
            render_transform = transform.inverse()
            prepareMode = PrepareMode(transform: transform)
        case .none:
            render_transform = Transform()
            prepareMode = PrepareMode(transform: .init())
        }

        // Build paint data.
        let paintInfo = self.scene.build_paint_info(&sink.paint_texture_manager, render_transform)

        let render_commands = paintInfo.render_commands
        let paint_metadata = paintInfo.paint_metadata

        for render_command in render_commands {
            self.sink.listener.send(render_command)
        }

        var scene_is_dirty = true
        if let last_scene = sink.last_scene {
            scene_is_dirty =
                last_scene.scene_id != scene.id || !last_scene.scene_epoch.equals(scene.epoch)
        }

        if scene_is_dirty {
            let built_segments = BuiltSegments(scene: self.scene)
            self.sink.listener.send(
                .uploadSceneD3D11(
                    draw_segments: built_segments.draw_segments,
                    clip_segments: built_segments.clip_segments
                )
            )
            self.sink.last_scene = LastSceneInfo(
                scene_id: self.scene.id,
                scene_epoch: self.scene.epoch,
                draw_segment_ranges: built_segments.draw_segment_ranges,
                clip_segment_ranges: built_segments.clip_segment_ranges
            )
        }

        self.finish_building(paint_metadata, prepareMode)

        let cpu_build_time = start_time.distance(to: DispatchTime.now())
        self.sink.listener.send(.finish(cpu_build_time: cpu_build_time))
    }

    func needs_readable_framebuffer() -> Bool {
        var framebuffer_nesting = 0

        for display_item in self.scene.display_list {
            switch display_item {
            case .pushRenderTarget:
                framebuffer_nesting += 1
            case .popRenderTarget:
                framebuffer_nesting -= 1
            case .drawPaths(let draw_path_id_range):
                if framebuffer_nesting > 0 {
                    continue
                }

                for draw_path_id in draw_path_id_range {
                    let blend_mode = self.scene.draw_paths[Int(draw_path_id)].blend_mode
                    if blend_mode.needs_readable_framebuffer() {
                        return true
                    }
                }
            }
        }

        return false
    }

    mutating func finish_building(
        _ paint_metadata: [Paint.PaintMetadata],
        _ prepare_mode: PrepareMode
    ) {
        self.build_tile_batches(paint_metadata, prepare_mode)
    }

    mutating func build_tile_batches(
        _ paint_metadata: [Paint.PaintMetadata],
        _ prepare_mode: PrepareMode
    ) {
        var tile_batch_builder = TileBatchBuilder()

        // Prepare display items.
        for display_item in self.scene.display_list {
            switch display_item {
            case .pushRenderTarget(let render_target_id):
                tile_batch_builder.draw_commands.append(.pushRenderTarget(render_target_id))
            case .popRenderTarget:
                tile_batch_builder.draw_commands.append(.popRenderTarget)
            case .drawPaths(let path_id_range):
                tile_batch_builder.build_tile_batches_for_draw_path_display_item(
                    self.scene,
                    self.sink,
                    self.built_options,
                    path_id_range,
                    paint_metadata,
                    prepare_mode
                )
            }
        }

        // Send commands.
        tile_batch_builder.send_to(sink: self.sink)
    }
}

extension Scene.BlendMode {
    static let COMBINER_CTRL_COMPOSITE_NORMAL: Int32 = 0x0
    static let COMBINER_CTRL_COMPOSITE_MULTIPLY: Int32 = 0x1
    static let COMBINER_CTRL_COMPOSITE_SCREEN: Int32 = 0x2
    static let COMBINER_CTRL_COMPOSITE_OVERLAY: Int32 = 0x3
    static let COMBINER_CTRL_COMPOSITE_DARKEN: Int32 = 0x4
    static let COMBINER_CTRL_COMPOSITE_LIGHTEN: Int32 = 0x5
    static let COMBINER_CTRL_COMPOSITE_COLOR_DODGE: Int32 = 0x6
    static let COMBINER_CTRL_COMPOSITE_COLOR_BURN: Int32 = 0x7
    static let COMBINER_CTRL_COMPOSITE_HARD_LIGHT: Int32 = 0x8
    static let COMBINER_CTRL_COMPOSITE_SOFT_LIGHT: Int32 = 0x9
    static let COMBINER_CTRL_COMPOSITE_DIFFERENCE: Int32 = 0xa
    static let COMBINER_CTRL_COMPOSITE_EXCLUSION: Int32 = 0xb
    static let COMBINER_CTRL_COMPOSITE_HUE: Int32 = 0xc
    static let COMBINER_CTRL_COMPOSITE_SATURATION: Int32 = 0xd
    static let COMBINER_CTRL_COMPOSITE_COLOR: Int32 = 0xe
    static let COMBINER_CTRL_COMPOSITE_LUMINOSITY: Int32 = 0xf

    func needs_readable_framebuffer() -> Bool {
        switch self {
        case .clear,
            .srcOver,
            .destOver,
            .srcIn,
            .destIn,
            .srcOut,
            .destOut,
            .srcAtop,
            .destAtop,
            .xor,
            .lighter,
            .copy:
            return false
        case .lighten,
            .darken,
            .multiply,
            .screen,
            .hardLight,
            .overlay,
            .colorDodge,
            .colorBurn,
            .softLight,
            .difference,
            .exclusion,
            .hue,
            .saturation,
            .color,
            .luminosity:
            return true
        }
    }

    func is_destructive() -> Bool {
        switch self {
        case .clear,
            .copy,
            .srcIn,
            .destIn,
            .srcOut,
            .destAtop:
            return true
        case .srcOver,
            .destOver,
            .destOut,
            .srcAtop,
            .xor,
            .lighter,
            .lighten,
            .darken,
            .multiply,
            .screen,
            .hardLight,
            .overlay,
            .colorDodge,
            .colorBurn,
            .softLight,
            .difference,
            .exclusion,
            .hue,
            .saturation,
            .color,
            .luminosity:
            return false
        }
    }

    func occludes_backdrop() -> Bool {
        switch self {
        case .srcOver, .clear:
            return true
        case .destOver,
            .destOut,
            .srcAtop,
            .xor,
            .lighter,
            .lighten,
            .darken,
            .copy,
            .srcIn,
            .destIn,
            .srcOut,
            .destAtop,
            .multiply,
            .screen,
            .hardLight,
            .overlay,
            .colorDodge,
            .colorBurn,
            .softLight,
            .difference,
            .exclusion,
            .hue,
            .saturation,
            .color,
            .luminosity:
            return false
        }
    }

    func to_composite_ctrl() -> Int32 {
        switch self {
        case .srcOver,
            .srcAtop,
            .destOver,
            .destOut,
            .xor,
            .lighter,
            .clear,
            .copy,
            .srcIn,
            .srcOut,
            .destIn,
            .destAtop:
            return Self.COMBINER_CTRL_COMPOSITE_NORMAL
        case .multiply:
            return Self.COMBINER_CTRL_COMPOSITE_MULTIPLY
        case .darken:
            return Self.COMBINER_CTRL_COMPOSITE_DARKEN
        case .lighten:
            return Self.COMBINER_CTRL_COMPOSITE_LIGHTEN
        case .screen:
            return Self.COMBINER_CTRL_COMPOSITE_SCREEN
        case .overlay:
            return Self.COMBINER_CTRL_COMPOSITE_OVERLAY
        case .colorDodge:
            return Self.COMBINER_CTRL_COMPOSITE_COLOR_DODGE
        case .colorBurn:
            return Self.COMBINER_CTRL_COMPOSITE_COLOR_BURN
        case .hardLight:
            return Self.COMBINER_CTRL_COMPOSITE_HARD_LIGHT
        case .softLight:
            return Self.COMBINER_CTRL_COMPOSITE_SOFT_LIGHT
        case .difference:
            return Self.COMBINER_CTRL_COMPOSITE_DIFFERENCE
        case .exclusion:
            return Self.COMBINER_CTRL_COMPOSITE_EXCLUSION
        case .hue:
            return Self.COMBINER_CTRL_COMPOSITE_HUE
        case .saturation:
            return Self.COMBINER_CTRL_COMPOSITE_SATURATION
        case .color:
            return Self.COMBINER_CTRL_COMPOSITE_COLOR
        case .luminosity:
            return Self.COMBINER_CTRL_COMPOSITE_LUMINOSITY
        }
    }
}

extension Canvas.CompositeOperation {
    func to_blend_mode() -> Scene.BlendMode {
        switch self {
        case .copy: return .copy
        case .sourceAtop: return .srcAtop
        case .destinationOver: return .destOver
        case .destinationOut: return .destOut
        case .xor: return .xor
        case .lighter: return .lighter
        case .multiply: return .multiply
        case .screen: return .screen
        case .overlay: return .overlay
        case .darken: return .darken
        case .lighten: return .lighten
        case .colorDodge: return .colorDodge
        case .colorBurn: return .colorBurn
        case .hardLight: return .hardLight
        case .softLight: return .softLight
        case .difference: return .difference
        case .exclusion: return .exclusion
        case .hue: return .hue
        case .saturation: return .saturation
        case .color: return .color
        case .luminosity: return .luminosity
        case .sourceOver: return .srcOver
        case .sourceIn: return .srcIn
        case .sourceOut: return .srcOut
        case .destinationIn: return .destIn
        case .destinationAtop: return .destAtop
        }
    }
}
