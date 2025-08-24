import Foundation
import Metal
import QuartzCore

struct Renderer {
    struct ClearProgram {
        var program: PFDevice.Program
        var rect_uniform: PFDevice.Uniform
        var framebuffer_size_uniform: PFDevice.Uniform
        var color_uniform: PFDevice.Uniform

        init(_ device: PFDevice) {
            let program = device.create_raster_program("clear")
            let rect_uniform = device.get_uniform(program, "Rect")
            let framebuffer_size_uniform = device.get_uniform(program, "FramebufferSize")
            let color_uniform = device.get_uniform(program, "Color")

            self.program = program
            self.rect_uniform = rect_uniform
            self.framebuffer_size_uniform = framebuffer_size_uniform
            self.color_uniform = color_uniform
        }
    }

    struct StencilProgram {
        var program: PFDevice.Program

        init(_ device: PFDevice) {
            program = device.create_raster_program("stencil")
        }
    }

    struct ReprojectionProgram {
        var program: PFDevice.Program
        var old_transform_uniform: PFDevice.Uniform
        var new_transform_uniform: PFDevice.Uniform
        var texture: PFDevice.TextureParameter

        init(_ device: PFDevice) {
            let program = device.create_raster_program("reproject")
            let old_transform_uniform = device.get_uniform(program, "OldTransform")
            let new_transform_uniform = device.get_uniform(program, "NewTransform")
            let texture = device.get_texture_parameter(program, "Texture")

            self.program = program
            self.old_transform_uniform = old_transform_uniform
            self.new_transform_uniform = new_transform_uniform
            self.texture = texture
        }
    }

    enum RenderTarget {
        case `default`
        case framebuffer(PFDevice.Framebuffer)
    }

    struct FilterParams {
        var p0: SIMD4<Float32>
        var p1: SIMD4<Float32>
        var p2: SIMD4<Float32>
        var p3: SIMD4<Float32>
        var p4: SIMD4<Float32>
        var ctrl: Int32
    }

    var core: RendererCore
    var level_impl: RendererD3D11

    // Shaders
    var blit_program: ProgramsCore.BlitProgram
    var clear_program: ClearProgram
    var stencil_program: StencilProgram
    var reprojection_program: ReprojectionProgram

    // Frames
    var frame: Frame
}

struct RendererCore {
    static let MASK_TILES_ACROSS: UInt32 = 256
    static let MASK_TILES_DOWN: UInt32 = 256
    static let MASK_FRAMEBUFFER_WIDTH: Int32 =
        Int32(SceneBuilder.TILE_WIDTH) * Int32(MASK_TILES_ACROSS)
    static let MASK_FRAMEBUFFER_HEIGHT: Int32 =
        Int32(SceneBuilder.TILE_HEIGHT) / 4 * Int32(MASK_TILES_DOWN)

    struct FramebufferFlags: OptionSet {
        let rawValue: UInt8

        static let MASK_FRAMEBUFFER_IS_DIRTY = FramebufferFlags(rawValue: 0x01)
        static let DEST_FRAMEBUFFER_IS_DIRTY = FramebufferFlags(rawValue: 0x02)
    }

    // Basic data
    var device: PFDevice
    var allocator: GPUMemoryAllocator
    var options: RendererOptions
    var renderer_flags: RendererFlags

    // Performance monitoring
    var stats: RenderStats
    var current_timer: PendingTimer?
    var timer_query_cache: TimerQueryCache

    // Core shaders
    var programs: ProgramsCore
    var vertex_arrays: VertexArraysCore

    // Read-only static core resources
    var quad_vertex_positions_buffer_id: UInt64
    var quad_vertex_indices_buffer_id: UInt64
    var area_lut_texture_id: UInt64
    var gamma_lut_texture_id: UInt64

    // Read-write static core resources
    var intermediate_dest_framebuffer_id: UInt64
    var intermediate_dest_framebuffer_size: SIMD2<Int32>
    var texture_metadata_texture_id: UInt64

    // Dynamic resources and associated metadata
    var render_targets: [RenderTargetInfo]
    var render_target_stack: [Scene.RenderTargetId]
    var pattern_texture_pages: [PatternTexturePage?]
    var mask_storage: MaskStorage?
    var alpha_tile_count: UInt32
    var framebuffer_flags: FramebufferFlags

    func draw_render_target() -> Renderer.RenderTarget {
        if let render_target_id = render_target_stack.last {
            let texture_page_id = self.render_target_location(render_target_id).page
            let framebuffer = self.texture_page_framebuffer(texture_page_id)
            return .framebuffer(framebuffer)
        }

        if self.renderer_flags.contains(.INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED) {
            let intermediate_dest_framebuffer = self.allocator.get_framebuffer(
                self.intermediate_dest_framebuffer_id
            )
            return .framebuffer(intermediate_dest_framebuffer)
        }

        switch self.options.dest {
        case .default: return .default
        case .other(let framebuffer):
            return .framebuffer(framebuffer)
        }
    }

    func render_target_location(
        _ render_target_id: Scene.RenderTargetId
    )
        -> SceneBuilder.TextureLocation
    {
        self.render_targets[Int(render_target_id.render_target)].location
    }

    func texture_page_framebuffer(_ id: UInt32) -> PFDevice.Framebuffer {
        let framebuffer_id = self.pattern_texture_pages[Int(id)]!.framebuffer_id
        return self.allocator.get_framebuffer(framebuffer_id)
    }

    func texture_page(_ id: UInt32) -> PFDevice.Texture {
        return device.framebuffer_texture(texture_page_framebuffer(id))
    }

    func draw_viewport() -> PFRect<Int32> {
        if let render_target_id = render_target_stack.last {
            return self.render_target_location(render_target_id).rect
        }

        return self.main_viewport()
    }

    func main_viewport() -> PFRect<Int32> {
        switch self.options.dest {
        case .default(let viewport, _): return viewport
        case .other(let framebuffer):
            let texture = self.device.framebuffer_texture(framebuffer)
            let size = self.device.texture_size(texture)
            return .init(origin: .zero, size: size)
        }
    }

    func tile_size() -> SIMD2<Int32> {
        let temp =
            draw_viewport().size
            &+ .init(Int32(SceneBuilder.TILE_WIDTH) - 1, Int32(SceneBuilder.TILE_HEIGHT) - 1)
        return .init(
            temp.x / Int32(SceneBuilder.TILE_WIDTH),
            temp.y / Int32(SceneBuilder.TILE_HEIGHT)
        )
    }

    func pixel_size_to_tile_size(_ pixel_size: SIMD2<Int32>) -> SIMD2<Int32> {
        // Round up.
        let tile_size = SIMD2<Int32>(
            Int32(SceneBuilder.TILE_WIDTH) - 1,
            Int32(SceneBuilder.TILE_HEIGHT) - 1
        )
        let size = pixel_size &+ tile_size
        return .init(
            size.x / Int32(SceneBuilder.TILE_WIDTH),
            size.y / Int32(SceneBuilder.TILE_HEIGHT)
        )
    }

    func framebuffer_tile_size() -> SIMD2<Int32> {
        return pixel_size_to_tile_size(options.dest.window_size(device))
    }

    mutating func reallocate_alpha_tile_pages_if_necessary(_ copy_existing: Bool) {
        let alpha_tile_pages_needed = UInt32((alpha_tile_count + 0xffff) >> 16)
        if let mask_storage = mask_storage {
            if alpha_tile_pages_needed <= mask_storage.allocated_page_count {
                return
            }
        }

        let new_size = SIMD2<Int32>(
            Self.MASK_FRAMEBUFFER_WIDTH,
            Self.MASK_FRAMEBUFFER_HEIGHT * Int32(alpha_tile_pages_needed)
        )
        let format = TextureAllocation.TextureFormat.rgba8
        let mask_framebuffer_id = allocator.allocate_framebuffer(
            device,
            new_size,
            format,
            "TileAlphaMask"
        )
        let mask_framebuffer = allocator.get_framebuffer(mask_framebuffer_id)
        let old_mask_storage = mask_storage
        mask_storage = MaskStorage(
            framebuffer_id: mask_framebuffer_id,
            allocated_page_count: alpha_tile_pages_needed
        )

        // Copy over existing content if needed.
        let old_mask_framebuffer_id: UInt64?
        if let old_storage = old_mask_storage, copy_existing {
            old_mask_framebuffer_id = old_storage.framebuffer_id
        } else {
            return
        }

        guard let old_framebuffer_id = old_mask_framebuffer_id else { return }
        let old_mask_framebuffer = allocator.get_framebuffer(old_framebuffer_id)
        let old_mask_texture = device.framebuffer_texture(old_mask_framebuffer)
        let old_size = device.texture_size(old_mask_texture)

        var state = RenderState(
            target: .framebuffer(mask_framebuffer),
            program: programs.blit_program.program,
            vertex_array: vertex_arrays.blit_vertex_array.vertex_array,
            primitive: .triangles,
            uniforms: [
                (
                    programs.blit_program.framebuffer_size_uniform,
                    .vec2(.init(Float32(new_size.x), Float32(new_size.y)))
                ),
                (
                    programs.blit_program.dest_rect_uniform,
                    .vec4(.init(0.0, 0.0, Float32(old_size.x), Float32(old_size.y)))
                ),
            ],
            textures: [(programs.blit_program.src_texture, old_mask_texture)],
            images: [],
            storage_buffers: [],
            viewport: .init(origin: .zero, size: new_size),
            options: .init(
                clear_ops: .init(color: Color<Float>(r: 0.0, g: 0.0, b: 0.0, a: 0.0))
            )
        )

        device.draw_elements(6, &state)
        programs.blit_program.program = state.program
        programs.blit_program.framebuffer_size_uniform = state.uniforms[0].0
        programs.blit_program.dest_rect_uniform = state.uniforms[1].0
        programs.blit_program.src_texture = state.textures[0].0
        stats.drawcall_count += 1
    }

    func clear_color_for_draw_operation() -> Color<Float>? {
        let must_preserve_contents: Bool
        if let render_target_id = render_target_stack.last {
            let texture_page = render_target_location(render_target_id).page
            guard let pattern_texture_page = pattern_texture_pages[Int(texture_page)] else {
                fatalError("Draw target texture page not allocated!")
            }
            must_preserve_contents = pattern_texture_page.must_preserve_contents
        } else {
            must_preserve_contents = framebuffer_flags.contains(.DEST_FRAMEBUFFER_IS_DIRTY)
        }

        if must_preserve_contents {
            return nil
        } else if render_target_stack.isEmpty {
            return options.background_color
        } else {
            return Color<Float>(simd: .zero)
        }
    }

    mutating func preserve_draw_framebuffer() {
        if let render_target_id = render_target_stack.last {
            let texture_page = render_target_location(render_target_id).page
            guard pattern_texture_pages[Int(texture_page)] != nil else {
                fatalError("Draw target texture page not allocated!")
            }
            pattern_texture_pages[Int(texture_page)]!.must_preserve_contents = true
        } else {
            framebuffer_flags.formUnion(.DEST_FRAMEBUFFER_IS_DIRTY)
        }
    }
}

struct RenderState {
    enum Primitive {
        case triangles
        case lines

        func to_metal_primitive() -> MTLPrimitiveType {
            switch self {
            case .triangles: .triangle
            case .lines: .line
            }
        }
    }

    enum UniformData {
        case float(Float32)
        case ivec2(SIMD2<Int32>)
        case ivec3(SIMD3<Int32>)
        case int(Int32)
        case mat2(SIMD4<Float32>)
        case mat4(SIMD4<Float32>, SIMD4<Float32>, SIMD4<Float32>, SIMD4<Float32>)
        case vec2(SIMD2<Float32>)
        case vec3(SIMD3<Float32>)
        case vec4(SIMD4<Float32>)
    }

    enum ImageAccess {
        case read
        case write
        case readWrite
    }

    enum BlendFactor {
        case zero
        case one
        case srcAlpha
        case oneMinusSrcAlpha
        case destAlpha
        case oneMinusDestAlpha
        case destColor

        func to_metal_blend_factor() -> MTLBlendFactor {
            switch self {
            case .zero: .zero
            case .one: .one
            case .srcAlpha: .sourceAlpha
            case .oneMinusSrcAlpha: .oneMinusSourceAlpha
            case .destAlpha: .destinationAlpha
            case .oneMinusDestAlpha: .oneMinusDestinationAlpha
            case .destColor: .destinationColor
            }
        }
    }

    enum BlendOp {
        case add
        case subtract
        case reverseSubtract
        case min
        case max

        func to_metal_blend_op() -> MTLBlendOperation {
            switch self {
            case .add: .add
            case .subtract: .subtract
            case .reverseSubtract: .reverseSubtract
            case .min: .min
            case .max: .max
            }
        }
    }

    struct BlendState {
        var dest_rgb_factor: BlendFactor
        var dest_alpha_factor: BlendFactor
        var src_rgb_factor: BlendFactor
        var src_alpha_factor: BlendFactor
        var op: BlendOp
    }

    enum DepthFunc {
        case less
        case always

        func to_metal_compare_function() -> MTLCompareFunction {
            switch self {
            case .less: .less
            case .always: .always
            }
        }
    }

    enum StencilFunc {
        case always
        case equal

        func to_metal_compare_function() -> MTLCompareFunction {
            switch self {
            case .always: .always
            case .equal: .equal
            }
        }
    }

    struct DepthState {
        var `func`: DepthFunc
        var write: Bool
    }

    struct StencilState {
        var `func`: StencilFunc
        var reference: UInt32
        var mask: UInt32
        var write: Bool
    }

    struct ClearOps {
        var color: Color<Float>? = nil
        var depth: Float32? = nil
        var stencil: UInt8? = nil
    }

    struct RenderOptions {
        var blend: BlendState? = nil
        var depth: DepthState? = nil
        var stencil: StencilState? = nil
        var clear_ops: ClearOps = .init()
        var color_mask: Bool = true
    }

    typealias UniformBinding<U> = (U, UniformData)
    typealias TextureBinding<TP, T> = (TP, T)
    typealias ImageBinding<IP, T> = (IP, T, ImageAccess)

    var target: Renderer.RenderTarget
    var program: PFDevice.Program
    var vertex_array: PFDevice.VertexArray
    var primitive: Primitive
    var uniforms: [UniformBinding<PFDevice.Uniform>]
    var textures: [TextureBinding<PFDevice.TextureParameter, PFDevice.Texture>]
    var images: [ImageBinding<PFDevice.ImageParameter, PFDevice.Texture>]
    var storage_buffers: [(PFDevice.StorageBuffer, PFDevice.Buffer)]
    var viewport: PFRect<Int32>
    var options: RenderOptions
}

struct ComputeState {
    var program: PFDevice.Program
    var uniforms: [RenderState.UniformBinding<PFDevice.Uniform>]
    var textures: [RenderState.TextureBinding<PFDevice.TextureParameter, PFDevice.Texture>]
    var images: [RenderState.ImageBinding<PFDevice.ImageParameter, PFDevice.Texture>]
    var storage_buffers: [(PFDevice.StorageBuffer, PFDevice.Buffer)]
}

struct GPUMemoryAllocator {
    var general_buffers_in_use: [UInt64: BufferAllocation] = [:]
    var index_buffers_in_use: [UInt64: BufferAllocation] = [:]
    var textures_in_use: [UInt64: TextureAllocation] = [:]
    var framebuffers_in_use: [UInt64: FramebufferAllocation] = [:]
    var free_objects: [FreeObject] = .init()
    var next_general_buffer_id: UInt64 = 0
    var next_index_buffer_id: UInt64 = 0
    var next_texture_id: UInt64 = 0
    var next_framebuffer_id: UInt64 = 0
    var bytes_committed: UInt64 = 0
    var bytes_allocated: UInt64 = 0
}

struct BufferAllocation {
    var buffer: PFDevice.Buffer
    var size: UInt64
    var tag: String
}

struct TextureAllocation {
    enum TextureFormat {
        case r8
        case r16F
        case rgba8
        case rgba16F
        case rgba32F

        var bytes_per_pixel: Int {
            switch self {
            case .r8: return 1
            case .r16F: return 2
            case .rgba8: return 4
            case .rgba16F: return 8
            case .rgba32F: return 16
            }
        }
    }

    struct TextureDescriptor: Equatable {
        var width: UInt32
        var height: UInt32
        var format: TextureFormat

        var byte_size: UInt64 {
            UInt64(width) * UInt64(height) * UInt64(self.format.bytes_per_pixel)
        }
    }

    var texture: PFDevice.Texture
    var descriptor: TextureDescriptor
    var tag: String
}

struct FramebufferAllocation {
    var framebuffer: PFDevice.Framebuffer
    var descriptor: TextureAllocation.TextureDescriptor
    var tag: String
}

struct FreeObject {
    var timestamp: DispatchTime
    var kind: FreeObjectKind
}

enum FreeObjectKind {
    case generalBuffer(id: UInt64, allocation: BufferAllocation)
    case indexBuffer(id: UInt64, allocation: BufferAllocation)
    case texture(id: UInt64, allocation: TextureAllocation)
    case framebuffer(id: UInt64, allocation: FramebufferAllocation)
}

struct RendererOptions {
    enum DestFramebuffer {
        /// The rendered content should go to the default framebuffer (e.g. the window in OpenGL).
        case `default`(
            /// The rectangle within the window to draw in, in device pixels.
            viewport: PFRect<Int32>,
            /// The total size of the window in device pixels.
            window_size: SIMD2<Int32>
        )
        /// The rendered content should go to a non-default framebuffer (off-screen, typically).
        case other(PFDevice.Framebuffer)

        func window_size(_ device: PFDevice) -> SIMD2<Int32> {
            switch self {
            case .default(viewport: _, let window_size):
                return window_size
            case .other(let framebuffer):
                return device.texture_size(device.framebuffer_texture(framebuffer))
            }
        }

        static func full_window(_ window_size: SIMD2<Int32>) -> DestFramebuffer {
            let viewport = PFRect(origin: .zero, size: window_size)
            return .default(viewport: viewport, window_size: window_size)
        }
    }

    /// Where the rendering should go: either to the default framebuffer (i.e. screen) or to a
    /// custom framebuffer.
    var dest: DestFramebuffer
    /// The background color. If not present, transparent is assumed.
    var background_color: Color<Float>?
}

struct RendererFlags: OptionSet {
    let rawValue: UInt8

    // Whether we need a depth buffer.
    static let USE_DEPTH = RendererFlags(rawValue: 0x01)
    // Whether an intermediate destination framebuffer is needed.
    //
    // This will be true if any exotic blend modes are used at the top level (not inside a
    // render target), *and* the output framebuffer is the default framebuffer.
    static let INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED = RendererFlags(rawValue: 0x02)
}

struct RenderStats {
    /// The total number of path objects in the scene.
    var path_count: Int = 0
    /// The number of fill operations it took to render the scene.
    ///
    /// A fill operation is a single edge in a 16x16 device pixel tile.
    var fill_count: Int = 0
    /// The total number of 16x16 device pixel tile masks generated.
    var alpha_tile_count: Int = 0
    /// The total number of 16x16 tiles needed to render the scene, including both alpha tiles and
    /// solid-color tiles.
    var total_tile_count: Int = 0
    /// The amount of CPU time it took to build the scene.
    var cpu_build_time: DispatchTimeInterval = .never
    /// The number of GPU API draw calls it took to render the scene.
    var drawcall_count: UInt32 = 0
    /// The number of bytes of VRAM Pathfinder has allocated.
    ///
    /// This may be higher than `gpu_bytes_committed` because Pathfinder caches some data for
    /// faster reuse.
    var gpu_bytes_allocated: UInt64 = 0
    /// The number of bytes of VRAM Pathfinder actually used for the frame.
    var gpu_bytes_committed: UInt64 = 0
}

struct PendingTimer {
    enum TimerFuture {
        case pending(PFDevice.TimerQuery)
        case resolved(DispatchTimeInterval)
    }

    var dice_times: [TimerFuture] = []
    var bin_times: [TimerFuture] = []
    var fill_times: [TimerFuture] = []
    var composite_times: [TimerFuture] = []
    var other_times: [TimerFuture] = []
}

struct TimerQueryCache {
    var free_queries: [PFDevice.TimerQuery] = []
}

struct ProgramsCore {
    struct BlitProgram {
        var program: PFDevice.Program
        var dest_rect_uniform: PFDevice.Uniform
        var framebuffer_size_uniform: PFDevice.Uniform
        var src_texture: PFDevice.TextureParameter

        init(_ device: PFDevice) {
            program = device.create_raster_program("blit")
            dest_rect_uniform = device.get_uniform(program, "DestRect")
            framebuffer_size_uniform = device.get_uniform(program, "FramebufferSize")
            src_texture = device.get_texture_parameter(program, "Src")
        }
    }

    var blit_program: BlitProgram

    init(_ device: PFDevice) {
        blit_program = BlitProgram(device)
    }
}

struct VertexArraysCore {
    struct BlitVertexArray {
        var vertex_array: PFDevice.VertexArray

        init(
            _ device: PFDevice,
            _ blit_program: ProgramsCore.BlitProgram,
            _ quad_vertex_positions_buffer: PFDevice.Buffer,
            _ quad_vertex_indices_buffer: PFDevice.Buffer
        ) {
            var vertex_array = device.create_vertex_array()
            let position_attr = device.get_vertex_attr(blit_program.program, "Position")!

            device.bind_buffer(&vertex_array, quad_vertex_positions_buffer, .vertex)
            device.configure_vertex_attr(
                &vertex_array,
                position_attr,
                .init(
                    size: 2,
                    class: .int,
                    attr_type: .i16,
                    stride: 4,
                    offset: 0,
                    divisor: 0,
                    buffer_index: 0
                )
            )
            device.bind_buffer(&vertex_array, quad_vertex_indices_buffer, .index)

            self.vertex_array = vertex_array
        }
    }

    var blit_vertex_array: BlitVertexArray

    init(
        _ device: PFDevice,
        _ programs: ProgramsCore,
        _ quad_vertex_positions_buffer: PFDevice.Buffer,
        _ quad_vertex_indices_buffer: PFDevice.Buffer
    ) {
        blit_vertex_array = BlitVertexArray(
            device,
            programs.blit_program,
            quad_vertex_positions_buffer,
            quad_vertex_indices_buffer
        )
    }
}

struct RenderTargetInfo {
    var location: SceneBuilder.TextureLocation
}

struct PatternTexturePage {
    var framebuffer_id: UInt64
    var must_preserve_contents: Bool
}

struct MaskStorage {
    var framebuffer_id: UInt64
    var allocated_page_count: UInt32
}

struct RendererD3D11 {
    static let INITIAL_ALLOCATED_FILL_COUNT: UInt32 = 1024 * 16
    static let INITIAL_ALLOCATED_MICROLINE_COUNT: UInt32 = 1024 * 16

    struct SceneSourceBuffers {
        var points_buffer: UInt64? = nil
        var points_capacity: UInt32 = 0
        var point_indices_buffer: UInt64? = nil
        var point_indices_count: UInt32 = 0
        var point_indices_capacity: UInt32 = 0

        mutating func upload(
            _ allocator: inout GPUMemoryAllocator,
            _ device: PFDevice,
            _ segments: RenderCommand.SegmentsD3D11
        ) {
            let needed_points_capacity = UInt32(segments.points.count).nextPowerOfTwo
            let needed_point_indices_capacity = UInt32(segments.indices.count).nextPowerOfTwo

            if points_capacity < needed_points_capacity {
                points_buffer = allocator.allocate_general_buffer(
                    device,
                    Int(needed_points_capacity) * MemoryLayout<SIMD2<Float32>>.stride,
                    "PointsD3D11"
                )
                points_capacity = needed_points_capacity
            }

            if point_indices_capacity < needed_point_indices_capacity {
                point_indices_buffer = allocator.allocate_general_buffer(
                    device,
                    Int(needed_point_indices_capacity)
                        * MemoryLayout<RenderCommand.SegmentIndicesD3D11>.stride,
                    "PointIndicesD3D11"
                )
                point_indices_capacity = needed_point_indices_capacity
            }

            var pointsBuffer = allocator.general_buffers_in_use[points_buffer!]!
            device.upload_to_buffer(
                &pointsBuffer.buffer,
                0,
                segments.points,
                .storage
            )
            allocator.general_buffers_in_use[points_buffer!] = pointsBuffer

            var indicesBuffer = allocator.general_buffers_in_use[point_indices_buffer!]!
            device.upload_to_buffer(
                &indicesBuffer.buffer,
                0,
                segments.indices,
                .storage
            )
            allocator.general_buffers_in_use[point_indices_buffer!] = indicesBuffer

            point_indices_count = UInt32(segments.indices.count)
        }
    }

    struct SceneBuffers {
        var draw: SceneSourceBuffers = .init()
        var clip: SceneSourceBuffers = .init()

        mutating func upload(
            _ allocator: inout GPUMemoryAllocator,
            _ device: PFDevice,
            _ draw_segments: RenderCommand.SegmentsD3D11,
            _ clip_segments: RenderCommand.SegmentsD3D11
        ) {
            draw.upload(&allocator, device, draw_segments)
            clip.upload(&allocator, device, clip_segments)
        }
    }

    struct TileBatchInfoD3D11 {
        var tile_count: UInt32
        var z_buffer_id: UInt64
        var tiles_d3d11_buffer_id: UInt64
        var propagate_metadata_buffer_id: UInt64
        var first_tile_map_buffer_id: UInt64
    }

    var programs: ProgramsD3D11
    var allocated_microline_count: UInt32
    var allocated_fill_count: UInt32
    var scene_buffers: SceneBuffers
    var tile_batch_info: [Int: TileBatchInfoD3D11]

    init(_ core: RendererCore) {
        let programs = ProgramsD3D11(core.device)

        self.programs = programs
        allocated_fill_count = RendererD3D11.INITIAL_ALLOCATED_FILL_COUNT
        allocated_microline_count = RendererD3D11.INITIAL_ALLOCATED_MICROLINE_COUNT
        scene_buffers = SceneBuffers()
        tile_batch_info = [:]
    }
}

struct ProgramsD3D11 {
    static let BOUND_WORKGROUP_SIZE: UInt32 = 64
    static let DICE_WORKGROUP_SIZE: UInt32 = 64
    static let BIN_WORKGROUP_SIZE: UInt32 = 64
    static let PROPAGATE_WORKGROUP_SIZE: UInt32 = 64
    static let SORT_WORKGROUP_SIZE: UInt32 = 64

    struct ComputeDimensions {
        var x: UInt32
        var y: UInt32
        var z: UInt32

        func to_metal_size() -> MTLSize {
            return MTLSize(width: Int(x), height: Int(y), depth: Int(z))
        }
    }

    struct BoundProgramD3D11 {
        var program: PFDevice.Program
        var path_count_uniform: PFDevice.Uniform
        var tile_count_uniform: PFDevice.Uniform
        var tile_path_info_storage_buffer: PFDevice.StorageBuffer
        var tiles_storage_buffer: PFDevice.StorageBuffer

        init(_ device: PFDevice) {
            var program = device.create_compute_program("bound")
            let dimensions = ComputeDimensions(x: ProgramsD3D11.BOUND_WORKGROUP_SIZE, y: 1, z: 1)

            device.set_compute_program_local_size(&program, dimensions)

            let path_count_uniform = device.get_uniform(program, "PathCount")
            let tile_count_uniform = device.get_uniform(program, "TileCount")

            let tile_path_info_storage_buffer = device.get_storage_buffer(program, "TilePathInfo", 0)
            let tiles_storage_buffer = device.get_storage_buffer(program, "Tiles", 1)

            self.program = program
            self.path_count_uniform = path_count_uniform
            self.tile_count_uniform = tile_count_uniform
            self.tile_path_info_storage_buffer = tile_path_info_storage_buffer
            self.tiles_storage_buffer = tiles_storage_buffer
        }
    }

    struct DiceProgramD3D11 {
        var program: PFDevice.Program
        var transform_uniform: PFDevice.Uniform
        var translation_uniform: PFDevice.Uniform
        var path_count_uniform: PFDevice.Uniform
        var last_batch_segment_index_uniform: PFDevice.Uniform
        var max_microline_count_uniform: PFDevice.Uniform
        var compute_indirect_params_storage_buffer: PFDevice.StorageBuffer
        var dice_metadata_storage_buffer: PFDevice.StorageBuffer
        var points_storage_buffer: PFDevice.StorageBuffer
        var input_indices_storage_buffer: PFDevice.StorageBuffer
        var microlines_storage_buffer: PFDevice.StorageBuffer

        init(_ device: PFDevice) {
            var program = device.create_compute_program("dice")

            let dimensions = ComputeDimensions(x: ProgramsD3D11.DICE_WORKGROUP_SIZE, y: 1, z: 1)
            device.set_compute_program_local_size(&program, dimensions)

            let transform_uniform = device.get_uniform(program, "Transform")
            let translation_uniform = device.get_uniform(program, "Translation")
            let path_count_uniform = device.get_uniform(program, "PathCount")
            let last_batch_segment_index_uniform = device.get_uniform(program, "LastBatchSegmentIndex")
            let max_microline_count_uniform = device.get_uniform(program, "MaxMicrolineCount")

            let compute_indirect_params_storage_buffer = device.get_storage_buffer(
                program,
                "ComputeIndirectParams",
                0
            )
            let dice_metadata_storage_buffer = device.get_storage_buffer(program, "DiceMetadata", 1)
            let points_storage_buffer = device.get_storage_buffer(program, "Points", 2)
            let input_indices_storage_buffer = device.get_storage_buffer(program, "InputIndices", 3)
            let microlines_storage_buffer = device.get_storage_buffer(program, "Microlines", 4)

            self.program = program
            self.transform_uniform = transform_uniform
            self.translation_uniform = translation_uniform
            self.path_count_uniform = path_count_uniform
            self.last_batch_segment_index_uniform = last_batch_segment_index_uniform
            self.max_microline_count_uniform = max_microline_count_uniform
            self.compute_indirect_params_storage_buffer = compute_indirect_params_storage_buffer
            self.dice_metadata_storage_buffer = dice_metadata_storage_buffer
            self.points_storage_buffer = points_storage_buffer
            self.input_indices_storage_buffer = input_indices_storage_buffer
            self.microlines_storage_buffer = microlines_storage_buffer
        }
    }

    struct BinProgramD3D11 {
        var program: PFDevice.Program
        var microline_count_uniform: PFDevice.Uniform
        var max_fill_count_uniform: PFDevice.Uniform
        var microlines_storage_buffer: PFDevice.StorageBuffer
        var metadata_storage_buffer: PFDevice.StorageBuffer
        var indirect_draw_params_storage_buffer: PFDevice.StorageBuffer
        var fills_storage_buffer: PFDevice.StorageBuffer
        var tiles_storage_buffer: PFDevice.StorageBuffer
        var backdrops_storage_buffer: PFDevice.StorageBuffer

        init(_ device: PFDevice) {
            var program = device.create_compute_program("bin")
            let dimensions = ComputeDimensions(x: ProgramsD3D11.BIN_WORKGROUP_SIZE, y: 1, z: 1)
            device.set_compute_program_local_size(&program, dimensions)

            let microline_count_uniform = device.get_uniform(program, "MicrolineCount")
            let max_fill_count_uniform = device.get_uniform(program, "MaxFillCount")

            let microlines_storage_buffer = device.get_storage_buffer(program, "Microlines", 0)
            let metadata_storage_buffer = device.get_storage_buffer(program, "Metadata", 1)
            let indirect_draw_params_storage_buffer = device.get_storage_buffer(
                program,
                "IndirectDrawParams",
                2
            )
            let fills_storage_buffer = device.get_storage_buffer(program, "Fills", 3)
            let tiles_storage_buffer = device.get_storage_buffer(program, "Tiles", 4)
            let backdrops_storage_buffer = device.get_storage_buffer(program, "Backdrops", 5)

            self.program = program
            self.microline_count_uniform = microline_count_uniform
            self.max_fill_count_uniform = max_fill_count_uniform
            self.metadata_storage_buffer = metadata_storage_buffer
            self.indirect_draw_params_storage_buffer = indirect_draw_params_storage_buffer
            self.fills_storage_buffer = fills_storage_buffer
            self.tiles_storage_buffer = tiles_storage_buffer
            self.microlines_storage_buffer = microlines_storage_buffer
            self.backdrops_storage_buffer = backdrops_storage_buffer
        }
    }

    struct PropagateProgramD3D11 {
        var program: PFDevice.Program
        var framebuffer_tile_size_uniform: PFDevice.Uniform
        var column_count_uniform: PFDevice.Uniform
        var first_alpha_tile_index_uniform: PFDevice.Uniform
        var draw_metadata_storage_buffer: PFDevice.StorageBuffer
        var clip_metadata_storage_buffer: PFDevice.StorageBuffer
        var backdrops_storage_buffer: PFDevice.StorageBuffer
        var draw_tiles_storage_buffer: PFDevice.StorageBuffer
        var clip_tiles_storage_buffer: PFDevice.StorageBuffer
        var z_buffer_storage_buffer: PFDevice.StorageBuffer
        var first_tile_map_storage_buffer: PFDevice.StorageBuffer
        var alpha_tiles_storage_buffer: PFDevice.StorageBuffer

        init(_ device: PFDevice) {
            var program = device.create_compute_program("propagate")
            let local_size = ComputeDimensions(x: ProgramsD3D11.PROPAGATE_WORKGROUP_SIZE, y: 1, z: 1)
            device.set_compute_program_local_size(&program, local_size)

            let framebuffer_tile_size_uniform = device.get_uniform(program, "FramebufferTileSize")
            let column_count_uniform = device.get_uniform(program, "ColumnCount")
            let first_alpha_tile_index_uniform = device.get_uniform(program, "FirstAlphaTileIndex")
            let draw_metadata_storage_buffer = device.get_storage_buffer(program, "DrawMetadata", 0)
            let clip_metadata_storage_buffer = device.get_storage_buffer(program, "ClipMetadata", 1)
            let backdrops_storage_buffer = device.get_storage_buffer(program, "Backdrops", 2)
            let draw_tiles_storage_buffer = device.get_storage_buffer(program, "DrawTiles", 3)
            let clip_tiles_storage_buffer = device.get_storage_buffer(program, "ClipTiles", 4)
            let z_buffer_storage_buffer = device.get_storage_buffer(program, "ZBuffer", 5)
            let first_tile_map_storage_buffer = device.get_storage_buffer(program, "FirstTileMap", 6)
            let alpha_tiles_storage_buffer = device.get_storage_buffer(program, "AlphaTiles", 7)

            self.program = program
            self.framebuffer_tile_size_uniform = framebuffer_tile_size_uniform
            self.column_count_uniform = column_count_uniform
            self.first_alpha_tile_index_uniform = first_alpha_tile_index_uniform
            self.draw_metadata_storage_buffer = draw_metadata_storage_buffer
            self.clip_metadata_storage_buffer = clip_metadata_storage_buffer
            self.backdrops_storage_buffer = backdrops_storage_buffer
            self.draw_tiles_storage_buffer = draw_tiles_storage_buffer
            self.clip_tiles_storage_buffer = clip_tiles_storage_buffer
            self.z_buffer_storage_buffer = z_buffer_storage_buffer
            self.first_tile_map_storage_buffer = first_tile_map_storage_buffer
            self.alpha_tiles_storage_buffer = alpha_tiles_storage_buffer
        }
    }

    struct SortProgramD3D11 {
        var program: PFDevice.Program
        var tile_count_uniform: PFDevice.Uniform
        var tiles_storage_buffer: PFDevice.StorageBuffer
        var first_tile_map_storage_buffer: PFDevice.StorageBuffer
        var z_buffer_storage_buffer: PFDevice.StorageBuffer

        init(_ device: PFDevice) {
            var program = device.create_compute_program("sort")
            let dimensions = ComputeDimensions(x: ProgramsD3D11.SORT_WORKGROUP_SIZE, y: 1, z: 1)
            device.set_compute_program_local_size(&program, dimensions)

            let tile_count_uniform = device.get_uniform(program, "TileCount")
            let tiles_storage_buffer = device.get_storage_buffer(program, "Tiles", 0)
            let first_tile_map_storage_buffer = device.get_storage_buffer(program, "FirstTileMap", 1)
            let z_buffer_storage_buffer = device.get_storage_buffer(program, "ZBuffer", 2)

            self.program = program
            self.tile_count_uniform = tile_count_uniform
            self.tiles_storage_buffer = tiles_storage_buffer
            self.first_tile_map_storage_buffer = first_tile_map_storage_buffer
            self.z_buffer_storage_buffer = z_buffer_storage_buffer
        }
    }

    struct FillProgramD3D11 {
        var program: PFDevice.Program
        var dest_image: PFDevice.ImageParameter
        var area_lut_texture: PFDevice.TextureParameter
        var alpha_tile_range_uniform: PFDevice.Uniform
        var fills_storage_buffer: PFDevice.StorageBuffer
        var tiles_storage_buffer: PFDevice.StorageBuffer
        var alpha_tiles_storage_buffer: PFDevice.StorageBuffer

        init(_ device: PFDevice) {
            var program = device.create_compute_program("fill")
            let local_size = ComputeDimensions(
                x: SceneBuilder.TILE_WIDTH,
                y: SceneBuilder.TILE_HEIGHT / 4,
                z: 1
            )
            device.set_compute_program_local_size(&program, local_size)

            let dest_image = device.get_image_parameter(program, "Dest")
            let area_lut_texture = device.get_texture_parameter(program, "AreaLUT")
            let alpha_tile_range_uniform = device.get_uniform(program, "AlphaTileRange")
            let fills_storage_buffer = device.get_storage_buffer(program, "Fills", 0)
            let tiles_storage_buffer = device.get_storage_buffer(program, "Tiles", 1)
            let alpha_tiles_storage_buffer = device.get_storage_buffer(program, "AlphaTiles", 2)

            self.program = program
            self.dest_image = dest_image
            self.area_lut_texture = area_lut_texture
            self.alpha_tile_range_uniform = alpha_tile_range_uniform
            self.fills_storage_buffer = fills_storage_buffer
            self.tiles_storage_buffer = tiles_storage_buffer
            self.alpha_tiles_storage_buffer = alpha_tiles_storage_buffer
        }
    }

    struct TileProgramCommon {
        var program: PFDevice.Program
        var tile_size_uniform: PFDevice.Uniform
        var texture_metadata_texture: PFDevice.TextureParameter
        var texture_metadata_size_uniform: PFDevice.Uniform
        var z_buffer_texture: PFDevice.TextureParameter
        var z_buffer_texture_size_uniform: PFDevice.Uniform
        var color_texture_0: PFDevice.TextureParameter
        var color_texture_size_0_uniform: PFDevice.Uniform
        var mask_texture_0: PFDevice.TextureParameter
        var mask_texture_size_0_uniform: PFDevice.Uniform
        var gamma_lut_texture: PFDevice.TextureParameter
        var framebuffer_size_uniform: PFDevice.Uniform

        init(_ device: PFDevice, _ program: PFDevice.Program) {
            let tile_size_uniform = device.get_uniform(program, "TileSize")
            let texture_metadata_texture = device.get_texture_parameter(program, "TextureMetadata")
            let texture_metadata_size_uniform = device.get_uniform(program, "TextureMetadataSize")
            let z_buffer_texture = device.get_texture_parameter(program, "ZBuffer")
            let z_buffer_texture_size_uniform = device.get_uniform(program, "ZBufferSize")
            let color_texture_0 = device.get_texture_parameter(program, "ColorTexture0")
            let color_texture_size_0_uniform = device.get_uniform(program, "ColorTextureSize0")
            let mask_texture_0 = device.get_texture_parameter(program, "MaskTexture0")
            let mask_texture_size_0_uniform = device.get_uniform(program, "MaskTextureSize0")
            let gamma_lut_texture = device.get_texture_parameter(program, "GammaLUT")
            let framebuffer_size_uniform = device.get_uniform(program, "FramebufferSize")

            self.program = program
            self.tile_size_uniform = tile_size_uniform
            self.texture_metadata_texture = texture_metadata_texture
            self.texture_metadata_size_uniform = texture_metadata_size_uniform
            self.z_buffer_texture = z_buffer_texture
            self.z_buffer_texture_size_uniform = z_buffer_texture_size_uniform
            self.color_texture_0 = color_texture_0
            self.color_texture_size_0_uniform = color_texture_size_0_uniform
            self.mask_texture_0 = mask_texture_0
            self.mask_texture_size_0_uniform = mask_texture_size_0_uniform
            self.gamma_lut_texture = gamma_lut_texture
            self.framebuffer_size_uniform = framebuffer_size_uniform
        }
    }

    struct TileProgramD3D11 {
        var common: TileProgramCommon
        var load_action_uniform: PFDevice.Uniform
        var clear_color_uniform: PFDevice.Uniform
        var framebuffer_tile_size_uniform: PFDevice.Uniform
        var dest_image: PFDevice.ImageParameter
        var tiles_storage_buffer: PFDevice.StorageBuffer
        var first_tile_map_storage_buffer: PFDevice.StorageBuffer

        init(_ device: PFDevice) {
            var program = device.create_compute_program("tile")
            device.set_compute_program_local_size(&program, .init(x: 16, y: 4, z: 1))

            let load_action_uniform = device.get_uniform(program, "LoadAction")
            let clear_color_uniform = device.get_uniform(program, "ClearColor")
            let framebuffer_tile_size_uniform = device.get_uniform(program, "FramebufferTileSize")
            let dest_image = device.get_image_parameter(program, "DestImage")
            let tiles_storage_buffer = device.get_storage_buffer(program, "Tiles", 0)
            let first_tile_map_storage_buffer = device.get_storage_buffer(program, "FirstTileMap", 1)

            let common = TileProgramCommon(device, program)

            self.common = common
            self.load_action_uniform = load_action_uniform
            self.clear_color_uniform = clear_color_uniform
            self.framebuffer_tile_size_uniform = framebuffer_tile_size_uniform
            self.dest_image = dest_image
            self.tiles_storage_buffer = tiles_storage_buffer
            self.first_tile_map_storage_buffer = first_tile_map_storage_buffer
        }
    }

    var bound_program: BoundProgramD3D11
    var dice_program: DiceProgramD3D11
    var bin_program: BinProgramD3D11
    var propagate_program: PropagateProgramD3D11
    var sort_program: SortProgramD3D11
    var fill_program: FillProgramD3D11
    var tile_program: TileProgramD3D11

    init(_ device: PFDevice) {
        bound_program = BoundProgramD3D11(device)
        dice_program = DiceProgramD3D11(device)
        bin_program = BinProgramD3D11(device)
        propagate_program = PropagateProgramD3D11(device)
        sort_program = SortProgramD3D11(device)
        fill_program = FillProgramD3D11(device)
        tile_program = TileProgramD3D11(device)
    }
}

struct Frame {
    struct ClearVertexArray {
        var vertex_array: PFDevice.VertexArray

        init(
            _ device: PFDevice,
            _ clear_program: Renderer.ClearProgram,
            _ quad_vertex_positions_buffer: PFDevice.Buffer,
            _ quad_vertex_indices_buffer: PFDevice.Buffer
        ) {
            var vertex_array = device.create_vertex_array()
            let position_attr = device.get_vertex_attr(clear_program.program, "Position")!

            device.bind_buffer(&vertex_array, quad_vertex_positions_buffer, .vertex)
            device.configure_vertex_attr(
                &vertex_array,
                position_attr,
                .init(
                    size: 2,
                    class: .int,
                    attr_type: .i16,
                    stride: 4,
                    offset: 0,
                    divisor: 0,
                    buffer_index: 0
                )
            )
            device.bind_buffer(&vertex_array, quad_vertex_indices_buffer, .index)

            self.vertex_array = vertex_array
        }
    }

    struct StencilVertexArray {
        var vertex_array: PFDevice.VertexArray
        var vertex_buffer: PFDevice.Buffer
        var index_buffer: PFDevice.Buffer

        init(_ device: PFDevice, _ stencil_program: Renderer.StencilProgram) {
            var vertex_array = device.create_vertex_array()
            let vertex_buffer = device.create_buffer(.static)
            let index_buffer = device.create_buffer(.static)

            let position_attr = device.get_vertex_attr(stencil_program.program, "Position")!

            device.bind_buffer(&vertex_array, vertex_buffer, .vertex)
            device.configure_vertex_attr(
                &vertex_array,
                position_attr,
                .init(
                    size: 3,
                    class: .float,
                    attr_type: .f32,
                    stride: 4 * 4,
                    offset: 0,
                    divisor: 0,
                    buffer_index: 0
                )
            )
            device.bind_buffer(&vertex_array, index_buffer, .index)

            self.vertex_array = vertex_array
            self.vertex_buffer = vertex_buffer
            self.index_buffer = index_buffer
        }
    }

    struct ReprojectionVertexArray {
        var vertex_array: PFDevice.VertexArray

        init(
            _ device: PFDevice,
            _ reprojection_program: Renderer.ReprojectionProgram,
            _ quad_vertex_positions_buffer: PFDevice.Buffer,
            _ quad_vertex_indices_buffer: PFDevice.Buffer
        ) {
            var vertex_array = device.create_vertex_array()
            let position_attr = device.get_vertex_attr(reprojection_program.program, "Position")!

            device.bind_buffer(&vertex_array, quad_vertex_positions_buffer, .vertex)
            device.configure_vertex_attr(
                &vertex_array,
                position_attr,
                .init(
                    size: 2,
                    class: .int,
                    attr_type: .i16,
                    stride: 4,
                    offset: 0,
                    divisor: 0,
                    buffer_index: 0
                )
            )
            device.bind_buffer(&vertex_array, quad_vertex_indices_buffer, .index)

            self.vertex_array = vertex_array
        }
    }

    var blit_vertex_array: VertexArraysCore.BlitVertexArray
    var clear_vertex_array: ClearVertexArray
    var stencil_vertex_array: StencilVertexArray
    var reprojection_vertex_array: ReprojectionVertexArray

    init(
        _ device: PFDevice,
        _ allocator: GPUMemoryAllocator,
        _ blit_program: ProgramsCore.BlitProgram,
        _ clear_program: Renderer.ClearProgram,
        _ reprojection_program: Renderer.ReprojectionProgram,
        _ stencil_program: Renderer.StencilProgram,
        _ quad_vertex_positions_buffer_id: UInt64,
        _ quad_vertex_indices_buffer_id: UInt64
    ) {
        let quad_vertex_positions_buffer = allocator.get_general_buffer(quad_vertex_positions_buffer_id)
        let quad_vertex_indices_buffer = allocator.get_index_buffer(quad_vertex_indices_buffer_id)

        let blit_vertex_array = VertexArraysCore.BlitVertexArray(
            device,
            blit_program,
            quad_vertex_positions_buffer,
            quad_vertex_indices_buffer
        )
        let clear_vertex_array = Frame.ClearVertexArray(
            device,
            clear_program,
            quad_vertex_positions_buffer,
            quad_vertex_indices_buffer
        )
        let reprojection_vertex_array = Frame.ReprojectionVertexArray(
            device,
            reprojection_program,
            quad_vertex_positions_buffer,
            quad_vertex_indices_buffer
        )
        let stencil_vertex_array = Frame.StencilVertexArray(device, stencil_program)

        self.blit_vertex_array = blit_vertex_array
        self.clear_vertex_array = clear_vertex_array
        self.reprojection_vertex_array = reprojection_vertex_array
        self.stencil_vertex_array = stencil_vertex_array
    }
}

extension GPUMemoryAllocator {
    static let MAX_BUFFER_SIZE_CLASS: UInt64 = 16 * 1024 * 1024
    static let REUSE_TIME: Double = 0.015

    mutating func allocate_general_buffer(_ device: PFDevice, _ size: Int, _ tag: String) -> UInt64 {
        var byte_size = UInt64(size)
        if byte_size < GPUMemoryAllocator.MAX_BUFFER_SIZE_CLASS {
            byte_size = byte_size.nextPowerOfTwo
        }

        let now = DispatchTime.now()

        for free_object_index in 0..<self.free_objects.count {
            let object = self.free_objects[free_object_index]

            guard
                case .generalBuffer(id: _, allocation: let allocation) = object.kind,
                allocation.size == byte_size
                    && (object.timestamp.distance(to: now).seconds >= GPUMemoryAllocator.REUSE_TIME)
            else { continue }

            let element = self.free_objects.remove(at: free_object_index)

            guard case .generalBuffer(id: let id, allocation: var allocation) = element.kind else {
                fatalError()
            }

            allocation.tag = tag
            self.bytes_committed += allocation.size
            self.general_buffers_in_use[id] = allocation
            return id
        }

        var buffer = device.create_buffer(.dynamic)
        device.allocate_buffer(
            &buffer,
            PFDevice.BufferData<UInt8>.uninitialized(Int(byte_size)),
            .vertex
        )

        let id = self.next_general_buffer_id
        self.next_general_buffer_id += 1

        self.general_buffers_in_use[id] = .init(buffer: buffer, size: byte_size, tag: tag)
        self.bytes_allocated += byte_size
        self.bytes_committed += byte_size

        return id
    }

    mutating func allocate_index_buffer(_ device: PFDevice, _ size: Int, _ tag: String) -> UInt64 {
        var byte_size = UInt64(size)
        if byte_size < GPUMemoryAllocator.MAX_BUFFER_SIZE_CLASS {
            byte_size = byte_size.nextPowerOfTwo
        }

        let now = DispatchTime.now()

        for free_object_index in 0..<self.free_objects.count {
            let object = self.free_objects[free_object_index]

            guard
                case .indexBuffer(id: _, allocation: let allocation) = object.kind,
                allocation.size == byte_size
                    && (object.timestamp.distance(to: now).seconds >= GPUMemoryAllocator.REUSE_TIME)
            else { continue }

            let element = self.free_objects.remove(at: free_object_index)

            guard case .indexBuffer(id: let id, allocation: var allocation) = element.kind else {
                fatalError()
            }

            allocation.tag = tag
            self.bytes_committed += allocation.size
            self.index_buffers_in_use[id] = allocation
            return id
        }

        var buffer = device.create_buffer(.dynamic)
        device.allocate_buffer(
            &buffer,
            PFDevice.BufferData<UInt8>.uninitialized(Int(byte_size)),
            .index
        )

        let id = self.next_index_buffer_id
        self.next_index_buffer_id += 1

        self.index_buffers_in_use[id] = .init(buffer: buffer, size: byte_size, tag: tag)
        self.bytes_allocated += byte_size
        self.bytes_committed += byte_size

        return id
    }

    mutating func allocate_texture(
        _ device: PFDevice,
        _ size: SIMD2<Int32>,
        _ format: TextureAllocation.TextureFormat,
        _ tag: String
    ) -> UInt64 {
        let descriptor = TextureAllocation.TextureDescriptor(
            width: UInt32(size.x),
            height: UInt32(size.y),
            format: format
        )

        let byte_size = descriptor.byte_size

        for free_object_index in 0..<self.free_objects.count {
            let object = self.free_objects[free_object_index]

            guard
                case .texture(id: _, allocation: let allocation) = object.kind,
                allocation.descriptor == descriptor
            else { continue }

            let element = self.free_objects.remove(at: free_object_index)

            guard case .texture(id: let id, allocation: var allocation) = element.kind else {
                fatalError()
            }

            allocation.tag = tag
            self.bytes_committed += allocation.descriptor.byte_size
            self.textures_in_use[id] = allocation
            return id
        }

        let texture = device.create_texture(format, size)
        let id = self.next_texture_id
        self.next_texture_id += 1

        self.textures_in_use[id] = .init(texture: texture, descriptor: descriptor, tag: tag)

        self.bytes_allocated += byte_size
        self.bytes_committed += byte_size

        return id
    }

    mutating func allocate_framebuffer(
        _ device: PFDevice,
        _ size: SIMD2<Int32>,
        _ format: TextureAllocation.TextureFormat,
        _ tag: String
    ) -> UInt64 {
        let descriptor = TextureAllocation.TextureDescriptor(
            width: UInt32(size.x),
            height: UInt32(size.y),
            format: format
        )
        let byte_size = descriptor.byte_size

        for free_object_index in 0..<self.free_objects.count {
            let object = self.free_objects[free_object_index]

            guard
                case .framebuffer(id: _, allocation: let allocation) = object.kind,
                allocation.descriptor == descriptor
            else { continue }

            let element = self.free_objects.remove(at: free_object_index)

            guard case .framebuffer(id: let id, allocation: var allocation) = element.kind else {
                fatalError()
            }

            allocation.tag = tag
            self.bytes_committed += allocation.descriptor.byte_size
            self.framebuffers_in_use[id] = allocation
            return id
        }

        let texture = device.create_texture(format, size)
        let framebuffer = device.create_framebuffer(texture)

        let id = self.next_framebuffer_id
        self.next_framebuffer_id += 1

        self.framebuffers_in_use[id] = .init(framebuffer: framebuffer, descriptor: descriptor, tag: tag)

        self.bytes_allocated += byte_size
        self.bytes_committed += byte_size

        return id
    }

    func get_general_buffer(_ id: UInt64) -> PFDevice.Buffer {
        self.general_buffers_in_use[id]!.buffer
    }

    func get_index_buffer(_ id: UInt64) -> PFDevice.Buffer {
        self.index_buffers_in_use[id]!.buffer
    }

    func get_framebuffer(_ id: UInt64) -> PFDevice.Framebuffer {
        self.framebuffers_in_use[id]!.framebuffer
    }

    func get_texture(_ id: UInt64) -> PFDevice.Texture {
        self.textures_in_use[id]!.texture
    }

    mutating func setGeneralBuffer(_ id: UInt64, _ buffer: PFDevice.Buffer) {
        guard var value = self.general_buffers_in_use[id] else { return }
        value.buffer = buffer
        self.general_buffers_in_use[id] = value
    }

    mutating func free_framebuffer(_ id: UInt64) {
        let allocation = self.framebuffers_in_use.removeValue(forKey: id)!
        let byte_size = allocation.descriptor.byte_size
        self.bytes_committed -= byte_size
        self.free_objects.append(
            .init(
                timestamp: .now(),
                kind: .framebuffer(id: id, allocation: allocation)
            )
        )
    }

    mutating func free_general_buffer(_ id: UInt64) {
        guard let allocation = general_buffers_in_use.removeValue(forKey: id) else {
            fatalError("Attempted to free unallocated general buffer!")
        }

        bytes_committed -= allocation.size
        free_objects.append(
            FreeObject(
                timestamp: .now(),
                kind: .generalBuffer(id: id, allocation: allocation)
            )
        )
    }

    mutating func purge_if_needed() {
        let now = DispatchTime.now()
        while true {
            guard let first_object = free_objects.first else { break }

            if first_object.timestamp.distance(to: now).seconds < GPUMemoryAllocator.REUSE_TIME {
                break
            }

            let free_object = free_objects.removeFirst()

            switch free_object.kind {
            case .generalBuffer(_, let allocation):
                bytes_allocated -= allocation.size
            case .indexBuffer(_, let allocation):
                bytes_allocated -= allocation.size
            case .texture(_, let allocation):
                bytes_allocated -= allocation.descriptor.byte_size
            case .framebuffer(_, let allocation):
                bytes_allocated -= allocation.descriptor.byte_size
            }
        }
    }
}

extension Renderer {
    static let SQRT_2_PI_INV: Float32 = 0.3989422804014327

    static let QUAD_VERTEX_POSITIONS: [UInt16] = [0, 0, 1, 0, 1, 1, 0, 1]
    static let QUAD_VERTEX_INDICES: [UInt32] = [0, 1, 3, 1, 2, 3]
    static let TEXTURE_METADATA_ENTRIES_PER_ROW: Int32 = 128
    static let TEXTURE_METADATA_TEXTURE_WIDTH: Int32 = TEXTURE_METADATA_ENTRIES_PER_ROW * 10
    static let TEXTURE_METADATA_TEXTURE_HEIGHT: Int32 = 65536 / TEXTURE_METADATA_ENTRIES_PER_ROW

    static let COMBINER_CTRL_COLOR_FILTER_SHIFT: Int32 = 4
    static let COMBINER_CTRL_COLOR_COMBINE_SHIFT: Int32 = 8
    static let COMBINER_CTRL_COMPOSITE_SHIFT: Int32 = 10

    static let COMBINER_CTRL_FILTER_RADIAL_GRADIENT: Int32 = 0x1
    static let COMBINER_CTRL_FILTER_BLUR: Int32 = 0x3
    static let COMBINER_CTRL_FILTER_COLOR_MATRIX: Int32 = 0x4

    init(device: PFDevice, options: RendererOptions) {
        var allocator = GPUMemoryAllocator()

        device.begin_commands()

        let quad_vertex_positions_buffer_id = allocator.allocate_general_buffer(
            device,
            Renderer.QUAD_VERTEX_POSITIONS.count * MemoryLayout<UInt16>.stride,
            "QuadVertexPositions"
        )

        var buffer = allocator.get_general_buffer(quad_vertex_positions_buffer_id)
        device.upload_to_buffer(&buffer, 0, Renderer.QUAD_VERTEX_POSITIONS, .vertex)
        allocator.setGeneralBuffer(quad_vertex_positions_buffer_id, buffer)

        let quad_vertex_indices_buffer_id = allocator.allocate_index_buffer(
            device,
            Renderer.QUAD_VERTEX_INDICES.count * MemoryLayout<UInt32>.stride,
            "QuadVertexIndices"
        )

        var indexBuffer = allocator.index_buffers_in_use[quad_vertex_indices_buffer_id]!
        device.upload_to_buffer(&indexBuffer.buffer, 0, Renderer.QUAD_VERTEX_INDICES, .index)
        allocator.index_buffers_in_use[quad_vertex_indices_buffer_id] = indexBuffer

        let area_lut_texture_id = allocator.allocate_texture(
            device,
            SIMD2<Int32>(repeating: 256),
            .rgba8,
            "AreaLUT"
        )
        let gamma_lut_texture_id = allocator.allocate_texture(
            device,
            SIMD2<Int32>(256, 8),
            .r8,
            "GammaLUT"
        )

        var areaLutTexture = allocator.textures_in_use[area_lut_texture_id]!
        device.upload_png_to_texture("area-lut", &areaLutTexture.texture, .rgba8)
        allocator.textures_in_use[area_lut_texture_id] = areaLutTexture

        var gammaLutTexture = allocator.textures_in_use[gamma_lut_texture_id]!
        device.upload_png_to_texture("gamma-lut", &gammaLutTexture.texture, .r8)
        allocator.textures_in_use[gamma_lut_texture_id] = gammaLutTexture

        let window_size = options.dest.window_size(device)
        let intermediate_dest_framebuffer_id = allocator.allocate_framebuffer(
            device,
            window_size,
            .rgba8,
            "IntermediateDest"
        )

        let texture_metadata_texture_size = SIMD2<Int32>(
            Renderer.TEXTURE_METADATA_TEXTURE_WIDTH,
            Renderer.TEXTURE_METADATA_TEXTURE_HEIGHT
        )

        let texture_metadata_texture_id = allocator.allocate_texture(
            device,
            texture_metadata_texture_size,
            .rgba16F,
            "TextureMetadata"
        )

        let core_programs = ProgramsCore(device)
        let core_vertex_arrays = VertexArraysCore(
            device,
            core_programs,
            allocator.get_general_buffer(quad_vertex_positions_buffer_id),
            allocator.get_index_buffer(quad_vertex_indices_buffer_id)
        )

        let core = RendererCore(
            device: device,
            allocator: allocator,
            options: options,
            renderer_flags: .init(),
            stats: .init(),
            current_timer: nil,
            timer_query_cache: .init(),

            programs: core_programs,
            vertex_arrays: core_vertex_arrays,

            quad_vertex_positions_buffer_id: quad_vertex_positions_buffer_id,
            quad_vertex_indices_buffer_id: quad_vertex_indices_buffer_id,
            area_lut_texture_id: area_lut_texture_id,
            gamma_lut_texture_id: gamma_lut_texture_id,

            intermediate_dest_framebuffer_id: intermediate_dest_framebuffer_id,
            intermediate_dest_framebuffer_size: window_size,

            texture_metadata_texture_id: texture_metadata_texture_id,
            render_targets: [],
            render_target_stack: [],
            pattern_texture_pages: [],
            mask_storage: nil,
            alpha_tile_count: 0,
            framebuffer_flags: .init()
        )

        let level_impl = RendererD3D11(core)

        let blit_program = ProgramsCore.BlitProgram(core.device)
        let clear_program = Renderer.ClearProgram(core.device)
        let stencil_program = Renderer.StencilProgram(core.device)
        let reprojection_program = Renderer.ReprojectionProgram(core.device)

        let frame = Frame(
            core.device,
            core.allocator,
            blit_program,
            clear_program,
            reprojection_program,
            stencil_program,
            quad_vertex_positions_buffer_id,
            quad_vertex_indices_buffer_id
        )

        core.device.end_commands()

        self.core = core
        self.level_impl = level_impl

        self.blit_program = blit_program
        self.clear_program = clear_program

        self.frame = frame

        self.stencil_program = stencil_program
        self.reprojection_program = reprojection_program
    }

    func present(drawable: CAMetalDrawable) {
        core.device.present_drawable(drawable)
    }

    mutating func begin_scene() {
        self.core.framebuffer_flags = .init()

        self.core.device.begin_commands()
        self.core.current_timer = PendingTimer()
        self.core.stats = .init()

        self.core.alpha_tile_count = 0
    }

    mutating func end_scene() {
        clear_dest_framebuffer_if_necessary()
        blit_intermediate_dest_framebuffer_if_necessary()

        core.stats.gpu_bytes_allocated = core.allocator.bytes_allocated
        core.stats.gpu_bytes_committed = core.allocator.bytes_committed

        level_impl.end_frame(&core)

        core.allocator.purge_if_needed()
        core.device.end_commands()
    }

    mutating func clear_dest_framebuffer_if_necessary() {
        guard let background_color = core.options.background_color else {
            return
        }

        if core.framebuffer_flags.contains(.DEST_FRAMEBUFFER_IS_DIRTY) {
            return
        }

        let main_viewport = core.main_viewport()
        let uniforms = [
            (
                clear_program.rect_uniform,
                RenderState.UniformData.vec4(main_viewport.f32.value)
            ),
            (
                clear_program.framebuffer_size_uniform,
                .vec2(.init(main_viewport.size))
            ),
            (
                clear_program.color_uniform,
                .vec4(background_color.simd)
            ),
        ]

        var state = RenderState(
            target: .default,
            program: clear_program.program,
            vertex_array: frame.clear_vertex_array.vertex_array,
            primitive: .triangles,
            uniforms: uniforms,
            textures: [],
            images: [],
            storage_buffers: [],
            viewport: main_viewport,
            options: .init()
        )

        core.device.draw_elements(6, &state)
        clear_program.program = state.program
        clear_program.rect_uniform = state.uniforms[0].0
        clear_program.framebuffer_size_uniform = state.uniforms[1].0
        clear_program.color_uniform = state.uniforms[2].0
        core.stats.drawcall_count += 1
    }

    mutating func blit_intermediate_dest_framebuffer_if_necessary() {
        if !core.renderer_flags.contains(.INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED) {
            return
        }

        let main_viewport = core.main_viewport()

        if core.intermediate_dest_framebuffer_size != main_viewport.size {
            core.allocator.free_framebuffer(core.intermediate_dest_framebuffer_id)
            core.intermediate_dest_framebuffer_id =
                core.allocator.allocate_framebuffer(
                    core.device,
                    main_viewport.size,
                    .rgba8,
                    "IntermediateDest"
                )
            core.intermediate_dest_framebuffer_size = main_viewport.size
        }

        let intermediate_dest_framebuffer =
            core.allocator.get_framebuffer(core.intermediate_dest_framebuffer_id)

        let textures = [
            (blit_program.src_texture, core.device.framebuffer_texture(intermediate_dest_framebuffer))
        ]

        var state = RenderState(
            target: .default,
            program: blit_program.program,
            vertex_array: frame.blit_vertex_array.vertex_array,
            primitive: .triangles,
            uniforms: [
                (blit_program.framebuffer_size_uniform, .vec2(.init(main_viewport.size))),
                (
                    blit_program.dest_rect_uniform,
                    .vec4(PFRect<Float32>(origin: .zero, size: .init(main_viewport.size)).value)
                ),
            ],
            textures: textures,
            images: [],
            storage_buffers: [],
            viewport: main_viewport,
            options: .init(
                clear_ops: .init(color: Color<Float>(r: 0.0, g: 0.0, b: 0.0, a: 1.0))
            )
        )

        core.device.draw_elements(6, &state)
        blit_program.program = state.program
        blit_program.framebuffer_size_uniform = state.uniforms[0].0
        blit_program.dest_rect_uniform = state.uniforms[1].0
        blit_program.src_texture = state.textures[0].0
        core.stats.drawcall_count += 1
    }

    mutating func render_command(command: RenderCommand) {
        switch command {
        case .start(let path_count, let bounding_quad, let needs_readable_framebuffer):
            self.start_rendering(bounding_quad, path_count, needs_readable_framebuffer)
        case .allocateTexturePage(let page_id, let descriptor):
            self.allocate_pattern_texture_page(page_id, descriptor)
        case .uploadTexelData(let texels, let location):
            self.upload_texel_data(texels, location)
        case .declareRenderTarget(let id, let location):
            self.declare_render_target(id, location)
        case .uploadTextureMetadata(let metadata):
            self.upload_texture_metadata(metadata)
        case .uploadSceneD3D11(let draw_segments, let clip_segments):
            self.level_impl.upload_scene(&self.core, draw_segments, clip_segments)
        case .pushRenderTarget(let render_target_id):
            self.push_render_target(render_target_id)
        case .popRenderTarget:
            self.pop_render_target()
        case .prepareClipTilesD3D11(let batch):
            self.level_impl.prepare_tiles(&self.core, batch)
        case .drawTilesD3D11(let batch):
            self.level_impl.prepare_and_draw_tiles(&self.core, batch)
        case .finish(let cpu_build_time):
            self.core.stats.cpu_build_time = cpu_build_time
        }
    }

    mutating func start_rendering(
        _ bounding_quad: [SIMD4<Float32>],
        _ path_count: Int,
        _ needs_readable_framebuffer: Bool
    ) {
        switch self.core.options.dest {
        case .other(_):
            self.core
                .renderer_flags
                .subtract(.INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED)
        case .default:
            self.core
                .renderer_flags
                .formUnion(.INTERMEDIATE_DEST_FRAMEBUFFER_NEEDED)
        }

        if self.core.renderer_flags.contains(.USE_DEPTH) {
            self.draw_stencil(bounding_quad)
        }

        self.core.stats.path_count = path_count
        self.core.render_targets.removeAll()
    }

    mutating func draw_stencil(_ quad_positions: [SIMD4<Float32>]) {
        self.core.device.allocate_buffer(
            &self.frame.stencil_vertex_array.vertex_buffer,
            .memory(quad_positions),
            .vertex
        )

        // Create indices for a triangle fan. (This is OK because the clipped quad should always be
        // convex.)
        var indices: [UInt32] = []
        for index in 1..<(UInt32(quad_positions.count) - 1) {
            indices.append(contentsOf: [0, index, index + 1])
        }

        self.core.device.allocate_buffer(
            &self.frame.stencil_vertex_array.index_buffer,
            .memory(indices),
            .index
        )

        var state = RenderState(
            target: self.core.draw_render_target(),
            program: self.stencil_program.program,
            vertex_array: self.frame.stencil_vertex_array.vertex_array,
            primitive: .triangles,
            uniforms: [],
            textures: [],
            images: [],
            storage_buffers: [],
            viewport: self.core.draw_viewport(),
            options: .init(
                // FIXME(pcwalton): Should we really write to the depth buffer?
                blend: nil,
                depth: .init(func: .less, write: true),
                stencil: .init(
                    func: .always,
                    reference: 1,
                    mask: 1,
                    write: true
                ),
                clear_ops: .init(color: nil, depth: nil, stencil: 0),
                color_mask: false
            )
        )

        self.core.device.draw_elements(indices.count, &state)
        stencil_program.program = state.program
        self.core.stats.drawcall_count += 1
    }

    mutating func allocate_pattern_texture_page(
        _ page_id: UInt32,
        _ descriptor: RenderCommand.TexturePageDescriptor
    ) {
        let page_index = Int(page_id)
        // Fill in IDs up to the requested page ID.
        while self.core.pattern_texture_pages.count < page_index + 1 {
            self.core.pattern_texture_pages.append(nil)
        }

        // Clear out any existing texture.
        if let old_texture_page = self.core.pattern_texture_pages[page_index] {
            self.core.allocator.free_framebuffer(old_texture_page.framebuffer_id)
        }

        // Allocate texture.
        let texture_size = descriptor.size
        let framebuffer_id = self.core.allocator.allocate_framebuffer(
            self.core.device,
            texture_size,
            .rgba8,
            "PatternPage"
        )
        self.core.pattern_texture_pages[page_index] = .init(
            framebuffer_id: framebuffer_id,
            must_preserve_contents: false
        )
    }

    mutating func upload_texel_data(_ texels: [Color<UInt8>], _ location: SceneBuilder.TextureLocation) {
        var texture_page = self.core.pattern_texture_pages[Int(location.page)]!
        let framebuffer_id = texture_page.framebuffer_id
        var buffer = self.core.allocator.framebuffers_in_use[framebuffer_id]!

        let texels = Color<UInt8>.toU8Array(texels)
        self.core.device.upload_to_texture(&buffer.framebuffer.value, location.rect, .u8(texels))
        texture_page.must_preserve_contents = true

        self.core.pattern_texture_pages[Int(location.page)] = texture_page
        self.core.allocator.framebuffers_in_use[framebuffer_id] = buffer
    }

    mutating func declare_render_target(
        _ render_target_id: Scene.RenderTargetId,
        _ location: SceneBuilder.TextureLocation
    ) {
        while self.core.render_targets.count < render_target_id.render_target + 1 {
            self.core.render_targets.append(.init(location: .init(page: .max, rect: .zero)))
        }

        self.core.render_targets[Int(render_target_id.render_target)].location = location
    }

    func alignup_i32(_ a: Int32, _ b: Int32) -> Int32 {
        (a + b - 1) / b
    }

    func compute_filter_params(
        _ filter: Filter,
        _ blend_mode: Scene.BlendMode,
        _ color_0_combine_mode: RenderCommand.ColorCombineMode
    ) -> FilterParams {
        var ctrl: Int32 = 0
        ctrl |= blend_mode.to_composite_ctrl() << Self.COMBINER_CTRL_COMPOSITE_SHIFT
        ctrl |= color_0_combine_mode.to_composite_ctrl() << Self.COMBINER_CTRL_COLOR_COMBINE_SHIFT

        switch filter {
        case .radialGradient(let line, let radii, let uv_origin):
            return .init(
                p0: .init(line.from.x, line.from.y, line.vector.x, line.vector.y),
                p1: .init(radii.x, radii.y, uv_origin.x, uv_origin.y),
                p2: .zero,
                p3: .zero,
                p4: .zero,
                ctrl: ctrl
                    | (Self.COMBINER_CTRL_FILTER_RADIAL_GRADIENT << Self.COMBINER_CTRL_COLOR_FILTER_SHIFT)
            )

        case .patternFilter(.blur(let direction, let sigma)):
            let sigma_inv = 1.0 / sigma
            let gauss_coeff_x = Self.SQRT_2_PI_INV * sigma_inv
            let gauss_coeff_y = exp(-0.5 * sigma_inv * sigma_inv)
            let gauss_coeff_z = gauss_coeff_y * gauss_coeff_y

            let src_offset =
                switch direction {
                case .x: SIMD2<Float32>(1.0, 0.0)
                case .y: SIMD2<Float32>(0.0, 1.0)
                }

            let support = ceil(1.5 * sigma) * 2.0

            return .init(
                p0: .init(src_offset.x, src_offset.y, support, 0.0),
                p1: .init(x: gauss_coeff_x, y: gauss_coeff_y, z: gauss_coeff_z, w: 0.0),
                p2: .zero,
                p3: .zero,
                p4: .zero,
                ctrl: ctrl | (Self.COMBINER_CTRL_FILTER_BLUR << Self.COMBINER_CTRL_COLOR_FILTER_SHIFT)
            )

        case .patternFilter(.colorMatrix(let matrix)):
            return .init(
                p0: matrix.f1,
                p1: matrix.f2,
                p2: matrix.f3,
                p3: matrix.f4,
                p4: matrix.f5,
                ctrl: ctrl
                    | (Self.COMBINER_CTRL_FILTER_COLOR_MATRIX << Self.COMBINER_CTRL_COLOR_FILTER_SHIFT)
            )

        case .none:
            return FilterParams(
                p0: .zero,
                p1: .zero,
                p2: .zero,
                p3: .zero,
                p4: .zero,
                ctrl: ctrl
            )
        }
    }

    mutating func upload_texture_metadata(_ metadata: [RenderCommand.TextureMetadataEntry]) {
        let padded_texel_size =
            alignup_i32(
                Int32(metadata.count),
                Renderer.TEXTURE_METADATA_ENTRIES_PER_ROW
            ) * Renderer.TEXTURE_METADATA_TEXTURE_WIDTH * 4

        var texels: [Float16] = []
        texels.reserveCapacity(Int(padded_texel_size))

        for entry in metadata {
            let base_color = entry.base_color.f32
            let filter_params = compute_filter_params(
                entry.filter,
                entry.blend_mode,
                entry.color_0_combine_mode
            )
            texels.append(contentsOf: [
                // 0
                Float16(entry.color_0_transform.m11),
                Float16(entry.color_0_transform.m21),
                Float16(entry.color_0_transform.m12),
                Float16(entry.color_0_transform.m22),
                // 1
                Float16(entry.color_0_transform.m13),
                Float16(entry.color_0_transform.m23),
                0.0,
                0.0,
                // 2
                Float16(base_color.r),
                Float16(base_color.g),
                Float16(base_color.b),
                Float16(base_color.a),
            ])

            texels.append(contentsOf: [
                // 3
                Float16(filter_params.p0.x),
                Float16(filter_params.p0.y),
                Float16(filter_params.p0.z),
                Float16(filter_params.p0.w),
                // 4
                Float16(filter_params.p1.x),
                Float16(filter_params.p1.y),
                Float16(filter_params.p1.z),
                Float16(filter_params.p1.w),
            ])

            texels.append(contentsOf: [
                // 5
                Float16(filter_params.p2.x),
                Float16(filter_params.p2.y),
                Float16(filter_params.p2.z),
                Float16(filter_params.p2.w),
                // 6
                Float16(filter_params.p3.x),
                Float16(filter_params.p3.y),
                Float16(filter_params.p3.z),
                Float16(filter_params.p3.w),
                // 7
                Float16(filter_params.p4.x),
                Float16(filter_params.p4.y),
                Float16(filter_params.p4.z),
                Float16(filter_params.p4.w),
                // 8
                Float16(Float32(filter_params.ctrl)),
                0.0,
                0.0,
                0.0,
                // 9
                0.0,
                0.0,
                0.0,
                0.0,
            ])
        }

        while texels.count < padded_texel_size {
            texels.append(0.0)
        }

        let texture_id = core.texture_metadata_texture_id
        let width = Self.TEXTURE_METADATA_TEXTURE_WIDTH
        let height = Int32(texels.count) / (4 * Self.TEXTURE_METADATA_TEXTURE_WIDTH)
        let rect = PFRect<Int32>(origin: .zero, size: .init(x: width, y: height))

        var value = self.core.allocator.textures_in_use[texture_id]!
        core.device.upload_to_texture(&value.texture, rect, .f16(texels))
        self.core.allocator.textures_in_use[texture_id] = value
    }

    mutating func push_render_target(_ render_target_id: Scene.RenderTargetId) {
        core.render_target_stack.append(render_target_id)
    }

    mutating func pop_render_target() {
        guard core.render_target_stack.popLast() != nil else {
            fatalError("Render target stack underflow!")
        }
    }
}

extension RendererD3D11 {
    static let FILL_INDIRECT_DRAW_PARAMS_INSTANCE_COUNT_INDEX: Int = 1
    static let FILL_INDIRECT_DRAW_PARAMS_ALPHA_TILE_COUNT_INDEX: Int = 4
    static let FILL_INDIRECT_DRAW_PARAMS_SIZE: Int = 8

    static let BIN_INDIRECT_DRAW_PARAMS_MICROLINE_COUNT_INDEX: Int = 3

    static let BOUND_WORKGROUP_SIZE: UInt32 = 64
    static let DICE_WORKGROUP_SIZE: UInt32 = 64
    static let BIN_WORKGROUP_SIZE: UInt32 = 64
    static let PROPAGATE_WORKGROUP_SIZE: UInt32 = 64
    static let SORT_WORKGROUP_SIZE: UInt32 = 64

    static let LOAD_ACTION_CLEAR: Int32 = 0
    static let LOAD_ACTION_LOAD: Int32 = 1

    struct ClipBufferIDs {
        var metadata: UInt64?
        var tiles: UInt64
    }

    struct PropagateMetadataBufferIDsD3D11 {
        var propagate_metadata: UInt64
        var backdrops: UInt64
    }

    struct MicrolinesBufferIDsD3D11 {
        var buffer_id: UInt64
        var count: UInt32
    }

    struct FillBufferInfoD3D11 {
        var fill_vertex_buffer_id: UInt64
    }

    struct LineSegmentU16 {
        let from_x: UInt16
        let from_y: UInt16
        let to_x: UInt16
        let to_y: UInt16
    }

    struct Fill {
        let line_segment: LineSegmentU16
        // The meaning of this field depends on whether fills are being done with the GPU rasterizer or
        // GPU compute. If raster, this field names the index of the alpha tile that this fill belongs
        // to. If compute, this field names the index of the next fill in the singly-linked list of
        // fills belonging to this alpha tile.
        let link: UInt32
    }

    struct PropagateTilesInfoD3D11 {
        var alpha_tile_range: Range<UInt32>
    }

    struct FirstTileD3D11 {
        var first_tile: Int32

        static func `default`() -> FirstTileD3D11 {
            .init(first_tile: -1)
        }
    }

    struct TileD3D11 {
        var next_tile_id: Int32
        var first_fill_id: Int32
        var alpha_tile_id_lo: Int16
        var alpha_tile_id_hi: Int8
        var backdrop_delta: Int8
        var color: UInt16
        var ctrl: UInt8
        var backdrop: Int8
    }

    struct AlphaTileD3D11 {
        var alpha_tile_index: UInt32
        var clip_tile_index: UInt32
    }

    mutating func upload_scene(
        _ core: inout RendererCore,
        _ draw_segments: RenderCommand.SegmentsD3D11,
        _ clip_segments: RenderCommand.SegmentsD3D11
    ) {
        scene_buffers.upload(&core.allocator, core.device, draw_segments, clip_segments)
    }

    mutating func prepare_tiles(
        _ core: inout RendererCore,
        _ batch: RenderCommand.TileBatchDataD3D11
    ) {
        core.stats.total_tile_count += Int(batch.tile_count)

        // Upload tiles to GPU or allocate them as appropriate.
        let tiles_d3d11_buffer_id = allocate_tiles(&core, batch.tile_count)

        // Fetch and/or allocate clip storage as needed.
        let clip_buffer_ids: ClipBufferIDs?

        switch batch.clipped_path_info {
        case .some(let clipped_path_info):
            let clip_batch_id = clipped_path_info.clip_batch_id
            let clip_tile_batch_info = tile_batch_info[Int(clip_batch_id)]!
            let metadata = clip_tile_batch_info.propagate_metadata_buffer_id
            let tiles = clip_tile_batch_info.tiles_d3d11_buffer_id
            clip_buffer_ids = .init(metadata: metadata, tiles: tiles)
        case .none:
            clip_buffer_ids = nil
        }

        // Allocate a Z-buffer.
        let z_buffer_id = allocate_z_buffer(&core)

        // Propagate backdrops, bin fills, render fills, and/or perform clipping on GPU if
        // necessary.
        // Allocate space for tile lists.
        let first_tile_map_buffer_id = allocate_first_tile_map(&core)

        let propagate_metadata_buffer_ids = upload_propagate_metadata(
            &core,
            batch.prepare_info.propagate_metadata,
            batch.prepare_info.backdrops
        )

        // Dice (flatten) segments into microlines. We might have to do this twice if our
        // first attempt runs out of space in the storage buffer.
        var microlines_storage: MicrolinesBufferIDsD3D11? = nil
        for _ in 0..<2 {
            microlines_storage = dice_segments(
                &core,
                batch.prepare_info.dice_metadata,
                batch.segment_count,
                batch.path_source,
                batch.prepare_info.transform
            )
            if microlines_storage != nil {
                break
            }
        }

        guard let microlines_storage = microlines_storage else {
            fatalError("Ran out of space for microlines when dicing!")
        }

        // Initialize tiles, and bin segments. We might have to do this twice if our first
        // attempt runs out of space in the fill buffer.
        var fill_buffer_info: FillBufferInfoD3D11? = nil
        for _ in 0..<2 {
            bound(&core, tiles_d3d11_buffer_id, batch.tile_count, batch.prepare_info.tile_path_info)

            upload_initial_backdrops(
                &core,
                propagate_metadata_buffer_ids.backdrops,
                batch.prepare_info.backdrops
            )

            fill_buffer_info = bin_segments(
                &core,
                microlines_storage,
                propagate_metadata_buffer_ids,
                tiles_d3d11_buffer_id,
                z_buffer_id
            )
            if fill_buffer_info != nil {
                break
            }
        }
        guard let fill_buffer_info = fill_buffer_info else {
            fatalError("Ran out of space for fills when binning!")
        }

        core.allocator.free_general_buffer(microlines_storage.buffer_id)

        // TODO(pcwalton): If we run out of space for alpha tile indices, propagate
        // multiple times.

        let alpha_tiles_buffer_id = allocate_alpha_tile_info(&core, batch.tile_count)

        let propagate_tiles_info = propagate_tiles(
            &core,
            UInt32(batch.prepare_info.backdrops.count),
            tiles_d3d11_buffer_id,
            z_buffer_id,
            first_tile_map_buffer_id,
            alpha_tiles_buffer_id,
            propagate_metadata_buffer_ids,
            clip_buffer_ids
        )

        core.allocator.free_general_buffer(propagate_metadata_buffer_ids.backdrops)

        // FIXME(pcwalton): Don't unconditionally pass true for copying here.
        core.reallocate_alpha_tile_pages_if_necessary(true)
        draw_fills(
            &core,
            fill_buffer_info,
            tiles_d3d11_buffer_id,
            alpha_tiles_buffer_id,
            propagate_tiles_info
        )

        core.allocator.free_general_buffer(fill_buffer_info.fill_vertex_buffer_id)
        core.allocator.free_general_buffer(alpha_tiles_buffer_id)

        // FIXME(pcwalton): This seems like the wrong place to do this...
        sort_tiles(&core, tiles_d3d11_buffer_id, first_tile_map_buffer_id, z_buffer_id)

        // Record tile batch info.
        tile_batch_info[Int(batch.batch_id)] = .init(
            tile_count: batch.tile_count,
            z_buffer_id: z_buffer_id,
            tiles_d3d11_buffer_id: tiles_d3d11_buffer_id,
            propagate_metadata_buffer_id: propagate_metadata_buffer_ids.propagate_metadata,
            first_tile_map_buffer_id: first_tile_map_buffer_id
        )
    }

    mutating func allocate_tiles(_ core: inout RendererCore, _ tile_count: UInt32) -> UInt64 {
        return core.allocator.allocate_general_buffer(
            core.device,
            Int(tile_count) * MemoryLayout<TileD3D11>.stride,
            "TileD3D11"
        )
    }

    mutating func allocate_z_buffer(_ core: inout RendererCore) -> UInt64 {
        // This includes the fill indirect draw params because some drivers limit the number of
        // SSBOs to 8 (#373).
        let tileSize = core.tile_size()
        let size = Int(tileSize.x * tileSize.y) + Int(Self.FILL_INDIRECT_DRAW_PARAMS_SIZE)
        return core.allocator.allocate_general_buffer(
            core.device,
            size * MemoryLayout<Int32>.stride,
            "ZBufferD3D11"
        )
    }

    mutating func allocate_first_tile_map(_ core: inout RendererCore) -> UInt64 {
        let tileSize = core.tile_size()
        return core.allocator.allocate_general_buffer(
            core.device,
            Int(tileSize.x * tileSize.y) * MemoryLayout<FirstTileD3D11>.stride,
            "FirstTileD3D11"
        )
    }

    mutating func upload_propagate_metadata(
        _ core: inout RendererCore,
        _ propagate_metadata: [RenderCommand.PropagateMetadataD3D11],
        _ backdrops: [RenderCommand.BackdropInfoD3D11]
    ) -> PropagateMetadataBufferIDsD3D11 {
        let propagate_metadata_storage_id = core.allocator.allocate_general_buffer(
            core.device,
            propagate_metadata.count * MemoryLayout<RenderCommand.PropagateMetadataD3D11>.stride,
            "PropagateMetadataD3D11"
        )

        var propagateBuffer = core.allocator.general_buffers_in_use[propagate_metadata_storage_id]!
        core.device.upload_to_buffer(
            &propagateBuffer.buffer,
            0,
            propagate_metadata,
            .storage
        )
        core.allocator.general_buffers_in_use[propagate_metadata_storage_id] = propagateBuffer

        let backdrops_storage_id = core.allocator.allocate_general_buffer(
            core.device,
            backdrops.count * MemoryLayout<RenderCommand.BackdropInfoD3D11>.stride,
            "BackdropInfoD3D11"
        )

        return PropagateMetadataBufferIDsD3D11(
            propagate_metadata: propagate_metadata_storage_id,
            backdrops: backdrops_storage_id
        )
    }

    mutating func dice_segments(
        _ core: inout RendererCore,
        _ dice_metadata: [RenderCommand.DiceMetadataD3D11],
        _ batch_segment_count: UInt32,
        _ path_source: RenderCommand.PathSource,
        _ transform: Transform
    ) -> MicrolinesBufferIDsD3D11? {
        let dice_program = programs.dice_program

        let microlines_buffer_id = core.allocator.allocate_general_buffer(
            core.device,
            Int(allocated_microline_count) * MemoryLayout<RenderCommand.MicrolineD3D11>.stride,
            "MicrolineD3D11"
        )
        let dice_metadata_buffer_id = core.allocator.allocate_general_buffer(
            core.device,
            dice_metadata.count * MemoryLayout<RenderCommand.DiceMetadataD3D11>.stride,
            "DiceMetadataD3D11"
        )
        let dice_indirect_draw_params_buffer_id = core.allocator.allocate_general_buffer(
            core.device,
            8 * MemoryLayout<UInt32>.stride,
            "DiceIndirectDrawParamsD3D11"
        )

        let microlines_buffer = core.allocator.get_general_buffer(microlines_buffer_id)

        let scene_buffers = self.scene_buffers
        let scene_source_buffers: SceneSourceBuffers
        switch path_source {
        case .draw:
            scene_source_buffers = scene_buffers.draw
        case .clip:
            scene_source_buffers = scene_buffers.clip
        }

        let points_buffer_id = scene_source_buffers.points_buffer!
        let point_indices_buffer_id = scene_source_buffers.point_indices_buffer!
        let point_indices_count = scene_source_buffers.point_indices_count

        let points_buffer = core.allocator.get_general_buffer(points_buffer_id)
        let point_indices_buffer = core.allocator.get_general_buffer(point_indices_buffer_id)

        var indirectBuffer = core.allocator.general_buffers_in_use[dice_indirect_draw_params_buffer_id]!
        core.device.upload_to_buffer(
            &indirectBuffer.buffer,
            0,
            [0, 0, 0, 0, point_indices_count, 0, 0, 0],
            .storage
        )
        core.allocator.general_buffers_in_use[dice_indirect_draw_params_buffer_id] = indirectBuffer

        var metadataBuffer = core.allocator.general_buffers_in_use[dice_metadata_buffer_id]!
        core.device.upload_to_buffer(
            &metadataBuffer.buffer,
            0,
            dice_metadata,
            .storage
        )
        core.allocator.general_buffers_in_use[dice_metadata_buffer_id] = metadataBuffer

        let workgroup_count =
            (batch_segment_count + Self.DICE_WORKGROUP_SIZE - 1) / Self.DICE_WORKGROUP_SIZE
        let compute_dimensions = ProgramsD3D11.ComputeDimensions(x: workgroup_count, y: 1, z: 1)

        var state = ComputeState(
            program: dice_program.program,
            uniforms: [
                (dice_program.transform_uniform, .mat2(transform.matrix)),
                (dice_program.translation_uniform, .vec2(transform.vector)),
                (dice_program.path_count_uniform, .int(Int32(dice_metadata.count))),
                (dice_program.last_batch_segment_index_uniform, .int(Int32(batch_segment_count))),
                (dice_program.max_microline_count_uniform, .int(Int32(allocated_microline_count))),
            ],
            textures: [],
            images: [],
            storage_buffers: [
                (dice_program.compute_indirect_params_storage_buffer, indirectBuffer.buffer),
                (dice_program.points_storage_buffer, points_buffer),
                (dice_program.input_indices_storage_buffer, point_indices_buffer),
                (dice_program.microlines_storage_buffer, microlines_buffer),
                (dice_program.dice_metadata_storage_buffer, metadataBuffer.buffer),
            ]
        )

        core.device.dispatch_compute(compute_dimensions, &state)
        programs.dice_program.program = state.program
        programs.dice_program.transform_uniform = state.uniforms[0].0
        programs.dice_program.translation_uniform = state.uniforms[1].0
        programs.dice_program.path_count_uniform = state.uniforms[2].0
        programs.dice_program.last_batch_segment_index_uniform = state.uniforms[3].0
        programs.dice_program.max_microline_count_uniform = state.uniforms[4].0
        programs.dice_program.compute_indirect_params_storage_buffer = state.storage_buffers[0].0
        programs.dice_program.points_storage_buffer = state.storage_buffers[1].0
        programs.dice_program.input_indices_storage_buffer = state.storage_buffers[2].0
        programs.dice_program.microlines_storage_buffer = state.storage_buffers[3].0
        programs.dice_program.dice_metadata_storage_buffer = state.storage_buffers[4].0
        core.stats.drawcall_count += 1

        let indirect_compute_params_receiver = core.device.read_buffer(
            indirectBuffer.buffer,
            .storage,
            0..<32
        )

        let indirect_compute_params = core.device.recv_buffer(indirect_compute_params_receiver)
        let indirect_compute_params_array = indirect_compute_params.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: UInt32.self))
        }

        core.allocator.free_general_buffer(dice_metadata_buffer_id)
        core.allocator.free_general_buffer(dice_indirect_draw_params_buffer_id)
        let microline_count = indirect_compute_params_array[
            Self.BIN_INDIRECT_DRAW_PARAMS_MICROLINE_COUNT_INDEX
        ]
        if microline_count > allocated_microline_count {
            allocated_microline_count = microline_count.nextPowerOfTwo
            return nil
        }

        return .init(buffer_id: microlines_buffer_id, count: microline_count)
    }

    mutating func bound(
        _ core: inout RendererCore,
        _ tiles_d3d11_buffer_id: UInt64,
        _ tile_count: UInt32,
        _ tile_path_info: [RenderCommand.TilePathInfoD3D11]
    ) {
        let bound_program = programs.bound_program

        let path_info_buffer_id = core.allocator.allocate_general_buffer(
            core.device,
            tile_path_info.count * MemoryLayout<RenderCommand.TilePathInfoD3D11>.stride,
            "TilePathInfoD3D11"
        )

        var tilePathBuffer = core.allocator.general_buffers_in_use[path_info_buffer_id]!
        core.device.upload_to_buffer(
            &tilePathBuffer.buffer,
            0,
            tile_path_info,
            .storage
        )
        core.allocator.general_buffers_in_use[path_info_buffer_id] = tilePathBuffer

        let tiles_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id)

        let compute_dimensions = ProgramsD3D11.ComputeDimensions(
            x: (tile_count + Self.BOUND_WORKGROUP_SIZE - 1) / Self.BOUND_WORKGROUP_SIZE,
            y: 1,
            z: 1
        )

        var state = ComputeState(
            program: bound_program.program,
            uniforms: [
                (bound_program.path_count_uniform, .int(Int32(tile_path_info.count))),
                (bound_program.tile_count_uniform, .int(Int32(tile_count))),
            ],
            textures: [],
            images: [],
            storage_buffers: [
                (bound_program.tile_path_info_storage_buffer, tilePathBuffer.buffer),
                (bound_program.tiles_storage_buffer, tiles_buffer),
            ]
        )

        core.device.dispatch_compute(compute_dimensions, &state)
        programs.bound_program.program = state.program
        programs.bound_program.path_count_uniform = state.uniforms[0].0
        programs.bound_program.tile_count_uniform = state.uniforms[1].0
        programs.bound_program.tile_path_info_storage_buffer = state.storage_buffers[0].0
        programs.bound_program.tiles_storage_buffer = state.storage_buffers[1].0
        core.stats.drawcall_count += 1

        core.allocator.free_general_buffer(path_info_buffer_id)
    }

    func upload_initial_backdrops(
        _ core: inout RendererCore,
        _ backdrops_buffer_id: UInt64,
        _ backdrops: [RenderCommand.BackdropInfoD3D11]
    ) {
        var backdropBuffer = core.allocator.general_buffers_in_use[backdrops_buffer_id]!
        core.device.upload_to_buffer(&backdropBuffer.buffer, 0, backdrops, .storage)
        core.allocator.general_buffers_in_use[backdrops_buffer_id] = backdropBuffer
    }

    mutating func bin_segments(
        _ core: inout RendererCore,
        _ microlines_storage: MicrolinesBufferIDsD3D11,
        _ propagate_metadata_buffer_ids: PropagateMetadataBufferIDsD3D11,
        _ tiles_d3d11_buffer_id: UInt64,
        _ z_buffer_id: UInt64
    ) -> FillBufferInfoD3D11? {
        let bin_program = programs.bin_program

        let fill_vertex_buffer_id = core.allocator.allocate_general_buffer(
            core.device,
            Int(allocated_fill_count) * MemoryLayout<Fill>.stride,
            "Fill"
        )

        let fill_vertex_buffer = core.allocator.get_general_buffer(fill_vertex_buffer_id)
        let microlines_buffer = core.allocator.get_general_buffer(microlines_storage.buffer_id)
        let tiles_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id)
        let propagate_metadata_buffer = core.allocator.get_general_buffer(
            propagate_metadata_buffer_ids.propagate_metadata
        )
        let backdrops_buffer = core.allocator.get_general_buffer(
            propagate_metadata_buffer_ids.backdrops
        )

        // Upload fill indirect draw params to header of the Z-buffer.
        //
        // This is in the Z-buffer, not its own buffer, to work around the 8 SSBO limitation on
        // some drivers (#373).
        let indirect_draw_params: [UInt32] = [6, 0, 0, 0, 0, microlines_storage.count, 0, 0]

        var zBuffer = core.allocator.general_buffers_in_use[z_buffer_id]!
        core.device.upload_to_buffer(
            &zBuffer.buffer,
            0,
            indirect_draw_params,
            .storage
        )
        core.allocator.general_buffers_in_use[z_buffer_id] = zBuffer

        let compute_dimensions = ProgramsD3D11.ComputeDimensions(
            x: (microlines_storage.count + Self.BIN_WORKGROUP_SIZE - 1) / Self.BIN_WORKGROUP_SIZE,
            y: 1,
            z: 1
        )

        var state = ComputeState(
            program: bin_program.program,
            uniforms: [
                (bin_program.microline_count_uniform, .int(Int32(microlines_storage.count))),
                (bin_program.max_fill_count_uniform, .int(Int32(allocated_fill_count))),
            ],
            textures: [],
            images: [],
            storage_buffers: [
                (bin_program.microlines_storage_buffer, microlines_buffer),
                (bin_program.metadata_storage_buffer, propagate_metadata_buffer),
                (bin_program.indirect_draw_params_storage_buffer, zBuffer.buffer),
                (bin_program.fills_storage_buffer, fill_vertex_buffer),
                (bin_program.tiles_storage_buffer, tiles_buffer),
                (bin_program.backdrops_storage_buffer, backdrops_buffer),
            ]
        )

        core.device.dispatch_compute(compute_dimensions, &state)
        programs.bin_program.program = state.program
        programs.bin_program.microline_count_uniform = state.uniforms[0].0
        programs.bin_program.max_fill_count_uniform = state.uniforms[1].0
        programs.bin_program.microlines_storage_buffer = state.storage_buffers[0].0
        programs.bin_program.metadata_storage_buffer = state.storage_buffers[1].0
        programs.bin_program.indirect_draw_params_storage_buffer = state.storage_buffers[2].0
        programs.bin_program.fills_storage_buffer = state.storage_buffers[3].0
        programs.bin_program.tiles_storage_buffer = state.storage_buffers[4].0
        programs.bin_program.backdrops_storage_buffer = state.storage_buffers[5].0
        core.stats.drawcall_count += 1

        let indirect_draw_params_receiver = core.device.read_buffer(
            zBuffer.buffer,
            .storage,
            0..<32
        )
        let indirect_draw_params_data = core.device.recv_buffer(indirect_draw_params_receiver)
        let indirect_draw_params_from_data = indirect_draw_params_data.withUnsafeBytes { bytes in
            bytes.bindMemory(to: UInt32.self)
        }

        let needed_fill_count = indirect_draw_params_from_data[
            Self.FILL_INDIRECT_DRAW_PARAMS_INSTANCE_COUNT_INDEX
        ]
        if needed_fill_count > allocated_fill_count {
            allocated_fill_count = needed_fill_count.nextPowerOfTwo
            return nil
        }

        core.stats.fill_count += Int(needed_fill_count)

        return .init(fill_vertex_buffer_id: fill_vertex_buffer_id)
    }

    mutating func sort_tiles(
        _ core: inout RendererCore,
        _ tiles_d3d11_buffer_id: UInt64,
        _ first_tile_map_buffer_id: UInt64,
        _ z_buffer_id: UInt64
    ) {
        let sort_program = programs.sort_program

        let tiles_d3d11_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id)
        let first_tile_map_buffer = core.allocator.get_general_buffer(first_tile_map_buffer_id)
        let z_buffer = core.allocator.get_general_buffer(z_buffer_id)

        let frameBufferTileSize = core.framebuffer_tile_size()
        let tile_count = frameBufferTileSize.x * frameBufferTileSize.y

        let dimensions = ProgramsD3D11.ComputeDimensions(
            x: (UInt32(tile_count) + Self.SORT_WORKGROUP_SIZE - 1) / Self.SORT_WORKGROUP_SIZE,
            y: 1,
            z: 1
        )

        var state = ComputeState(
            program: sort_program.program,
            uniforms: [(sort_program.tile_count_uniform, .int(tile_count))],
            textures: [],
            images: [],
            storage_buffers: [
                (sort_program.tiles_storage_buffer, tiles_d3d11_buffer),
                (sort_program.first_tile_map_storage_buffer, first_tile_map_buffer),
                (sort_program.z_buffer_storage_buffer, z_buffer),
            ]
        )

        core.device.dispatch_compute(dimensions, &state)
        programs.sort_program.program = state.program
        programs.sort_program.tile_count_uniform = state.uniforms[0].0
        programs.sort_program.tiles_storage_buffer = state.storage_buffers[0].0
        programs.sort_program.first_tile_map_storage_buffer = state.storage_buffers[1].0
        programs.sort_program.z_buffer_storage_buffer = state.storage_buffers[2].0
        core.stats.drawcall_count += 1
    }

    func allocate_alpha_tile_info(_ core: inout RendererCore, _ index_count: UInt32) -> UInt64 {
        return core.allocator.allocate_general_buffer(
            core.device,
            Int(index_count) * MemoryLayout<AlphaTileD3D11>.stride,
            "AlphaTileD3D11"
        )
    }

    mutating func propagate_tiles(
        _ core: inout RendererCore,
        _ column_count: UInt32,
        _ tiles_d3d11_buffer_id: UInt64,
        _ z_buffer_id: UInt64,
        _ first_tile_map_buffer_id: UInt64,
        _ alpha_tiles_buffer_id: UInt64,
        _ propagate_metadata_buffer_ids: PropagateMetadataBufferIDsD3D11,
        _ clip_buffer_ids: ClipBufferIDs?
    ) -> PropagateTilesInfoD3D11 {
        let propagate_program = programs.propagate_program

        let tiles_d3d11_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id)
        let propagate_metadata_storage_buffer =
            core.allocator.get_general_buffer(propagate_metadata_buffer_ids.propagate_metadata)
        let backdrops_storage_buffer =
            core.allocator.get_general_buffer(propagate_metadata_buffer_ids.backdrops)

        // TODO(pcwalton): Zero out the Z-buffer on GPU?
        let z_buffer_size = core.tile_size()
        let tile_area = Int(z_buffer_size.x * z_buffer_size.y)

        var zBuffer = core.allocator.general_buffers_in_use[z_buffer_id]!
        core.device.upload_to_buffer(
            &zBuffer.buffer,
            0,
            Array(repeating: Int32(0), count: tile_area),
            .storage
        )
        core.allocator.general_buffers_in_use[z_buffer_id] = zBuffer

        // TODO(pcwalton): Initialize the first tiles buffer on GPU?
        var firstTileMapBuffer = core.allocator.general_buffers_in_use[first_tile_map_buffer_id]!
        core.device.upload_to_buffer(
            &firstTileMapBuffer.buffer,
            0,
            Array(repeating: FirstTileD3D11.default(), count: tile_area),
            .storage
        )
        core.allocator.general_buffers_in_use[first_tile_map_buffer_id] = firstTileMapBuffer

        let alpha_tiles_storage_buffer = core.allocator.get_general_buffer(alpha_tiles_buffer_id)

        var storage_buffers = [
            (propagate_program.draw_metadata_storage_buffer, propagate_metadata_storage_buffer),
            (propagate_program.backdrops_storage_buffer, backdrops_storage_buffer),
            (propagate_program.draw_tiles_storage_buffer, tiles_d3d11_buffer),
            (propagate_program.z_buffer_storage_buffer, zBuffer.buffer),
            (propagate_program.first_tile_map_storage_buffer, firstTileMapBuffer.buffer),
            (propagate_program.alpha_tiles_storage_buffer, alpha_tiles_storage_buffer),
        ]

        if let clip_buffer_ids = clip_buffer_ids {
            let clip_metadata_buffer_id = clip_buffer_ids.metadata!
            let clip_metadata_buffer = core.allocator
                .get_general_buffer(clip_metadata_buffer_id)
            let clip_tile_buffer = core.allocator.get_general_buffer(clip_buffer_ids.tiles)
            storage_buffers.append(
                (
                    propagate_program.clip_metadata_storage_buffer,
                    clip_metadata_buffer
                )
            )
            storage_buffers.append(
                (
                    propagate_program.clip_tiles_storage_buffer,
                    clip_tile_buffer
                )
            )
        } else {
            // Just attach any old buffers to these, to satisfy Metal.
            storage_buffers.append(
                (
                    propagate_program.clip_metadata_storage_buffer,
                    propagate_metadata_storage_buffer
                )
            )
            storage_buffers.append(
                (
                    propagate_program.clip_tiles_storage_buffer,
                    tiles_d3d11_buffer
                )
            )
        }

        let dimensions = ProgramsD3D11.ComputeDimensions(
            x: (column_count + Self.PROPAGATE_WORKGROUP_SIZE - 1) / Self.PROPAGATE_WORKGROUP_SIZE,
            y: 1,
            z: 1
        )

        var state = ComputeState(
            program: propagate_program.program,
            uniforms: [
                (propagate_program.framebuffer_tile_size_uniform, .ivec2(core.framebuffer_tile_size())),
                (propagate_program.column_count_uniform, .int(Int32(column_count))),
                (propagate_program.first_alpha_tile_index_uniform, .int(Int32(core.alpha_tile_count))),
            ],
            textures: [],
            images: [],
            storage_buffers: storage_buffers
        )

        core.device.dispatch_compute(dimensions, &state)
        programs.propagate_program.program = state.program
        programs.propagate_program.framebuffer_tile_size_uniform = state.uniforms[0].0
        programs.propagate_program.column_count_uniform = state.uniforms[1].0
        programs.propagate_program.first_alpha_tile_index_uniform = state.uniforms[2].0

        programs.propagate_program.draw_metadata_storage_buffer = state.storage_buffers[0].0
        programs.propagate_program.backdrops_storage_buffer = state.storage_buffers[1].0
        programs.propagate_program.draw_tiles_storage_buffer = state.storage_buffers[2].0
        programs.propagate_program.z_buffer_storage_buffer = state.storage_buffers[3].0
        programs.propagate_program.first_tile_map_storage_buffer = state.storage_buffers[4].0
        programs.propagate_program.alpha_tiles_storage_buffer = state.storage_buffers[5].0
        programs.propagate_program.clip_metadata_storage_buffer = state.storage_buffers[6].0
        programs.propagate_program.clip_tiles_storage_buffer = state.storage_buffers[7].0

        core.stats.drawcall_count += 1

        let fill_indirect_draw_params_receiver = core.device.read_buffer(
            zBuffer.buffer,
            .storage,
            0..<32
        )
        let fill_indirect_draw_params = core.device.recv_buffer(fill_indirect_draw_params_receiver)
        let fill_indirect_draw_params_slice: [UInt32] = fill_indirect_draw_params.withUnsafeBytes {
            bytes in
            let boundPointer = bytes.bindMemory(to: UInt32.self)
            return Array(UnsafeBufferPointer(start: boundPointer.baseAddress, count: boundPointer.count))
        }

        let batch_alpha_tile_count = fill_indirect_draw_params_slice[
            Self.FILL_INDIRECT_DRAW_PARAMS_ALPHA_TILE_COUNT_INDEX
        ]

        let alpha_tile_start = core.alpha_tile_count
        core.alpha_tile_count += batch_alpha_tile_count
        core.stats.alpha_tile_count += Int(batch_alpha_tile_count)
        let alpha_tile_end = core.alpha_tile_count

        return .init(alpha_tile_range: alpha_tile_start..<alpha_tile_end)
    }

    mutating func draw_fills(
        _ core: inout RendererCore,
        _ fill_storage_info: FillBufferInfoD3D11,
        _ tiles_d3d11_buffer_id: UInt64,
        _ alpha_tiles_buffer_id: UInt64,
        _ propagate_tiles_info: PropagateTilesInfoD3D11
    ) {
        let fill_vertex_buffer_id = fill_storage_info.fill_vertex_buffer_id
        let alpha_tile_range = propagate_tiles_info.alpha_tile_range

        let fill_program = programs.fill_program
        let fill_vertex_buffer = core.allocator.get_general_buffer(fill_vertex_buffer_id)

        guard let mask_storage = core.mask_storage else {
            fatalError("Where's the mask storage?")
        }
        let mask_framebuffer_id = mask_storage.framebuffer_id
        let mask_framebuffer = core.allocator.get_framebuffer(mask_framebuffer_id)
        let image_texture = core.device.framebuffer_texture(mask_framebuffer)

        let tiles_d3d11_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id)
        let alpha_tiles_buffer = core.allocator.get_general_buffer(alpha_tiles_buffer_id)

        let area_lut_texture = core.allocator.get_texture(core.area_lut_texture_id)

        // This setup is an annoying workaround for the 64K limit of compute invocation in OpenGL.
        let alpha_tile_count = alpha_tile_range.upperBound - alpha_tile_range.lowerBound
        let dimensions = ProgramsD3D11.ComputeDimensions(
            x: min(alpha_tile_count, 1 << 15),
            y: (alpha_tile_count + (1 << 15) - 1) >> 15,
            z: 1
        )

        var state = ComputeState(
            program: fill_program.program,
            uniforms: [
                (
                    fill_program.alpha_tile_range_uniform,
                    .ivec2(
                        SIMD2<Int32>(Int32(alpha_tile_range.lowerBound), Int32(alpha_tile_range.upperBound))
                    )
                )
            ],
            textures: [(fill_program.area_lut_texture, area_lut_texture)],
            images: [(fill_program.dest_image, image_texture, .readWrite)],
            storage_buffers: [
                (fill_program.fills_storage_buffer, fill_vertex_buffer),
                (fill_program.tiles_storage_buffer, tiles_d3d11_buffer),
                (fill_program.alpha_tiles_storage_buffer, alpha_tiles_buffer),
            ]
        )

        core.device.dispatch_compute(dimensions, &state)
        programs.fill_program.program = state.program
        programs.fill_program.alpha_tile_range_uniform = state.uniforms[0].0
        programs.fill_program.area_lut_texture = state.textures[0].0
        programs.fill_program.dest_image = state.images[0].0
        programs.fill_program.fills_storage_buffer = state.storage_buffers[0].0
        programs.fill_program.tiles_storage_buffer = state.storage_buffers[1].0
        programs.fill_program.alpha_tiles_storage_buffer = state.storage_buffers[2].0
        core.stats.drawcall_count += 1

        core.framebuffer_flags.formUnion(.MASK_FRAMEBUFFER_IS_DIRTY)
    }

    mutating func prepare_and_draw_tiles(
        _ core: inout RendererCore,
        _ batch: RenderCommand.DrawTileBatchD3D11
    ) {
        let tile_batch_id = batch.tile_batch_data.batch_id
        prepare_tiles(&core, batch.tile_batch_data)
        let batch_info = tile_batch_info[Int(tile_batch_id)]!
        draw_tiles(
            &core,
            batch_info.tiles_d3d11_buffer_id,
            batch_info.first_tile_map_buffer_id,
            batch.color_texture
        )
    }

    mutating func draw_tiles(
        _ core: inout RendererCore,
        _ tiles_d3d11_buffer_id: UInt64,
        _ first_tile_map_buffer_id: UInt64,
        _ color_texture_0: RenderCommand.TileBatchTexture?
    ) {
        let tile_program = programs.tile_program

        var images: [RenderState.ImageBinding<PFDevice.ImageParameter, PFDevice.Texture>] = []
        var textures: [RenderState.TextureBinding<PFDevice.TextureParameter, PFDevice.Texture>] = []
        var uniforms: [RenderState.UniformBinding<PFDevice.Uniform>] = []

        let draw_viewport = core.draw_viewport()

        textures.append(
            (tile_program.common.gamma_lut_texture, core.allocator.get_texture(core.gamma_lut_texture_id))
        )
        textures.append(
            (
                tile_program.common.texture_metadata_texture,
                core.allocator.get_texture(core.texture_metadata_texture_id)
            )
        )

        uniforms.append(
            (
                tile_program.common.tile_size_uniform,
                .vec2(SIMD2<Float>(Float(SceneBuilder.TILE_WIDTH), Float(SceneBuilder.TILE_HEIGHT)))
            )
        )
        uniforms.append(
            (
                tile_program.common.framebuffer_size_uniform,
                .vec2(SIMD2<Float32>(Float32(draw_viewport.size.x), Float32(draw_viewport.size.y)))
            )
        )
        uniforms.append(
            (
                tile_program.common.texture_metadata_size_uniform,
                .ivec2(
                    SIMD2<Int32>(
                        Renderer.TEXTURE_METADATA_TEXTURE_WIDTH,
                        Renderer.TEXTURE_METADATA_TEXTURE_HEIGHT
                    )
                )
            )
        )

        var hasMaskStorage = false
        if let mask_storage = core.mask_storage {
            hasMaskStorage = true
            let mask_framebuffer_id = mask_storage.framebuffer_id
            let mask_framebuffer = core.allocator.get_framebuffer(mask_framebuffer_id)
            let mask_texture = core.device.framebuffer_texture(mask_framebuffer)
            uniforms.append(
                (
                    tile_program.common.mask_texture_size_0_uniform,
                    .vec2(.init(core.device.texture_size(mask_texture)))
                )
            )
            textures.append((tile_program.common.mask_texture_0, mask_texture))
        }

        if let color_texture = color_texture_0 {
            var color_texture_page = core.texture_page(color_texture.page)
            let color_texture_size = SIMD2<Float32>(core.device.texture_size(color_texture_page))
            core.device.set_texture_sampling_mode(&color_texture_page, color_texture.sampling_flags)
            textures.append((tile_program.common.color_texture_0, color_texture_page))
            uniforms.append((tile_program.common.color_texture_size_0_uniform, .vec2(color_texture_size)))
        } else {
            // Attach any old texture, just to satisfy Metal.
            textures.append(
                (
                    tile_program.common.color_texture_0,
                    core.allocator.get_texture(core.texture_metadata_texture_id)
                )
            )
            uniforms.append((tile_program.common.color_texture_size_0_uniform, .vec2(.init())))
        }

        uniforms.append(
            (tile_program.framebuffer_tile_size_uniform, .ivec2(core.framebuffer_tile_size()))
        )

        switch core.draw_render_target() {
        case .default:
            fatalError("Can't draw to the default framebuffer with compute!")
        case .framebuffer(let framebuffer):
            let dest_texture = core.device.framebuffer_texture(framebuffer)
            images.append((tile_program.dest_image, dest_texture, .readWrite))
        }

        let clear_color = core.clear_color_for_draw_operation()
        if let clear_color = clear_color {
            uniforms.append((tile_program.load_action_uniform, .int(Self.LOAD_ACTION_CLEAR)))
            uniforms.append((tile_program.clear_color_uniform, .vec4(clear_color.simd)))
        } else {
            uniforms.append((tile_program.load_action_uniform, .int(Self.LOAD_ACTION_LOAD)))
            uniforms.append((tile_program.clear_color_uniform, .vec4(.init())))
        }

        let tiles_d3d11_buffer = core.allocator.get_general_buffer(tiles_d3d11_buffer_id)
        let first_tile_map_storage_buffer = core.allocator
            .get_general_buffer(first_tile_map_buffer_id)

        let framebuffer_tile_size = core.framebuffer_tile_size()
        let compute_dimensions = ProgramsD3D11.ComputeDimensions(
            x: UInt32(framebuffer_tile_size.x),
            y: UInt32(framebuffer_tile_size.y),
            z: 1
        )

        var state = ComputeState(
            program: tile_program.common.program,
            uniforms: uniforms,
            textures: textures,
            images: images,
            storage_buffers: [
                (tile_program.tiles_storage_buffer, tiles_d3d11_buffer),
                (tile_program.first_tile_map_storage_buffer, first_tile_map_storage_buffer),
            ]
        )

        core.device.dispatch_compute(compute_dimensions, &state)
        programs.tile_program.common.program = state.program

        programs.tile_program.common.gamma_lut_texture = state.textures[0].0
        programs.tile_program.common.texture_metadata_texture = state.textures[1].0

        programs.tile_program.common.tile_size_uniform = state.uniforms[0].0
        programs.tile_program.common.framebuffer_size_uniform = state.uniforms[1].0
        programs.tile_program.common.texture_metadata_size_uniform = state.uniforms[2].0

        var nextUniform = 2
        var nextTexture = 1
        if hasMaskStorage {
            programs.tile_program.common.mask_texture_size_0_uniform = state.uniforms[nextUniform + 1].0
            programs.tile_program.common.mask_texture_0 = state.textures[nextTexture + 1].0
            nextUniform += 1
            nextTexture += 1
        }

        programs.tile_program.common.color_texture_0 = state.textures[nextTexture + 1].0
        programs.tile_program.common.color_texture_size_0_uniform = state.uniforms[nextUniform + 1].0

        programs.tile_program.framebuffer_tile_size_uniform = state.uniforms[nextUniform + 2].0

        programs.tile_program.dest_image = state.images[0].0
        programs.tile_program.load_action_uniform = state.uniforms[nextUniform + 3].0
        programs.tile_program.clear_color_uniform = state.uniforms[nextUniform + 4].0

        core.stats.drawcall_count += 1
        core.preserve_draw_framebuffer()
    }

    mutating func end_frame(_ core: inout RendererCore) {
        free_tile_batch_buffers(&core)
    }

    mutating func free_tile_batch_buffers(_ core: inout RendererCore) {
        for (_, tile_batch_info) in tile_batch_info {
            core.allocator.free_general_buffer(tile_batch_info.z_buffer_id)
            core.allocator.free_general_buffer(tile_batch_info.tiles_d3d11_buffer_id)
            core.allocator.free_general_buffer(tile_batch_info.propagate_metadata_buffer_id)
            core.allocator.free_general_buffer(tile_batch_info.first_tile_map_buffer_id)
        }
        tile_batch_info.removeAll()
    }
}

extension DispatchTimeInterval {
    var seconds: Double {
        switch self {
        case .seconds(let value):
            return Double(value)
        case .milliseconds(let value):
            return Double(value) / 1_000
        case .microseconds(let value):
            return Double(value) / 1_000_000
        case .nanoseconds(let value):
            return Double(value) / 1_000_000_000
        case .never:
            return Double.infinity
        @unknown default:
            return 0
        }
    }
}
