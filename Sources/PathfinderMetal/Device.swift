import CoreGraphics
import Metal
import MetalKit

public class PFDevice {
    static let FIRST_VERTEX_BUFFER_INDEX: UInt64 = 16

    struct StagingBuffer {
        var buffer: MTLBuffer
        var event_value: UInt64
    }

    struct BufferAllocations {
        var `private`: MTLBuffer?
        var shared: StagingBuffer?
        var byte_size: UInt64
    }

    enum BufferUploadMode {
        case `static`
        case dynamic

        func to_metal_resource_options() -> MTLResourceOptions {
            var options: MTLResourceOptions
            switch self {
            case .static:
                options = .cpuCacheModeWriteCombined
            case .dynamic:
                options = []
            }

            options.formUnion(.storageModePrivate)
            return options
        }
    }

    // MetalBuffer
    struct Buffer {
        var allocations: BufferAllocations
        var mode: BufferUploadMode
    }

    enum MetalDataReceiverState<T> {
        case pending
        case downloaded(T)
        case finished
    }

    class MetalBufferDataReceiverInfo {
        let cond = NSCondition()
        var state: MetalDataReceiverState<Data>
        var stagingBuffer: MTLBuffer

        init(stagingBuffer: MTLBuffer, state: MetalDataReceiverState<Data>) {
            self.stagingBuffer = stagingBuffer
            self.state = state
        }

        func download() {
            let staging_buffer_contents = stagingBuffer.contents()
            let staging_buffer_length = stagingBuffer.length

            let contents = Data(bytes: staging_buffer_contents, count: staging_buffer_length)

            cond.lock()
            defer { cond.unlock() }

            state = .downloaded(contents)
            cond.broadcast()
        }
    }

    // MetalBufferDataReceiver
    struct BufferDataReceiver {
        var value: MetalBufferDataReceiverInfo
    }

    enum MetalFenceStatus {
        case pending
        case resolved
    }

    class MetalFenceInfo {
        private let mutex = NSLock()
        private let cond = NSCondition()
        private var state: MetalFenceStatus

        init(state: MetalFenceStatus) {
            self.state = state
        }
    }

    // MetalFence
    struct Fence {
        var value: MetalFenceInfo
    }

    // MetalFramebuffer
    struct Framebuffer {
        var value: Texture
    }

    enum ProgramKind<T> {
        case raster(
            vertex: T,
            fragment: T
        )

        case compute(T)
    }

    // MetalImageParameter
    struct ImageParameter {
        var indices: ProgramKind<Int?>?
        var name: String
    }

    struct MetalRasterProgram {
        var vertex_shader: Shader
        var fragment_shader: Shader
    }

    struct MetalComputeProgram {
        var shader: Shader
        var local_size: MTLSize
    }

    // MetalProgram
    enum Program {
        case raster(MetalRasterProgram)
        case compute(MetalComputeProgram)
    }

    struct ArgumentArray {
        let argumentEncoder: MTLArgumentEncoder
        let argumentBuffer: MTLBuffer
    }

    // MetalShader
    struct Shader {
        var library: MTLLibrary
        var function: MTLFunction
        var name: String
        var arguments: [MTLBinding]?
    }

    // MetalStorageBuffer
    struct StorageBuffer {
        var indices: ProgramKind<Int?>?
        var name: String
    }

    // MetalTexture
    struct Texture {
        var private_texture: MTLTexture
        var shared_buffer: MTLBuffer?
        var sampling_flags: RenderCommand.TextureSamplingFlags
    }

    enum TextureData {
        case U8([UInt8])
        case U16([UInt16])
        case F16([Float16])
        case F32([Float32])
    }

    class MetalTextureDataReceiverInfo {
        private let mutex = NSLock()
        private let cond = NSCondition()
        private var state: MetalDataReceiverState<TextureData>
        var texture: MTLTexture
        var viewport: RectI

        init(texture: MTLTexture, viewport: RectI, state: MetalDataReceiverState<TextureData>) {
            self.texture = texture
            self.viewport = viewport
            self.state = state
        }
    }

    // MetalTextureDataReceiver
    struct TextureDataReceiver {
        var value: MetalTextureDataReceiverInfo
    }

    struct MetalTextureIndex {
        var main: Int
        var sampler: Int
    }

    // MetalTextureParameter
    struct TextureParameter {
        var indices: ProgramKind<MetalTextureIndex?>?
        var name: String
    }

    struct MetalTimerQueryData {
        var start_time: DispatchTime?
        var end_time: DispatchTime?
        //        var start_block: Option<RcBlock<(*mut Object, u64), ()>>,
        //        var end_block: Option<RcBlock<(*mut Object, u64), ()>>,
        var start_event_value: UInt64
    }

    class MetalTimerQueryInfo {
        private let mutex = NSLock()
        private let cond = NSCondition()
        private var state: MetalTimerQueryData

        init(state: MetalTimerQueryData) {
            self.state = state
        }
    }

    // MetalTimerQuery
    struct TimerQuery {
        var value: MetalTimerQueryInfo
    }

    // MetalUniform
    struct Uniform {
        var indices: ProgramKind<Int?>?
        var name: String
    }

    // MetalVertexArray
    struct VertexArray {
        var descriptor: MTLVertexDescriptor
        var vertex_buffers: [Buffer]
        var index_buffer: Buffer?
    }

    struct Scope {
        var command_buffer: MTLCommandBuffer
    }

    enum BufferData<T> {
        case uninitialized(Int)
        case memory([T])
    }

    enum BufferTarget {
        case vertex
        case index
        case storage
    }

    class BufferUploadEventData {
        let cond = NSCondition()
        var state: UInt64

        init(state: UInt64) {
            self.state = state
        }
    }

    enum TextureDataRef {
        case u8([UInt8])
        case f16([Float16])
        case f32([Float32])

        func check_and_extract_data_ptr(
            _ minimumSize: SIMD2<Int32>,
            _ format: TextureAllocation.TextureFormat
        ) -> UnsafeRawPointer {
            switch self {
            case .u8(let data):
                return data.withUnsafeBytes { bytes in
                    return bytes.baseAddress!
                }

            case .f16(let data):
                return data.withUnsafeBytes { bytes in
                    return bytes.baseAddress!
                }

            case .f32(let data):
                return data.withUnsafeBytes { bytes in
                    return bytes.baseAddress!
                }
            }
        }
    }

    enum ShaderKind {
        case vertex
        case fragment
        case compute
    }

    enum VertexAttrClass {
        case float
        case floatNorm
        case int
    }

    enum VertexAttrType {
        case f32
        case i8
        case i16
        case i32
        case u8
        case u16
    }

    struct VertexAttrDescriptor {
        var size: Int
        var `class`: VertexAttrClass
        var attr_type: VertexAttrType
        var stride: Int
        var offset: Int
        var divisor: UInt32
        var buffer_index: UInt32
    }

    struct UniformBuffer {
        var data: [UInt8]
        var ranges: [Range<Int>]
    }

    typealias VertexAttr = MTLVertexAttribute

    var main_color_texture: MTLTexture
    var main_depth_stencil_texture: MTLTexture
    var compute_fence: MTLFence?

    var sharedDevice: Device

    public init(_ device: Device, texture: CAMetalDrawable) {
        let texture = texture.texture

        let framebuffer_size = SIMD2<Int32>(Int32(texture.width), Int32(texture.height))
        let main_depth_stencil_texture = Self.createDepthStencilTexture(
            device: device.metalDevice,
            framebuffer_size
        )

        self.sharedDevice = device
        self.main_color_texture = texture
        self.main_depth_stencil_texture = main_depth_stencil_texture
        self.compute_fence = nil
    }

    static func createDepthStencilTexture(device: MTLDevice, _ size: SIMD2<Int32>) -> MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type2D
        descriptor.pixelFormat = .depth32Float_stencil8
        descriptor.width = Int(size.x)
        descriptor.height = Int(size.y)
        descriptor.storageMode = .private
        descriptor.usage = .unknown
        return device.makeTexture(descriptor: descriptor)!
    }

    func backend_name() -> String {
        "Metal"
    }

    public func present_drawable(_ drawable: CAMetalDrawable) {
        self.sharedDevice.begin_commands()
        self.sharedDevice.scopedCommandBuffer.present(drawable)
        self.sharedDevice.end_commands()
    }

    private func toMetalSize(dimension: ProgramsD3D11.ComputeDimensions) -> MTLSize {
        MTLSize(width: Int(dimension.x), height: Int(dimension.y), depth: Int(dimension.z))
    }

    // FIXME(pcwalton): Is there a way to introspect the shader to find `gl_WorkGroupSize`? That
    // would obviate the need for this function.
    func set_compute_program_local_size(
        _ program: inout Program,
        _ new_local_size: ProgramsD3D11.ComputeDimensions
    ) {
        switch program {
        case .compute(var computeProgram):
            computeProgram.local_size = toMetalSize(dimension: new_local_size)
            program = .compute(computeProgram)
        default:
            fatalError("Program was not a compute program!")
        }
    }

    func get_uniform(_: Program, _ name: String) -> Uniform {
        .init(indices: nil, name: name)
    }

    func get_texture_parameter(_: Program, _ name: String) -> TextureParameter {
        .init(indices: nil, name: name)
    }

    func get_image_parameter(_: Program, _ name: String) -> ImageParameter {
        .init(indices: nil, name: name)
    }

    func get_storage_buffer(_: Program, _ name: String, _: UInt32) -> StorageBuffer {
        .init(indices: nil, name: name)
    }

    func set_texture_sampling_mode(
        _ texture: inout Texture,
        _ flags: RenderCommand.TextureSamplingFlags
    ) {
        texture.sampling_flags = flags
    }

    func readAllBytes(_ tex: MTLTexture) -> [UInt8] {
        let bpp: Int
        switch tex.pixelFormat {
        case .r8Unorm: bpp = 1
        case .bgra8Unorm: bpp = 4
        default: fatalError("unsupported")
        }
        let w = tex.width
        let h = tex.height
        var bytes = [UInt8](repeating: 0, count: w * h * bpp)
        tex.getBytes(
            &bytes,
            bytesPerRow: w * bpp,
            from: MTLRegionMake2D(0, 0, w, h),
            mipmapLevel: 0
        )
        return bytes
    }

    func recv_buffer(_ buffer_data_receiver: BufferDataReceiver) -> Data {
        buffer_data_receiver.value.cond.lock()
        defer { buffer_data_receiver.value.cond.unlock() }

        let start = DispatchTime.now()
        while true {
            if let buffer_data = getBufferData(buffer_data_receiver.value) {
                print("redvbuf", start.distance(to: .now()).seconds)
                return buffer_data
            }

            buffer_data_receiver.value.cond.wait()
        }
    }

    func try_recv_buffer(_ buffer_data_receiver: BufferDataReceiver) -> Data? {
        buffer_data_receiver.value.cond.lock()
        defer { buffer_data_receiver.value.cond.unlock() }
        return getBufferData(buffer_data_receiver.value)
    }

    private func getBufferData(_ info: MetalBufferDataReceiverInfo) -> Data? {
        switch info.state {
        case .pending, .finished:
            return nil
        case .downloaded(let data):
            info.state = .finished
            return data
        }
    }

    func render_target_color_texture(_ render_target: Renderer.RenderTarget) -> MTLTexture {
        switch render_target {
        case .default: self.main_color_texture
        case .framebuffer(let framebuffer): framebuffer.value.private_texture
        }
    }

    func render_target_depth_texture(_ render_target: Renderer.RenderTarget) -> MTLTexture? {
        switch render_target {
        case .default: self.main_depth_stencil_texture
        case .framebuffer: nil
        }
    }

    func create_render_pass_descriptor(_ render_state: RenderState) -> MTLRenderPassDescriptor {
        let render_pass_descriptor = MTLRenderPassDescriptor()
        let color_attachment = render_pass_descriptor.colorAttachments[0]!
        color_attachment.texture = self.render_target_color_texture(render_state.target)

        if let color = render_state.options.clear_ops.color {
            let color = MTLClearColor(
                red: Double(color.r),
                green: Double(color.g),
                blue: Double(color.b),
                alpha: Double(color.a)
            )
            color_attachment.clearColor = color
            color_attachment.loadAction = .clear
        } else {
            color_attachment.loadAction = .load
        }

        color_attachment.storeAction = .store

        let depth_stencil_texture = self.render_target_depth_texture(render_state.target)

        if let depth_stencil_texture {
            let depth_attachment = render_pass_descriptor.depthAttachment!
            let stencil_attachment = render_pass_descriptor.stencilAttachment!

            depth_attachment.texture = depth_stencil_texture
            stencil_attachment.texture = depth_stencil_texture

            if let depth = render_state.options.clear_ops.depth {
                depth_attachment.clearDepth = Double(depth)
                depth_attachment.loadAction = .clear
            } else {
                depth_attachment.loadAction = .load
            }

            depth_attachment.storeAction = .store

            if let value = render_state.options.clear_ops.stencil {
                stencil_attachment.clearStencil = UInt32(value)
                stencil_attachment.loadAction = .clear
            } else {
                stencil_attachment.loadAction = .load
            }

            stencil_attachment.storeAction = .store
        }

        return render_pass_descriptor
    }

    func set_viewport(_ encoder: MTLRenderCommandEncoder, _ viewport: RectI) {
        encoder.setViewport(
            MTLViewport(
                originX: Double(viewport.origin.x),
                originY: Double(viewport.origin.y),
                width: Double(viewport.size.x),
                height: Double(viewport.size.y),
                znear: 0.0,
                zfar: 1.0
            )
        )
    }

    func prepare_pipeline_color_attachment_for_render(
        _ pipeline_color_attachment: MTLRenderPipelineColorAttachmentDescriptor,
        _ render_state: RenderState
    ) -> MTLRenderPipelineColorAttachmentDescriptor {
        let pixel_format = self.render_target_color_texture(render_state.target).pixelFormat
        pipeline_color_attachment.pixelFormat = pixel_format

        if let blend = render_state.options.blend {
            pipeline_color_attachment.isBlendingEnabled = true
            pipeline_color_attachment.sourceRGBBlendFactor = blend.src_rgb_factor.to_metal_blend_factor()
            pipeline_color_attachment.destinationRGBBlendFactor = blend.dest_rgb_factor
                .to_metal_blend_factor()
            pipeline_color_attachment.sourceAlphaBlendFactor = blend.src_alpha_factor
                .to_metal_blend_factor()
            pipeline_color_attachment.destinationAlphaBlendFactor = blend.dest_alpha_factor
                .to_metal_blend_factor()

            let blend_op = blend.op.to_metal_blend_op()
            pipeline_color_attachment.rgbBlendOperation = blend_op
            pipeline_color_attachment.alphaBlendOperation = blend_op
        } else {
            pipeline_color_attachment.isBlendingEnabled = false
        }

        if render_state.options.color_mask {
            pipeline_color_attachment.writeMask = .all
        } else {
            pipeline_color_attachment.writeMask = MTLColorWriteMask()
        }

        return pipeline_color_attachment
    }

    func render_target_has_depth(_ render_target: Renderer.RenderTarget) -> Bool {
        switch render_target {
        case .default: true
        case .framebuffer: false
        }
    }

    func appendMatrixToArray<T>(_ array: inout [UInt8], matrix: inout SIMD4<T>) {
        withUnsafeBytes(of: &matrix) { bytes in array.append(contentsOf: bytes) }
    }

    func appendUniformData(data: inout [UInt8], _ uniformData: RenderState.UniformData) {
        switch uniformData {
        case .float(let value):
            data.append(contentsOf: withUnsafeBytes(of: value.bitPattern) { Array($0) })

        case .ivec2(let vector):
            data.append(contentsOf: withUnsafeBytes(of: vector.x.littleEndian) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: vector.y.littleEndian) { Array($0) })

        case .ivec3(let values):
            data.append(contentsOf: withUnsafeBytes(of: values[0].littleEndian) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: values[1].littleEndian) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: values[2].littleEndian) { Array($0) })

        case .int(let value):
            data.append(contentsOf: withUnsafeBytes(of: value.littleEndian) { Array($0) })

        case .mat2(let matrix):
            data.append(contentsOf: withUnsafeBytes(of: matrix.x.bitPattern) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: matrix.y.bitPattern) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: matrix.z.bitPattern) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: matrix.w.bitPattern) { Array($0) })

        case .mat4(let m1, let m2, let m3, let m4):
            for column in [m1, m2, m3, m4] {
                data.append(contentsOf: withUnsafeBytes(of: column[0].bitPattern) { Array($0) })
                data.append(contentsOf: withUnsafeBytes(of: column[1].bitPattern) { Array($0) })
                data.append(contentsOf: withUnsafeBytes(of: column[2].bitPattern) { Array($0) })
                data.append(contentsOf: withUnsafeBytes(of: column[3].bitPattern) { Array($0) })
            }

        case .vec2(let vector):
            data.append(contentsOf: withUnsafeBytes(of: vector.x.bitPattern) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: vector.y.bitPattern) { Array($0) })

        case .vec3(let array):
            data.append(contentsOf: withUnsafeBytes(of: array[0].bitPattern) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: array[1].bitPattern) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: array[2].bitPattern) { Array($0) })

        case .vec4(let vector):
            data.append(contentsOf: withUnsafeBytes(of: vector.x.bitPattern) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: vector.y.bitPattern) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: vector.z.bitPattern) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: vector.w.bitPattern) { Array($0) })
        }
    }

    func create_uniform_buffer(_ uniforms: [(Uniform, RenderState.UniformData)]) -> UniformBuffer {
        var uniform_buffer_data: [UInt8] = []
        var uniform_buffer_ranges: [Range<Int>] = []

        for (_, uniform_data) in uniforms {
            let start_index = uniform_buffer_data.count

            appendUniformData(data: &uniform_buffer_data, uniform_data)

            let end_index = uniform_buffer_data.count

            while uniform_buffer_data.count % 256 != 0 {
                uniform_buffer_data.append(0)
            }

            uniform_buffer_ranges.append(start_index..<end_index)
        }

        return .init(
            data: uniform_buffer_data,
            ranges: uniform_buffer_ranges
        )
    }

    func get_uniform_index(_ shader: Shader, _ name: String) -> Int? {
        let arguments = shader.arguments!
        let main_name = "u\(name)"

        for argument_index in 0..<arguments.count {
            let argument = arguments[argument_index]
            let argument_name = argument.name

            if argument_name == main_name {
                return argument.index
            }
        }

        return nil
    }

    func get_texture_index(_ shader: Shader, _ name: String) -> MetalTextureIndex? {
        guard let arguments = shader.arguments else { fatalError() }

        let (main_name, sampler_name) = ("u\(name)", "u\(name)Smplr")

        var main_index: Int?
        var sampler_index: Int?

        for argument_index in 0..<arguments.count {
            let argument = arguments[argument_index]
            let argument_name = argument.name

            if argument_name == main_name {
                main_index = argument.index
            }

            if argument_name == sampler_name {
                sampler_index = argument.index
            }
        }

        guard let main_index, let sampler_index else { return nil }

        return .init(main: main_index, sampler: sampler_index)
    }

    func get_image_index(_ shader: Shader, _ name: String) -> Int? {
        guard let arguments = shader.arguments else { fatalError() }

        let main_name = "u\(name)"
        for argument_index in 0..<arguments.count {
            let argument = arguments[argument_index]
            let argument_name = argument.name
            if argument_name == main_name {
                return argument.index
            }
        }

        return nil
    }

    func get_storage_buffer_index(_ shader: Shader, _ name: String) -> Int? {
        guard let arguments = shader.arguments else {
            fatalError()
        }

        let main_name = "i\(name)"
        var main_argument: Int? = nil
        for argument_index in 0..<arguments.count {
            let argument = arguments[argument_index]

            if argument.type != .buffer {
                continue
            }

            guard
                let bufferBinding = argument as? MTLBufferBinding,
                bufferBinding.bufferDataType == .struct
            else { continue }

            let struct_type = bufferBinding.bufferStructType
            if struct_type?.memberByName(main_name) != nil {
                main_argument = argument.index
            }
        }

        return main_argument
    }

    func populate_uniform_indices_if_necessary(
        _ uniform: inout Uniform,
        _ program: Program
    ) {
        if uniform.indices != nil {
            return
        }

        switch program {
        case .raster(let program):
            uniform.indices = .raster(
                vertex: self.get_uniform_index(program.vertex_shader, uniform.name),
                fragment: self.get_uniform_index(program.fragment_shader, uniform.name)
            )
        case .compute(let computeProgram):
            let uniform_index = self.get_uniform_index(computeProgram.shader, uniform.name)
            uniform.indices = .compute(uniform_index)
        }
    }

    func set_vertex_uniform(
        _ argument_index: Int,
        _ buffer: [UInt8],
        _ buffer_range: Range<Int>,
        _ render_command_encoder: MTLRenderCommandEncoder
    ) {
        let data = buffer.withUnsafeBytes { bytes in
            bytes.baseAddress!.advanced(by: buffer_range.lowerBound)
        }

        render_command_encoder.setVertexBytes(
            data,
            length: buffer_range.upperBound - buffer_range.lowerBound,
            index: argument_index
        )
    }

    func set_fragment_uniform(
        _ argument_index: Int,
        _ buffer: [UInt8],
        _ buffer_range: Range<Int>,
        _ render_command_encoder: MTLRenderCommandEncoder
    ) {

        let data = buffer.withUnsafeBytes { bytes in
            bytes.baseAddress!.advanced(by: buffer_range.lowerBound)
        }

        render_command_encoder.setFragmentBytes(
            data,
            length: buffer_range.upperBound - buffer_range.lowerBound,
            index: argument_index
        )
    }

    func populate_texture_indices_if_necessary(
        _ texture_parameter: inout TextureParameter,
        _ program: Program
    ) {
        if texture_parameter.indices != nil {
            return
        }

        switch program {
        case .raster(let raster):
            texture_parameter.indices = .raster(
                vertex: self.get_texture_index(raster.vertex_shader, texture_parameter.name),
                fragment: self.get_texture_index(raster.fragment_shader, texture_parameter.name)
            )
        case .compute(let compute):
            let image_index = self.get_texture_index(compute.shader, texture_parameter.name)
            texture_parameter.indices = .compute(image_index)
        }
    }

    func populate_image_indices_if_necessary(
        _ image_parameter: inout ImageParameter,
        _ program: Program
    ) {
        if image_parameter.indices != nil {
            return
        }

        switch program {
        case .raster(let raster):
            image_parameter.indices = .raster(
                vertex: self.get_image_index(raster.vertex_shader, image_parameter.name),
                fragment: self.get_image_index(raster.fragment_shader, image_parameter.name)
            )
        case .compute(let compute):
            let image_index = self.get_image_index(compute.shader, image_parameter.name)
            image_parameter.indices = .compute(image_index)
        }
    }

    func populate_storage_buffer_indices_if_necessary(
        _ storage_buffer: inout StorageBuffer,
        _ program: Program
    ) {
        if storage_buffer.indices != nil {
            return
        }

        switch program {
        case .raster(let rasterProgram):
            storage_buffer.indices = .raster(
                vertex: self.get_storage_buffer_index(rasterProgram.vertex_shader, storage_buffer.name),
                fragment: self.get_storage_buffer_index(rasterProgram.fragment_shader, storage_buffer.name)
            )
        case .compute(let computeProgram):
            let storage_buffer_index = self.get_storage_buffer_index(
                computeProgram.shader,
                storage_buffer.name
            )
            storage_buffer.indices = .compute(storage_buffer_index)
        }
    }

    func encode_vertex_texture_parameter(
        _ argument_index: MetalTextureIndex,
        _ render_command_encoder: MTLRenderCommandEncoder,
        _ texture: Texture
    ) {
        render_command_encoder.setVertexTexture(texture.private_texture, index: argument_index.main)
        let sampler = self.sharedDevice.samplers[Int(texture.sampling_flags.rawValue)]
        render_command_encoder.setVertexSamplerState(sampler, index: argument_index.sampler)
    }

    func encode_fragment_texture_parameter(
        _ argument_index: MetalTextureIndex,
        _ render_command_encoder: MTLRenderCommandEncoder,
        _ texture: Texture
    ) {
        render_command_encoder.setFragmentTexture(texture.private_texture, index: argument_index.main)
        let sampler = self.sharedDevice.samplers[Int(texture.sampling_flags.rawValue)]
        render_command_encoder.setFragmentSamplerState(sampler, index: argument_index.sampler)
    }

    func set_raster_uniforms(
        _ render_command_encoder: MTLRenderCommandEncoder,
        _ render_state: RenderState
    ) {
        let program =
            switch render_state.program {
            case .raster(let raster_program): raster_program
            default: fatalError()
            }

        let vertex_arguments = program.vertex_shader.arguments
        let fragment_arguments = program.fragment_shader.arguments
        if vertex_arguments == nil && fragment_arguments == nil {
            return
        }

        // Set uniforms.
        let uniform_buffer = self.create_uniform_buffer(render_state.uniforms)
        for ((uniform, vv), buffer_range) in zip(render_state.uniforms, uniform_buffer.ranges) {
            let indices = uniform.indices!
            let (vertex_indices, fragment_indices) =
                switch indices {
                case .raster(let vertex, let fragment): (vertex, fragment)
                default: fatalError()
                }

            if let vertex_index = vertex_indices {
                self.set_vertex_uniform(
                    vertex_index,
                    uniform_buffer.data,
                    buffer_range,
                    render_command_encoder
                )
            }

            if let fragment_index = fragment_indices {
                self.set_fragment_uniform(
                    fragment_index,
                    uniform_buffer.data,
                    buffer_range,
                    render_command_encoder
                )
            }
        }

        // Set textures.
        for (texture_param, texture) in render_state.textures {
            let indices = texture_param.indices!
            let (vertex_indices, fragment_indices) =
                switch indices {
                case .raster(let vertex, let fragment): (vertex, fragment)
                default: fatalError()
                }

            if let vertex_index = vertex_indices {
                self.encode_vertex_texture_parameter(
                    vertex_index,
                    render_command_encoder,
                    texture
                )
            }

            if let fragment_index = fragment_indices {
                self.encode_fragment_texture_parameter(
                    fragment_index,
                    render_command_encoder,
                    texture
                )
            }
        }

        // Set images.
        for (image_param, image, vv) in render_state.images {
            let indices = image_param.indices!
            let (vertex_indices, fragment_indices) =
                switch indices {
                case .raster(let vertex, let fragment): (vertex, fragment)
                default: fatalError()
                }

            if let vertex_index = vertex_indices {
                render_command_encoder.setVertexTexture(image.private_texture, index: vertex_index)
            }

            if let fragment_index = fragment_indices {
                render_command_encoder.setFragmentTexture(image.private_texture, index: fragment_index)
            }
        }

        // Set storage buffers.
        for (storage_buffer_id, storage_buffer_binding) in render_state.storage_buffers {
            let indices = storage_buffer_id.indices!
            let (vertex_indices, fragment_indices) =
                switch indices {
                case .raster(let vertex, let fragment): (vertex, fragment)
                default: fatalError()
                }

            if let vertex_index = vertex_indices {
                if let buffer = storage_buffer_binding.allocations.private {
                    render_command_encoder.setVertexBuffer(buffer, offset: 0, index: vertex_index)
                }
            }
            if let fragment_index = fragment_indices {
                if let buffer = storage_buffer_binding.allocations.private {
                    render_command_encoder.setFragmentBuffer(buffer, offset: 0, index: fragment_index)
                }
            }
        }
    }

    func set_depth_stencil_state(
        _ encoder: MTLRenderCommandEncoder,
        _ render_state: RenderState
    ) {
        let depth_stencil_descriptor = MTLDepthStencilDescriptor()

        if let depth_state = render_state.options.depth {
            let compare_function = depth_state.func.to_metal_compare_function()
            depth_stencil_descriptor.depthCompareFunction = compare_function
            depth_stencil_descriptor.isDepthWriteEnabled = depth_state.write
        } else {
            depth_stencil_descriptor.depthCompareFunction = .always
            depth_stencil_descriptor.isDepthWriteEnabled = false
        }

        if let stencil_state = render_state.options.stencil {
            let stencil_descriptor = MTLStencilDescriptor()
            let compare_function = stencil_state.func.to_metal_compare_function()
            let (pass_operation, write_mask) =
                if stencil_state.write {
                    (MTLStencilOperation.replace, stencil_state.mask)
                } else {
                    (MTLStencilOperation.keep, 0)
                }

            stencil_descriptor.stencilCompareFunction = compare_function
            stencil_descriptor.stencilFailureOperation = .keep
            stencil_descriptor.depthFailureOperation = .keep
            stencil_descriptor.depthStencilPassOperation = pass_operation
            stencil_descriptor.writeMask = write_mask
            depth_stencil_descriptor.frontFaceStencil = stencil_descriptor
            depth_stencil_descriptor.backFaceStencil = stencil_descriptor
            encoder.setStencilReferenceValue(stencil_state.reference)
        } else {
            depth_stencil_descriptor.frontFaceStencil = nil
            depth_stencil_descriptor.backFaceStencil = nil
        }

        let depth_stencil_state = self.sharedDevice.metalDevice.makeDepthStencilState(
            descriptor: depth_stencil_descriptor
        )
        encoder.setDepthStencilState(depth_stencil_state)
    }

    private func prepare_to_draw(_ render_state: RenderState) -> MTLRenderCommandEncoder {
        let command_buffer = sharedDevice.scopedCommandBuffer

        let render_pass_descriptor = self.create_render_pass_descriptor(render_state)

        let encoder = command_buffer.makeRenderCommandEncoder(descriptor: render_pass_descriptor)!

        // Wait on the previous compute command, if any.
        let compute_fence = self.compute_fence
        if let compute_fence {
            encoder.waitForFence(compute_fence, before: .vertex)
        }

        let program =
            switch render_state.program {
            case .raster(let raster_program): raster_program
            default: fatalError("Raster render command must use a raster program!")
            }

        let info = SharedResource.rasterPipelineState(
            device: sharedDevice,
            vertex: program.vertex_shader.function,
            fragment: program.fragment_shader.function,
            hasDepth: self.render_target_has_depth(render_state.target),
            hasColorMask: render_state.options.color_mask,
            descriptor: render_state.vertex_array.descriptor
        )

        for (vertex_buffer_index, vertex_buffer) in render_state.vertex_array.vertex_buffers
            .enumerated()
        {
            let real_index = vertex_buffer_index + Int(PFDevice.FIRST_VERTEX_BUFFER_INDEX)
            let buffer = vertex_buffer.allocations.private
            encoder.setVertexBuffer(buffer, offset: 0, index: real_index)
        }

        encoder.setRenderPipelineState(info.status)
        self.set_viewport(encoder, render_state.viewport)
        self.set_raster_uniforms(encoder, render_state)
        self.set_depth_stencil_state(encoder, render_state)

        return encoder
    }

    func draw_elements(_ index_count: Int, _ render_state: RenderState) {
        let encoder = self.prepare_to_draw(render_state)
        let primitive = render_state.primitive.to_metal_primitive()
        let index_type = MTLIndexType.uint32
        let index_buffer = render_state.vertex_array.index_buffer!.allocations.private!
        encoder.drawIndexedPrimitives(
            type: primitive,
            indexCount: index_count,
            indexType: index_type,
            indexBuffer: index_buffer,
            indexBufferOffset: 0
        )
        encoder.endEncoding()
    }

    func dispatch_compute(
        _ size: ProgramsD3D11.ComputeDimensions,
        _ compute_state: ComputeState
    ) {
        // No-op if there is nothing to dispatch. Metal requires all dispatch dimensions > 0.
        if size.x == 0 || size.y == 0 || size.z == 0 {
            return
        }

        let command_buffer = sharedDevice.scopedCommandBuffer
        let encoder = command_buffer.makeComputeCommandEncoder()!

        let program: MetalComputeProgram
        switch compute_state.program {
        case .compute(let compute_program):
            program = compute_program
        default:
            fatalError("Compute render command must use a compute program!")
        }

        let info = SharedResource.computePipelineDescriptor(device: sharedDevice, function: program.shader.function)
        let compute_pipeline_state = info.status

        encoder.setComputePipelineState(compute_pipeline_state)
        set_compute_uniforms(encoder, compute_state)

        let local_size: MTLSize
        switch compute_state.program {
        case .compute(let compute_program):
            local_size = compute_program.local_size
        default:
            fatalError("Program was not a compute program!")
        }

        encoder.dispatchThreadgroups(size.to_metal_size(), threadsPerThreadgroup: local_size)

        let fence = sharedDevice.metalDevice.makeFence()!
        encoder.updateFence(fence)
        self.compute_fence = fence

        encoder.endEncoding()
    }

    func set_compute_uniforms(
        _ compute_command_encoder: MTLComputeCommandEncoder,
        _ compute_state: ComputeState
    ) {
        // Set uniforms.
        let uniform_buffer = create_uniform_buffer(compute_state.uniforms)
        for ((uniform, vv), buffer_range) in zip(compute_state.uniforms, uniform_buffer.ranges) {
            let indices = uniform.indices!
            guard case .compute(let compute_indices) = indices else {
                fatalError("Expected compute program indices")
            }

            if let indices = compute_indices {
                set_compute_uniform(
                    indices,
                    uniform_buffer.data,
                    buffer_range,
                    compute_command_encoder
                )
            }
        }

        // Set textures.
        for (texture_param, texture) in compute_state.textures {
            let indices = texture_param.indices!
            guard case .compute(let compute_indices) = indices else {
                fatalError("Expected compute program indices")
            }

            if let indices = compute_indices {
                encode_compute_texture_parameter(indices, compute_command_encoder, texture)
            }
        }

        // Set images.
        for (image_param, image, vv) in compute_state.images {
            let indices = image_param.indices!
            guard case .compute(let compute_indices) = indices else {
                fatalError("Expected compute program indices")
            }

            if let indices = compute_indices {
                compute_command_encoder.setTexture(image.private_texture, index: indices)
            }
        }

        // Set storage buffers.
        for (storage_buffer_id, storage_buffer_binding) in compute_state.storage_buffers {
            let indices = storage_buffer_id.indices!
            guard case .compute(let compute_indices) = indices else {
                fatalError("Expected compute program indices")
            }

            if let index = compute_indices {
                if let buffer = storage_buffer_binding.allocations.private {
                    compute_command_encoder.setBuffer(buffer, offset: 0, index: index)
                }
            }
        }
    }

    func set_compute_uniform(
        _ argument_index: Int,
        _ buffer: [UInt8],
        _ buffer_range: Range<Int>,
        _ compute_command_encoder: MTLComputeCommandEncoder
    ) {
        buffer.withUnsafeBufferPointer { bufferPtr in
            let startPtr = bufferPtr.baseAddress!.advanced(by: buffer_range.lowerBound)
            let length = buffer_range.count
            compute_command_encoder.setBytes(startPtr, length: length, index: argument_index)
        }
    }

    func encode_compute_texture_parameter(
        _ argument_index: MetalTextureIndex,
        _ compute_command_encoder: MTLComputeCommandEncoder,
        _ texture: Texture
    ) {
        compute_command_encoder.setTexture(texture.private_texture, index: argument_index.main)
        let sampler = sharedDevice.samplers[Int(texture.sampling_flags.rawValue)]
        compute_command_encoder.setSamplerState(sampler, index: argument_index.sampler)
    }
}

public class Device {
    let metalDevice: MTLDevice
    let command_queue: MTLCommandQueue
    let samplers: [MTLSamplerState]
    let buffer_upload_shared_event: MTLSharedEvent
    let dispatch_queue: DispatchQueue
    let shared_event_listener: MTLSharedEventListener
    let buffer_upload_event_data: PFDevice.BufferUploadEventData

    var scopes: [PFDevice.Scope] = []
    var next_buffer_upload_event_value: UInt64 = 1

    var scopedCommandBuffer: MTLCommandBuffer { scopes.last!.command_buffer }

    public init(device: MTLDevice) {
        metalDevice = device

        samplers = (0..<16).map { sampling_flags_value in
            let sampling_flags = RenderCommand.TextureSamplingFlags(
                rawValue: UInt8(sampling_flags_value)
            )
            let sampler_descriptor = MTLSamplerDescriptor()
            sampler_descriptor.supportArgumentBuffers = true
            sampler_descriptor.normalizedCoordinates = true
            sampler_descriptor.minFilter = sampling_flags.contains(.NEAREST_MIN) ? .nearest : .linear
            sampler_descriptor.magFilter = sampling_flags.contains(.NEAREST_MAG) ? .nearest : .linear
            sampler_descriptor.sAddressMode = sampling_flags.contains(.REPEAT_U) ? .repeat : .clampToEdge
            sampler_descriptor.tAddressMode = sampling_flags.contains(.REPEAT_V) ? .repeat : .clampToEdge

            return device.makeSamplerState(descriptor: sampler_descriptor)!
        }

        buffer_upload_shared_event = device.makeSharedEvent()!

        dispatch_queue = DispatchQueue(
            label: "graphics.pathfinder.queue",
            attributes: .concurrent
        )

        shared_event_listener = MTLSharedEventListener(dispatchQueue: dispatch_queue)
        buffer_upload_event_data = .init(state: 0)
        command_queue = device.makeCommandQueue()!
    }

    func begin_commands() {
        let commandBuffer = command_queue.makeCommandBuffer()!
        scopes.append(.init(command_buffer: commandBuffer))
    }

    func end_commands() {
        let scope = scopes.popLast()
        scope?.command_buffer.commit()
    }

    func upload_to_buffer<T>(
        _ dest_buffer: inout PFDevice.Buffer,
        _ start: Int,
        _ data: [T],
        _ target: PFDevice.BufferTarget
    ) {
        if data.isEmpty {
            return
        }

        let byte_start = start * MemoryLayout<T>.stride
        let byte_size = data.count * MemoryLayout<T>.stride

        if dest_buffer.allocations.shared == nil {
            let resource_options: MTLResourceOptions = [.cpuCacheModeWriteCombined, .storageModeShared]
            dest_buffer.allocations.shared = .init(
                buffer: metalDevice.makeBuffer(
                    length: Int(dest_buffer.allocations.byte_size),
                    options: resource_options
                )!,
                event_value: 0
            )
        }

        if dest_buffer.allocations.shared!.event_value != 0 {
            self.buffer_upload_event_data.cond.lock()
            defer { self.buffer_upload_event_data.cond.unlock() }

            var value = self.buffer_upload_event_data.state

            while value < dest_buffer.allocations.shared!.event_value {
                self.buffer_upload_event_data.cond.wait()
                value = self.buffer_upload_event_data.state
            }
        }

        let destinationPtr = dest_buffer.allocations.shared!.buffer.contents().advanced(
            by: Int(byte_start)
        )
        data.withUnsafeBytes { bytes in
            destinationPtr.copyMemory(from: bytes.baseAddress!, byteCount: Int(byte_size))
        }

        dest_buffer.allocations.shared?.event_value = self.next_buffer_upload_event_value
        self.next_buffer_upload_event_value += 1

        let scopes = self.scopes
        let command_buffer = scopes.last!.command_buffer
        let blit_command_encoder = command_buffer.makeBlitCommandEncoder()!

        blit_command_encoder.copy(
            from: dest_buffer.allocations.shared!.buffer,
            sourceOffset: byte_start,
            to: dest_buffer.allocations.private!,
            destinationOffset: byte_start,
            size: byte_size
        )

        blit_command_encoder.endEncoding()

        let event_value = dest_buffer.allocations.shared!.event_value

        command_buffer.encodeSignalEvent(self.buffer_upload_shared_event, value: event_value)

        let listenerBlock: MTLSharedEventNotificationBlock = { (event, value) in
            self.buffer_upload_event_data.cond.lock()
            defer { self.buffer_upload_event_data.cond.unlock() }

            self.buffer_upload_event_data.state = max(self.buffer_upload_event_data.state, event_value)
            self.buffer_upload_event_data.cond.broadcast()
        }

        self.buffer_upload_shared_event.notify(
            self.shared_event_listener,
            atValue: event_value,
            block: listenerBlock
        )

        // Flush to avoid deadlock.
        self.end_commands()
        self.begin_commands()
    }

    func upload_png_to_texture(
        _ name: String,
        _ texture: inout PFDevice.Texture,
        _ format: TextureAllocation.TextureFormat
    ) {
        guard
            let url = Bundle.module.url(
                forResource: name,
                withExtension: "png",
                subdirectory: "Resources/Shaders"
            )
        else { fatalError("LUT PNG not found: \(name)") }

        let loader = MTKTextureLoader(device: metalDevice)
        let options: [MTKTextureLoader.Option: Any] = [
            .SRGB: false,
            .origin: MTKTextureLoader.Origin.topLeft,
            .allocateMipmaps: false,
            .generateMipmaps: false,
        ]

        guard let srcTex = try? loader.newTexture(URL: url, options: options) else {
            fatalError("Failed to load LUT \(name)")
        }

        let w = srcTex.width
        let h = srcTex.height
        let region = MTLRegionMake2D(0, 0, w, h)

        switch format {
        case .rgba8:
            var rgba = [UInt8](repeating: 0, count: w * h * 4)
            switch srcTex.pixelFormat {
            case .rgba8Unorm, .rgba8Unorm_srgb:
                srcTex.getBytes(&rgba, bytesPerRow: w * 4, from: region, mipmapLevel: 0)
            case .bgra8Unorm, .bgra8Unorm_srgb:
                var bgra = [UInt8](repeating: 0, count: w * h * 4)
                srcTex.getBytes(&bgra, bytesPerRow: w * 4, from: region, mipmapLevel: 0)
                // BGRA -> RGBA
                for i in 0..<(w * h) {
                    let b = bgra[i * 4 + 0]
                    let g = bgra[i * 4 + 1]
                    let r = bgra[i * 4 + 2]
                    let a = bgra[i * 4 + 3]
                    rgba[i * 4 + 0] = r
                    rgba[i * 4 + 1] = g
                    rgba[i * 4 + 2] = b
                    rgba[i * 4 + 3] = a
                }
            default:
                fatalError("Unexpected pixelFormat for \(name): \(srcTex.pixelFormat)")
            }
            let rect = RectI(origin: .init(0, 0), size: .init(Int32(w), Int32(h)))
            upload_to_texture(&texture, rect, .u8(rgba))

        case .r8:
            var r8 = [UInt8](repeating: 0, count: w * h)
            switch srcTex.pixelFormat {
            case .r8Unorm:
                srcTex.getBytes(&r8, bytesPerRow: w, from: region, mipmapLevel: 0)
            case .rgba8Unorm, .rgba8Unorm_srgb, .bgra8Unorm, .bgra8Unorm_srgb:
                var rgba = [UInt8](repeating: 0, count: w * h * 4)
                srcTex.getBytes(&rgba, bytesPerRow: w * 4, from: region, mipmapLevel: 0)
                let rIndex =
                    (srcTex.pixelFormat == .bgra8Unorm || srcTex.pixelFormat == .bgra8Unorm_srgb) ? 2 : 0
                for i in 0..<(w * h) {
                    r8[i] = rgba[i * 4 + rIndex]
                }
            default:
                fatalError("Unexpected pixelFormat for \(name): \(srcTex.pixelFormat)")
            }
            let rect = RectI(origin: .init(0, 0), size: .init(Int32(w), Int32(h)))
            upload_to_texture(&texture, rect, .u8(r8))

        default:
            fatalError("Unsupported LUT texture format for \(name): \(format)")
        }
    }

    func upload_to_texture(
        _ dest_texture: inout PFDevice.Texture,
        _ rect: RectI,
        _ data: PFDevice.TextureDataRef
    ) {
        let command_buffer = scopedCommandBuffer

        let texture_size = self.texture_size(dest_texture)
        let texture_format = self.texture_format(dest_texture)

        let bytes_per_pixel = texture_format.bytes_per_pixel
        let texture_byte_size = Int(texture_size.x) * Int(texture_size.y) * bytes_per_pixel

        if dest_texture.shared_buffer == nil {
            let buffer = metalDevice.makeBuffer(
                length: texture_byte_size,
                options: [.cpuCacheModeWriteCombined, .storageModeShared]
            )
            dest_texture.shared_buffer = buffer
        }

        // TODO(pcwalton): Wait if necessary...
        let texture_data_ptr = data.check_and_extract_data_ptr(rect.size, texture_format)

        let src_stride = Int(rect.width) * bytes_per_pixel
        let dest_stride = Int(texture_size.x) * bytes_per_pixel

        let dest_contents = dest_texture.shared_buffer!.contents().assumingMemoryBound(to: UInt8.self)

        for srcY in 0..<rect.height {
            let destY = srcY + rect.originY
            let srcOffset = Int(srcY) * src_stride
            let destOffset = Int(destY) * dest_stride + Int(rect.originX) * bytes_per_pixel

            let srcPtr = texture_data_ptr.advanced(by: srcOffset).assumingMemoryBound(to: UInt8.self)
            let destPtr = dest_contents.advanced(by: destOffset)

            destPtr.initialize(from: srcPtr, count: src_stride)
        }

        let src_size = MTLSize(width: Int(rect.width), height: Int(rect.height), depth: 1)
        let dest_origin = MTLOrigin(x: Int(rect.originX), y: Int(rect.originY), z: 0)

        let dest_byte_offset = Int(rect.originY) * src_stride + Int(rect.originX) * bytes_per_pixel

        let blitCommandEncoder = command_buffer.makeBlitCommandEncoder()!
        blitCommandEncoder.copy(
            from: dest_texture.shared_buffer!,
            sourceOffset: dest_byte_offset,
            sourceBytesPerRow: dest_stride,
            sourceBytesPerImage: 0,
            sourceSize: src_size,
            to: dest_texture.private_texture,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: dest_origin
        )
        blitCommandEncoder.endEncoding()
    }

    func texture_format(_ texture: PFDevice.Texture) -> TextureAllocation.TextureFormat {
        switch texture.private_texture.pixelFormat {
        case .r8Unorm: return .r8
        case .r16Float: return .r16F
        case .rgba8Unorm: return .rgba8
        case .bgra8Unorm_srgb: return .bgra8
        case .rgba16Float: return .rgba16F
        case .rgba32Float: return .rgba32F
        default: fatalError("Unexpected Metal texture format!")
        }
    }

    func texture_size(_ texture: PFDevice.Texture) -> SIMD2<Int32> {
        .init(Int32(texture.private_texture.width), Int32(texture.private_texture.height))
    }

    func read_buffer(
        _ src_buffer: PFDevice.Buffer,
        _ target: PFDevice.BufferTarget,
        _ range: Range<Int>
    ) -> PFDevice.BufferDataReceiver {
        let buffer_data_receiver: PFDevice.BufferDataReceiver

        do {
            let command_buffer = scopedCommandBuffer

            var src_allocations = src_buffer.allocations
            guard let src_private_buffer = src_allocations.private else {
                fatalError("Private buffer not allocated!")
            }

            if src_allocations.shared == nil {
                let resource_options: MTLResourceOptions = [
                    .cpuCacheModeWriteCombined,
                    .storageModeShared,
                ]
                src_allocations.shared = .init(
                    buffer: metalDevice.makeBuffer(
                        length: Int(src_allocations.byte_size),
                        options: resource_options
                    )!,
                    event_value: 0
                )
            }

            let byte_size = range.count
            let blit_command_encoder = command_buffer.makeBlitCommandEncoder()!

            blit_command_encoder.copy(
                from: src_private_buffer,
                sourceOffset: 0,
                to: src_allocations.shared!.buffer,
                destinationOffset: range.lowerBound,
                size: byte_size
            )

            buffer_data_receiver = .init(
                value: .init(stagingBuffer: src_allocations.shared!.buffer, state: .pending)
            )

            blit_command_encoder.endEncoding()

            let buffer_data_receiver_for_block = buffer_data_receiver
            let block: MTLCommandBufferHandler = { _ in
                buffer_data_receiver_for_block.value.download()
            }
            command_buffer.addCompletedHandler(block)
        }

        end_commands()
        begin_commands()

        return buffer_data_receiver
    }

    func allocate_buffer<T>(
        _ buffer: inout PFDevice.Buffer,
        _ data: PFDevice.BufferData<T>,
        _ target: PFDevice.BufferTarget
    ) {
        let options = buffer.mode.to_metal_resource_options()

        let length: Int
        switch data {
        case .uninitialized(let size):
            length = size
        case .memory(let slice):
            length = slice.count
        }

        let byte_size = length * MemoryLayout<T>.stride
        let new_buffer = metalDevice.makeBuffer(length: byte_size, options: options)

        buffer.allocations = .init(
            private: new_buffer,
            shared: nil,
            byte_size: UInt64(byte_size)
        )

        switch data {
        case .uninitialized:
            break
        case .memory(let slice):
            self.upload_to_buffer(&buffer, 0, slice, target)
        }
    }

    func create_texture_descriptor(
        _ format: TextureAllocation.TextureFormat,
        _ size: SIMD2<Int32>
    )
        -> MTLTextureDescriptor
    {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type2D

        switch format {
        case .r8: descriptor.pixelFormat = .r8Unorm
        case .r16F: descriptor.pixelFormat = .r16Float
        case .rgba8: descriptor.pixelFormat = .rgba8Unorm
        case .bgra8: descriptor.pixelFormat = .bgra8Unorm_srgb
        case .rgba16F: descriptor.pixelFormat = .rgba16Float
        case .rgba32F: descriptor.pixelFormat = .rgba32Float
        }

        descriptor.width = Int(size.x)
        descriptor.height = Int(size.y)
        descriptor.usage = .unknown
        return descriptor
    }

    // TODO: Add texture usage hint.
    func create_texture(_ format: TextureAllocation.TextureFormat, _ size: SIMD2<Int32>) -> PFDevice.Texture {
        let descriptor = create_texture_descriptor(format, size)
        descriptor.storageMode = .private

        return .init(
            private_texture: metalDevice.makeTexture(descriptor: descriptor)!,
            shared_buffer: nil,
            sampling_flags: .init()
        )
    }

    func create_framebuffer(_ texture: PFDevice.Texture) -> PFDevice.Framebuffer {
        .init(value: texture)
    }

    func create_buffer(_ mode: PFDevice.BufferUploadMode) -> PFDevice.Buffer {
        .init(
            allocations: .init(private: nil, shared: nil, byte_size: 0),
            mode: mode
        )
    }

    func bind_buffer(
        _ vertex_array: inout PFDevice.VertexArray,
        _ buffer: PFDevice.Buffer,
        _ target: PFDevice.BufferTarget
    ) {
        switch target {
        case .vertex:
            vertex_array.vertex_buffers.append(buffer)
        case .index:
            vertex_array.index_buffer = buffer
        default:
            fatalError("Buffers bound to vertex arrays must be vertex or index buffers!")
        }
    }

    func configure_vertex_attr(
        _ vertexDescriptor: MTLVertexDescriptor,
        _ attr: MTLVertexAttribute,
        _ descriptor: PFDevice.VertexAttrDescriptor
    ) {
        let attribute_index = attr.attributeIndex
        let attr_info = vertexDescriptor.attributes[attribute_index]!

        let format: MTLVertexFormat =
            switch (descriptor.class, descriptor.attr_type, descriptor.size) {
            case (.int, .i8, 2): .char2
            case (.int, .i8, 3): .char3
            case (.int, .i8, 4): .char4
            case (.int, .u8, 2): .uchar2
            case (.int, .u8, 3): .uchar3
            case (.int, .u8, 4): .uchar4
            case (.floatNorm, .u8, 2): .uchar2Normalized
            case (.floatNorm, .u8, 3): .uchar3Normalized
            case (.floatNorm, .u8, 4): .uchar4Normalized
            case (.floatNorm, .i8, 2): .char2Normalized
            case (.floatNorm, .i8, 3): .char3Normalized
            case (.floatNorm, .i8, 4): .char4Normalized
            case (.int, .i16, 2): .short2
            case (.int, .i16, 3): .short3
            case (.int, .i16, 4): .short4
            case (.int, .u16, 2): .ushort2
            case (.int, .u16, 3): .ushort3
            case (.int, .u16, 4): .ushort4
            case (.floatNorm, .u16, 2): .ushort2Normalized
            case (.floatNorm, .u16, 3): .ushort3Normalized
            case (.floatNorm, .u16, 4): .ushort4Normalized
            case (.floatNorm, .i16, 2): .short2Normalized
            case (.floatNorm, .i16, 3): .short3Normalized
            case (.floatNorm, .i16, 4): .short4Normalized
            case (.float, .f32, 1): .float
            case (.float, .f32, 2): .float2
            case (.float, .f32, 3): .float3
            case (.float, .f32, 4): .float4
            case (.int, .i8, 1): .char
            case (.int, .u8, 1): .uchar
            case (.floatNorm, .i8, 1): .charNormalized
            case (.floatNorm, .u8, 1): .ucharNormalized
            case (.int, .i16, 1): .short
            case (.int, .i32, 1): .int
            case (.int, .u16, 1): .ushort
            case (.floatNorm, .u16, 1): .ushortNormalized
            case (.floatNorm, .i16, 1): .shortNormalized
            default:
                fatalError("Unsupported vertex class/type/size combination")
            }

        attr_info.format = format
        attr_info.offset = descriptor.offset

        let buffer_index = Int(descriptor.buffer_index) + Int(PFDevice.FIRST_VERTEX_BUFFER_INDEX)

        attr_info.bufferIndex = buffer_index

        // FIXME(pcwalton): Metal separates out per-buffer info from per-vertex info, while our
        // GL-like API does not. So we end up setting this state over and over again. Not great.
        let layout = vertexDescriptor.layouts[buffer_index]!
        if descriptor.divisor == 0 {
            layout.stepFunction = .perVertex
            layout.stepRate = 1
        } else {
            layout.stepFunction = .perInstance
            layout.stepRate = Int(descriptor.divisor)
        }

        layout.stride = descriptor.stride
    }

    func create_vertex_array() -> PFDevice.VertexArray {
        .init(
            descriptor: .init(),
            vertex_buffers: [],
            index_buffer: nil
        )
    }

    func get_vertex_attr(_ program: PFDevice.Program, _ name: String) -> MTLVertexAttribute? {
        // TODO(pcwalton): Cache the function?
        let attributes =
            switch program {
            case .raster(let rasterProgram):
                rasterProgram.vertex_shader.function.vertexAttributes!
            default:
                fatalError()
            }

        for attribute_index in 0..<attributes.count {
            let attribute = attributes[attribute_index]

            let this_name = attribute.name
            if this_name.hasPrefix("a") && this_name.dropFirst() == name {
                return attribute
            }
        }

        return nil
    }
}
