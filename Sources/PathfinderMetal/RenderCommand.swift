import Foundation

enum RenderCommand {
    static let CURVE_IS_QUADRATIC: UInt32 = 0x8000_0000
    static let CURVE_IS_CUBIC: UInt32 = 0x4000_0000

    struct TexturePageDescriptor {
        var size: SIMD2<Int32>
    }

    enum ColorCombineMode {
        static let COMBINER_CTRL_COLOR_COMBINE_SRC_IN: Int32 = 0x1
        static let COMBINER_CTRL_COLOR_COMBINE_DEST_IN: Int32 = 0x2

        case none
        case srcIn
        case destIn

        func to_composite_ctrl() -> Int32 {
            switch self {
            case .none:
                return 0
            case .srcIn:
                return Self.COMBINER_CTRL_COLOR_COMBINE_SRC_IN
            case .destIn:
                return Self.COMBINER_CTRL_COLOR_COMBINE_DEST_IN
            }
        }
    }

    struct TextureMetadataEntry {
        var color_0_transform: Transform
        var color_0_combine_mode: ColorCombineMode
        var base_color: ColorU
        var filter: Filter
        var blend_mode: Scene.BlendMode
    }

    struct SegmentIndicesD3D11 {
        var first_point_index: UInt32
        var flags: UInt32
    }

    struct SegmentsD3D11 {
        var points: [SIMD2<Float32>] = []
        var indices: [SegmentIndicesD3D11] = []
    }

    struct BackdropInfoD3D11 {
        var initial_backdrop: Int32
        // Column number, where 0 is the leftmost column in the tile rect.
        var tile_x_offset: Int32
        var path_index: UInt32
    }

    struct PropagateMetadataD3D11 {
        var tile_rect: PFRect<Int32>
        var tile_offset: UInt32
        var path_index: UInt32
        var z_write: UInt32
        // This will generally not refer to the same batch as `path_index`.
        var clip_path_index: UInt32
        var backdrop_offset: UInt32
        var pad0: UInt32
        var pad1: UInt32
        var pad2: UInt32
    }

    struct DiceMetadataD3D11 {
        var global_path_id: UInt32
        var first_global_segment_index: UInt32
        var first_batch_segment_index: UInt32
        var pad: UInt32
    }

    struct MicrolineD3D11 {
        let from_x_px: Int16
        let from_y_px: Int16
        let to_x_px: Int16
        let to_y_px: Int16
        let from_x_subpx: UInt8
        let from_y_subpx: UInt8
        let to_x_subpx: UInt8
        let to_y_subpx: UInt8
        let path_index: UInt32
    }

    struct TilePathInfoD3D11 {
        var tile_min_x: Int16
        var tile_min_y: Int16
        var tile_max_x: Int16
        var tile_max_y: Int16
        var first_tile_index: UInt32
        // Must match the order in `TileD3D11`.
        var color: UInt16
        var ctrl: UInt8
        var backdrop: Int8
    }

    struct PrepareTilesInfoD3D11 {
        /// Initial backdrop values for each tile column, packed together.
        var backdrops: [BackdropInfoD3D11]

        /// Mapping from path index to metadata needed to compute propagation on GPU.
        ///
        /// This contains indices into the `tiles` vector.
        var propagate_metadata: [PropagateMetadataD3D11]

        /// Metadata about each path that will be diced (flattened).
        var dice_metadata: [DiceMetadataD3D11]

        /// Sparse information about all the allocated tiles.
        var tile_path_info: [TilePathInfoD3D11]

        /// A transform to apply to the segments.
        var transform: Transform
    }

    enum PathSource {
        case draw
        case clip
    }

    struct Clip {
        var dest_tile_id: UInt32
        var dest_backdrop: Int32
        var src_tile_id: UInt32
        var src_backdrop: Int32
    }

    struct ClippedPathInfo {
        /// The ID of the batch containing the clips.
        var clip_batch_id: UInt32

        /// The number of paths that have clips.
        var clipped_path_count: UInt32

        /// The maximum number of clipped tiles.
        ///
        /// This is used to allocate vertex buffers.
        var max_clipped_tile_count: UInt32

        /// The actual clips, if calculated on CPU.
        var clips: [Clip]?
    }

    struct TextureSamplingFlags: OptionSet {
        let rawValue: UInt8

        static let REPEAT_U = TextureSamplingFlags(rawValue: 0x01)
        static let REPEAT_V = TextureSamplingFlags(rawValue: 0x02)
        static let NEAREST_MIN = TextureSamplingFlags(rawValue: 0x04)
        static let NEAREST_MAG = TextureSamplingFlags(rawValue: 0x08)
    }

    struct TileBatchTexture: Equatable {
        var page: UInt32
        var sampling_flags: TextureSamplingFlags
        var composite_op: Paint.PaintCompositeOp
    }

    struct TileBatchDataD3D11 {
        /// The ID of this batch.
        ///
        /// The renderer should not assume that these values are consecutive.
        var batch_id: UInt32
        /// The number of paths in this batch.
        var path_count: UInt32 = 0
        /// The number of tiles in this batch.
        var tile_count: UInt32 = 0
        /// The total number of segments in this batch.
        var segment_count: UInt32 = 0
        /// Information needed to prepare the tiles.
        var prepare_info: PrepareTilesInfoD3D11
        /// Where the paths come from (draw or clip).
        var path_source: PathSource
        /// Information about clips applied to paths, if any of the paths have clips.
        var clipped_path_info: ClippedPathInfo? = nil

        init(_ batch_id: UInt32, _ mode: SceneBuilder.PrepareMode, _ path_source: PathSource) {
            self.batch_id = batch_id
            self.path_source = path_source

            self.prepare_info = .init(
                backdrops: [],
                propagate_metadata: [],
                dice_metadata: [],
                tile_path_info: [],
                transform: mode.transform
            )
        }
    }

    struct DrawTileBatchD3D11 {
        /// Data for the tile batch.
        var tile_batch_data: TileBatchDataD3D11
        /// The color texture to use.
        var color_texture: TileBatchTexture?
    }

    // Starts rendering a frame.
    case start(
        /// The number of paths that will be rendered.
        path_count: Int,

        /// A bounding quad for the scene.
        bounding_quad: [SIMD4<Float32>],

        /// Whether the framebuffer we're rendering to must be readable.
        ///
        /// This is needed if a path that renders directly to the output framebuffer (i.e. not to a
        /// render target) uses one of the more exotic blend modes.
        needs_readable_framebuffer: Bool
    )

    // Allocates a texture page.
    case allocateTexturePage(page_id: UInt32, descriptor: TexturePageDescriptor)

    // Uploads data to a texture page.
    case uploadTexelData(texels: [ColorU], location: SceneBuilder.TextureLocation)

    // Associates a render target with a texture page.
    //
    // TODO(pcwalton): Add a rect to this so we can render to subrects of a page.
    case declareRenderTarget(id: Scene.RenderTargetId, location: SceneBuilder.TextureLocation)

    // Upload texture metadata.
    case uploadTextureMetadata([TextureMetadataEntry])

    /// Upload a scene to GPU.
    ///
    /// This will only be sent if dicing and binning is done on GPU.
    case uploadSceneD3D11(
        draw_segments: SegmentsD3D11,
        clip_segments: SegmentsD3D11
    )

    // Pushes a render target onto the stack. Draw commands go to the render target on top of the
    // stack.
    case pushRenderTarget(Scene.RenderTargetId)

    // Pops a render target from the stack.
    case popRenderTarget

    // Computes backdrops for tiles, prepares any Z-buffers, and performs clipping.
    case prepareClipTilesD3D11(TileBatchDataD3D11)

    // Draws a batch of tiles to the render target on top of the stack.
    case drawTilesD3D11(DrawTileBatchD3D11)

    // Presents a rendered frame.
    case finish(cpu_build_time: DispatchTimeInterval)
}

extension RenderCommand.SegmentsD3D11 {
    mutating func add_path(outline: Outline) -> Range<UInt32> {
        let first_segment_index = self.indices.count

        for contour in outline.contours {
            let point_count = contour.points.count
            self.points.reserveCapacity(point_count)

            for point_index in 0..<point_count {
                if contour.flags_of(point_index).intersection([.controlPoint0, .controlPoint1]).isEmpty {
                    var flags: UInt32 = 0
                    if point_index + 1 < point_count
                        && contour.flags_of(point_index + 1).contains(.controlPoint0)
                    {
                        if point_index + 2 < point_count
                            && contour.flags_of(point_index + 2).contains(.controlPoint1)
                        {
                            flags = RenderCommand.CURVE_IS_CUBIC
                        } else {
                            flags = RenderCommand.CURVE_IS_QUADRATIC
                        }
                    }

                    self.indices.append(
                        .init(
                            first_point_index: UInt32(self.points.count),
                            flags: flags
                        )
                    )
                }

                self.points.append(contour.position_of(point_index))
            }

            self.points.append(contour.position_of(0))
        }

        let last_segment_index = self.indices.count
        return UInt32(first_segment_index)..<UInt32(last_segment_index)
    }
}

extension RenderCommand.TileBatchDataD3D11 {
    func init_backdrops(
        _ backdrops: inout [RenderCommand.BackdropInfoD3D11],
        _ path_index: UInt32,
        _ tile_rect: PFRect<Int32>
    ) {
        for tile_x_offset in 0..<tile_rect.width {
            backdrops.append(
                .init(initial_backdrop: 0, tile_x_offset: tile_x_offset, path_index: path_index)
            )
        }
    }

    mutating func push(
        _ path: SceneBuilder.BuiltPath,
        _ global_path_id: UInt32,
        _ batch_clip_path_id: SceneBuilder.GlobalPathId?,
        _ z_write: Bool,
        _ sink: SceneBuilder.SceneSink
    ) -> UInt32 {
        let batch_path_index = self.path_count
        self.path_count += 1

        var clip_path_index = UInt32.max
        if let batch_clip_path_id {
            clip_path_index = batch_clip_path_id.path_index
        }

        self.prepare_info.propagate_metadata.append(
            .init(
                tile_rect: path.tile_bounds,
                tile_offset: self.tile_count,
                path_index: batch_path_index,
                z_write: UInt32(z_write ? 1 : 0),
                clip_path_index: clip_path_index,
                backdrop_offset: UInt32(self.prepare_info.backdrops.count),
                pad0: 0,
                pad1: 0,
                pad2: 0
            )
        )

        init_backdrops(&self.prepare_info.backdrops, batch_path_index, path.tile_bounds)

        // Add tiles.
        let last_scene = sink.last_scene!

        let segment_ranges: [Range<UInt32>]
        switch self.path_source {
        case .draw:
            segment_ranges = last_scene.draw_segment_ranges
        case .clip:
            segment_ranges = last_scene.clip_segment_ranges
        }

        let segment_range = segment_ranges[Int(global_path_id)]

        self.prepare_info.dice_metadata.append(
            .init(
                global_path_id: global_path_id,
                first_global_segment_index: segment_range.lowerBound,
                first_batch_segment_index: self.segment_count,
                pad: 0
            )
        )

        self.prepare_info.tile_path_info.append(
            .init(
                tile_min_x: Int16(path.tile_bounds.minX),
                tile_min_y: Int16(path.tile_bounds.minY),
                tile_max_x: Int16(path.tile_bounds.maxX),
                tile_max_y: Int16(path.tile_bounds.maxY),
                first_tile_index: self.tile_count,
                color: path.paint_id,
                ctrl: path.ctrl_byte,
                backdrop: 0
            )
        )

        self.tile_count += UInt32(path.tile_bounds.area)
        self.segment_count += segment_range.upperBound - segment_range.lowerBound

        // Handle clip.

        guard let batch_clip_path_id else { return batch_path_index }
        let clip_batch_id = batch_clip_path_id.batch_id

        if self.clipped_path_info == nil {
            self.clipped_path_info = .init(
                clip_batch_id: clip_batch_id,
                clipped_path_count: 0,
                max_clipped_tile_count: 0,
                clips: nil
            )
        }

        var clipped_path_info = self.clipped_path_info!
        clipped_path_info.clipped_path_count += 1
        clipped_path_info.max_clipped_tile_count += UInt32(path.tile_bounds.area)
        self.clipped_path_info = clipped_path_info

        return batch_path_index
    }
}
