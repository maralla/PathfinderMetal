struct Paint1: Hashable {
  enum PaintContents: Hashable {
    case gradient(Gradient1)
    case pattern(Pattern1)
  }

  enum PaintCompositeOp: Hashable {
    case srcIn
    case destIn
  }

  struct PaintOverlay: Hashable {
    var compositeOp: PaintCompositeOp
    var contents: PaintContents

    var isOpaque: Bool {
      // An overlay is opaque depending on its contents and composite operation
      switch contents {
      case .gradient(let gradient):
        return gradient.isOpaque
      case .pattern(let pattern):
        return pattern.isOpaque
      }
    }
  }

  enum PaintFilter {
    case none
    case radialGradient(
      /// The line segment that connects the two circles.
      line: LineSegment,
      /// The radii of the two circles.
      radii: SIMD2<Float32>
    )
    case patternFilter(Pattern1.PatternFilter)
  }

  struct PaintColorTextureMetadata {
    /// The location of the paint.
    var location: SceneBuilder1.TextureLocation
    /// The scale for the page this paint is on.
    var page_scale: SIMD2<Float32>
    /// The transform to apply to screen coordinates to translate them into UVs.
    var transform: Transform
    /// The sampling mode for the texture.
    var sampling_flags: RenderCommand1.TextureSamplingFlags
    /// The filter to be applied to this paint.
    var filter: PaintFilter
    /// How the color texture is to be composited over the base color.
    var composite_op: PaintCompositeOp
    /// How much of a border there needs to be around the image.
    ///
    /// The border ensures clamp-to-edge yields the right result.
    var border: SIMD2<Int32>

    var tile_batch_texture: RenderCommand1.TileBatchTexture {
      .init(
        page: self.location.page,
        sampling_flags: self.sampling_flags,
        composite_op: self.composite_op
      )
    }
  }

  struct PaintMetadata {
    /// Metadata associated with the color texture, if applicable.
    var color_texture_metadata: PaintColorTextureMetadata?
    /// The base color that the color texture gets mixed into.
    var base_color: ColorU
    var blend_mode: Scene1.BlendMode
    /// True if this paint is fully opaque.
    var is_opaque: Bool

    var filter: Filter {
      guard let color_metadata = color_texture_metadata else { return .none }

      switch color_metadata.filter {
      case .none:
        return .none
      case .radialGradient(let line, let radii):
        let uv_rect = (color_metadata.location.rect.f32 * color_metadata.page_scale).contract(
          SIMD2<Float32>(0.0, color_metadata.page_scale.y * 0.5))
        return .radialGradient(line: line, radii: radii, uv_origin: uv_rect.origin)
      case .patternFilter(let pattern_filter):
        return .patternFilter(pattern_filter)
      }
    }

    var tile_batch_texture: RenderCommand1.TileBatchTexture? {
      self.color_texture_metadata?.tile_batch_texture
    }
  }

  struct PaintInfo {
    /// The render commands needed to prepare the textures.
    var render_commands: [RenderCommand1]
    /// The metadata for each paint.
    ///
    /// The indices of this vector are paint IDs.
    var paint_metadata: [PaintMetadata]
  }

  var baseColor: ColorU
  var overlay: PaintOverlay?

  init(baseColor: ColorU, overlay: PaintOverlay?) {
    self.baseColor = baseColor
    self.overlay = overlay
  }

  init(color: ColorU) {
    self.baseColor = color
    self.overlay = nil
  }

  init(pattern: Pattern1) {
    self.baseColor = .white
    self.overlay = .init(
      compositeOp: .srcIn,
      contents: .pattern(pattern)
    )
  }

  var isOpaque: Bool {
    // A paint is opaque if its base color is opaque and it has no overlay
    // or if its overlay is also opaque
    return baseColor.a == 255 && (overlay == nil || overlay!.isOpaque)
  }

  static var black: Paint1 {
    return .init(color: .black)
  }

  /// A convenience function to create a transparent paint with all channels set to zero.
  static var transparent_black: Paint1 {
    return .init(color: .transparent_black)
  }

  var pattern: Pattern1? {
    get {
      guard let overlay = overlay else { return nil }

      if case .pattern(let pattern) = overlay.contents {
        return pattern
      }

      return nil
    }

    set {
      if let value = newValue, overlay != nil {
        overlay?.contents = .pattern(value)
      }
    }
  }

  mutating func apply_transform(_ transform: Transform) {
    if transform.isIdentity {
      return
    }

    if var overlay = overlay {
      switch overlay.contents {
      case .gradient(var gradient):
        gradient.apply_transform(transform)
        overlay.contents = .gradient(gradient)
      case .pattern(var pattern):
        pattern.apply_transform(transform)
        overlay.contents = .pattern(pattern)
      }

      self.overlay = overlay
    }
  }
}

struct Palette1 {
  struct RenderTargetMetadata {
    /// The location of the render target.
    var location: SceneBuilder1.TextureLocation
  }

  struct GradientTileBuilder {
    var tiles: [GradientTile] = []

    func create_render_commands(render_commands: inout [RenderCommand1]) {
      for tile in tiles {
        render_commands.append(
          .uploadTexelData(
            texels: tile.texels,
            location: .init(
              page: tile.page,
              rect: PFRect(
                origin: .zero, size: SIMD2<Int32>(repeating: Int32(Gradient1.GRADIENT_TILE_LENGTH)))
            )
          ))
      }
    }
  }

  struct GradientTile {
    var texels: [ColorU]
    var page: UInt32
    var next_index: UInt32
  }

  struct ImageTexelInfo {
    var location: SceneBuilder1.TextureLocation
    var texels: [ColorU]
  }

  struct PaintLocationsInfo {
    var paint_metadata: [Paint1.PaintMetadata]
    var gradient_tile_builder: GradientTileBuilder
    var image_texel_info: [ImageTexelInfo]
    var used_image_hashes: Set<UInt64>
  }

  var paints: [Paint1]
  var render_targets: [Scene1.RenderTarget]
  var cache: [Paint1: UInt16]
  var scene_id: UInt32
}

extension Palette1.GradientTileBuilder {
  mutating func allocate(
    allocator: inout TextureAllocator1,
    transient_paint_locations: inout [SceneBuilder1.TextureLocation],
    gradient: Gradient1
  ) -> SceneBuilder1.TextureLocation {
    if self.tiles.isEmpty || self.tiles.last!.next_index == Gradient1.GRADIENT_TILE_LENGTH {
      let size = SIMD2<Int32>(repeating: Int32(Gradient1.GRADIENT_TILE_LENGTH))
      let area = Int(size.x) * Int(size.y)

      let page_location = allocator.allocate(requested_size: size, mode: .ownPage)
      transient_paint_locations.append(page_location)

      self.tiles.append(
        .init(
          texels: Array(repeating: .black, count: area),
          page: page_location.page,
          next_index: 0
        ))
    }

    var data = self.tiles.last!
    let location = SceneBuilder1.TextureLocation(
      page: data.page,
      rect: PFRect<Int32>(
        origin: .init(0, Int32(data.next_index)),
        size: .init(Int32(Gradient1.GRADIENT_TILE_LENGTH), 1)
      )
    )

    data.next_index += 1

    // FIXME(pcwalton): Paint transparent if gradient line has zero size, per spec.
    // TODO(pcwalton): Optimize this:
    // 1. Calculate ∇t up front and use differencing in the inner loop.
    // 2. Go four pixels at a time with SIMD.
    let first_address = Int(location.rect.originY) * Int(Gradient1.GRADIENT_TILE_LENGTH)
    for x in 0..<(Gradient1.GRADIENT_TILE_LENGTH) {
      var t = (Float32(x) + 0.5) / Float32(Gradient1.GRADIENT_TILE_LENGTH)
      data.texels[first_address + Int(x)] = gradient.sample(t: t)
    }

    return location
  }
}

extension Palette1 {
  init(_ scene_id: UInt32) {
    self.paints = []
    self.render_targets = []
    self.cache = [:]
    self.scene_id = scene_id
  }

  mutating func push_paint(_ paint: Paint1) -> UInt16 {
    if let paint_id = cache[paint] {
      return paint_id
    }

    let paint_id = UInt16(paints.count)
    cache[paint] = paint_id
    paints.append(paint)
    return paint_id
  }

  mutating func push_render_target(_ render_target: Scene1.RenderTarget) -> Scene1.RenderTargetId {
    let id = UInt32(render_targets.count)
    render_targets.append(render_target)
    return .init(scene: scene_id, render_target: id)
  }

  func build_paint_info(
    texture_manager: inout SceneBuilder1.PaintTextureManager, render_transform: Transform
  ) -> Paint1.PaintInfo {
    // Assign render target locations.
    var transient_paint_locations: [SceneBuilder1.TextureLocation] = []
    let render_target_metadata = self.assign_render_target_locations(
      texture_manager: &texture_manager,
      transient_paint_locations: &transient_paint_locations
    )

    let info = self.assign_paint_locations(
      render_target_metadata, &texture_manager, &transient_paint_locations)
    var paint_metadata = info.paint_metadata
    let gradient_tile_builder = info.gradient_tile_builder
    let image_texel_info = info.image_texel_info
    let used_image_hashes = info.used_image_hashes

    // Calculate texture transforms.
    self.calculate_texture_transforms(&paint_metadata, texture_manager, render_transform)

    // Create texture metadata.
    let texture_metadata = self.create_texture_metadata(paint_metadata)

    var render_commands = [RenderCommand1.uploadTextureMetadata(texture_metadata)]

    // Allocate textures.
    self.allocate_textures(&render_commands, &texture_manager)

    // Create render commands.
    self.create_render_commands(
      &render_commands,
      render_target_metadata,
      gradient_tile_builder,
      image_texel_info)

    // Free transient locations and unused images, now that they're no longer needed.
    self.free_transient_locations(&texture_manager, transient_paint_locations)
    self.free_unused_images(&texture_manager, used_image_hashes)

    return .init(render_commands: render_commands, paint_metadata: paint_metadata)
  }

  func assign_render_target_locations(
    texture_manager: inout SceneBuilder1.PaintTextureManager,
    transient_paint_locations: inout [SceneBuilder1.TextureLocation]
  ) -> [RenderTargetMetadata] {
    var render_target_metadata: [RenderTargetMetadata] = []

    for render_target in render_targets {
      let location = texture_manager.allocator.allocate_image(requested_size: render_target.size)

      render_target_metadata.append(RenderTargetMetadata(location: location))
      transient_paint_locations.append(location)
    }

    return render_target_metadata
  }

  func assign_paint_locations(
    _ render_target_metadata: [RenderTargetMetadata],
    _ texture_manager: inout SceneBuilder1.PaintTextureManager,
    _ transient_paint_locations: inout [SceneBuilder1.TextureLocation]
  ) -> Palette1.PaintLocationsInfo {
    var paint_metadata: [Paint1.PaintMetadata] = []
    var gradient_tile_builder = GradientTileBuilder()
    var image_texel_info: [Palette1.ImageTexelInfo] = []
    var used_image_hashes = Set<UInt64>()

    for paint in paints {
      var color_texture_metadata: Paint1.PaintColorTextureMetadata? = nil
      if let overlay = paint.overlay {
        switch overlay.contents {
        case .gradient(let gradient):
          var sampling_flags = RenderCommand1.TextureSamplingFlags()
          switch gradient.wrap {
          case .repeat:
            sampling_flags.insert(.REPEAT_U)
          case .clamp:
            break
          }

          // FIXME(pcwalton): The gradient size might not be big enough. Detect
          // this.
          let location = gradient_tile_builder.allocate(
            allocator: &texture_manager.allocator,
            transient_paint_locations: &transient_paint_locations,
            gradient: gradient)
          let filter: Paint1.PaintFilter
          switch gradient.geometry {
          case .linear:
            filter = .none
          case .radial(let line, let radii, transform: _):
            filter = .radialGradient(line: line, radii: radii)
          }

          color_texture_metadata = .init(
            location: location,
            page_scale: texture_manager.allocator.page_scale(location.page),
            transform: Transform(),
            sampling_flags: sampling_flags,
            filter: filter,
            composite_op: overlay.compositeOp,
            border: .zero
          )
        case .pattern(let pattern):
          let border = SIMD2<Int32>(
            pattern.repeatX ? 0 : 1,
            pattern.repeatY ? 0 : 1)

          let location: SceneBuilder1.TextureLocation

          switch pattern.source {
          case .renderTarget(id: let render_target_id, _):
            let index = Int(render_target_id.render_target)
            location = render_target_metadata[index].location
          case .image(let image):
            // TODO(pcwalton): We should be able to use tile cleverness to
            // repeat inside the atlas in some cases.
            let image_hash = UInt64(image.hashValue)

            if let cached_location = texture_manager.cached_images[image_hash] {
              location = cached_location
              used_image_hashes.insert(image_hash)
            } else {
              // Leave a pixel of border on the side.
              location = texture_manager.allocator.allocate(
                requested_size: image.size &+ border &* 2, mode: .ownPage)
              texture_manager.cached_images[image_hash] = location
            }

            image_texel_info.append(
              .init(
                location: .init(page: location.page, rect: location.rect.contract(border)),
                texels: Array(image.pixels)
              ))
          }

          var sampling_flags = RenderCommand1.TextureSamplingFlags()
          if pattern.repeatX {
            sampling_flags.insert(.REPEAT_U)
          }
          if pattern.repeatY {
            sampling_flags.insert(.REPEAT_V)
          }
          if !pattern.smoothingEnabled {
            sampling_flags.insert([.NEAREST_MIN, .NEAREST_MAG])
          }

          var filter = Paint1.PaintFilter.none
          if let pattern_filter = pattern.filter {
            filter = .patternFilter(pattern_filter)
          }

          color_texture_metadata = .init(
            location: location,
            page_scale: texture_manager.allocator.page_scale(location.page),
            transform: Transform(translation: SIMD2<Float32>(border)),
            sampling_flags: sampling_flags,
            filter: filter,
            composite_op: overlay.compositeOp,
            border: border
          )
        }
      }

      paint_metadata.append(
        .init(
          color_texture_metadata: color_texture_metadata,
          base_color: paint.baseColor,
          blend_mode: .srcOver,
          is_opaque: paint.isOpaque
        ))
    }

    return .init(
      paint_metadata: paint_metadata,
      gradient_tile_builder: gradient_tile_builder,
      image_texel_info: image_texel_info,
      used_image_hashes: used_image_hashes
    )
  }

  func calculate_texture_transforms(
    _ paint_metadata: inout [Paint1.PaintMetadata],
    _ texture_manager: SceneBuilder1.PaintTextureManager,
    _ render_transform: Transform
  ) {

    for (i, (paint, metadata)) in zip(self.paints, paint_metadata).enumerated() {
      guard var color_texture_metadata = metadata.color_texture_metadata else { continue }

      let texture_scale = texture_manager.allocator.page_scale(color_texture_metadata.location.page)
      let texture_rect = color_texture_metadata.location.rect

      switch paint.overlay!.contents {
      case .gradient(let gradient):
        switch gradient.geometry {
        case .linear(let gradient_line):
          // Project gradient line onto (0.0-1.0, v0).
          let v0 = texture_rect.f32.center.y * texture_scale.y
          let dp = gradient_line.vector

          let m0 =
            SIMD4<Float32>(dp.x, dp.y, dp.x, dp.y)
            / SIMD4<Float32>(repeating: gradient_line.square_length)
          let m13 = SIMD2<Float32>(m0.z, m0.w) * -gradient_line.from

          color_texture_metadata.transform = Transform(
            m11: m0.x, m12: m0.y, m13: m13.x + m13.y, m21: 0.0, m22: 0.0, m23: v0)
        case .radial(line: _, radii: _, let transform):
          color_texture_metadata.transform = transform.inverse()
        }
      case .pattern(let pattern):
        switch pattern.source {
        case .image(_):
          let texture_origin_uv = (texture_rect.f32 * texture_scale).origin

          color_texture_metadata.transform =
            Transform(scale: texture_scale).translate(texture_origin_uv)
            * pattern.transform.inverse()
        case .renderTarget:
          // FIXME(pcwalton): Only do this in GL, not Metal!
          let texture_origin_uv = (texture_rect.f32 * texture_scale).lowerLeft

          color_texture_metadata.transform =
            Transform(translation: texture_origin_uv)
            * Transform(scale: texture_scale * SIMD2<Float32>(1.0, -1.0))
            * pattern.transform.inverse()
        }
      }

      color_texture_metadata.transform = color_texture_metadata.transform * render_transform

      var metadata = metadata
      metadata.color_texture_metadata = color_texture_metadata
      paint_metadata[i] = metadata
    }
  }

  func create_texture_metadata(_ paint_metadata: [Paint1.PaintMetadata]) -> [RenderCommand1
    .TextureMetadataEntry]
  {
    paint_metadata.map({ paint_metadata in
      let transform = paint_metadata.color_texture_metadata?.transform ?? .init()

      var combineMode = RenderCommand1.ColorCombineMode.none
      if paint_metadata.color_texture_metadata != nil {
        combineMode = .srcIn
      }

      return .init(
        color_0_transform: transform,
        color_0_combine_mode: combineMode,
        base_color: paint_metadata.base_color,
        filter: paint_metadata.filter,
        blend_mode: paint_metadata.blend_mode
      )
    })
  }

  func allocate_textures(
    _ render_commands: inout [RenderCommand1],
    _ texture_manager: inout SceneBuilder1.PaintTextureManager
  ) {
    for page_id in texture_manager.allocator.page_ids() {
      let page_size = texture_manager.allocator.page_size(page_id)

      let descriptor = RenderCommand1.TexturePageDescriptor(size: page_size)

      if texture_manager.allocator.page_is_new(page_id) {
        render_commands.append(.allocateTexturePage(page_id: page_id, descriptor: descriptor))
      }
    }

    texture_manager.allocator.mark_all_pages_as_allocated()
  }

  func create_render_commands(
    _ render_commands: inout [RenderCommand1],
    _ render_target_metadata: [Palette1.RenderTargetMetadata],
    _ gradient_tile_builder: Palette1.GradientTileBuilder,
    _ image_texel_info: [Palette1.ImageTexelInfo]
  ) {
    for (index, metadata) in render_target_metadata.enumerated() {
      let id = Scene1.RenderTargetId(scene: self.scene_id, render_target: UInt32(index))
      render_commands.append(.declareRenderTarget(id: id, location: metadata.location))
    }

    gradient_tile_builder.create_render_commands(render_commands: &render_commands)
    for image_texel_info in image_texel_info {
      render_commands.append(
        .uploadTexelData(
          texels: image_texel_info.texels,
          location: image_texel_info.location
        ))
    }
  }

  func free_transient_locations(
    _ texture_manager: inout SceneBuilder1.PaintTextureManager,
    _ transient_paint_locations: [SceneBuilder1.TextureLocation]
  ) {
    for location in transient_paint_locations {
      texture_manager.allocator.free(location: location)
    }
  }

  func free_unused_images(
    _ texture_manager: inout SceneBuilder1.PaintTextureManager,
    _ used_image_hashes: Set<UInt64>
  ) {
    var toBeRemoved: [UInt64] = []
    for (imageHash, location) in texture_manager.cached_images {
      let keep = used_image_hashes.contains(imageHash)
      if !keep {
        texture_manager.allocator.free(location: location)
        toBeRemoved.append(imageHash)
      }
    }

    for item in toBeRemoved {
      texture_manager.cached_images.removeValue(forKey: item)
    }
  }
}
