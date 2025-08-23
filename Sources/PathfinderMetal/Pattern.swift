struct Pattern1 {
  enum BlurDirection {
    /// The horizontal axis.
    case x
    /// The vertical axis.
    case y
  }

  enum PatternFilter {
    case blur(
      /// The axis of the blur: horizontal or vertical.
      direction: BlurDirection,
      /// Half the blur radius.
      sigma: Float32
    )

    /// A color matrix multiplication.
    ///
    /// The matrix is stored in 5 columns of `F32x4`. See the `feColorMatrix` element in the SVG
    /// specification.
    case colorMatrix(ColorMatrix)
  }

  struct Image: Hashable {
    var size: SIMD2<Int32>
    var pixels: [ColorU]
    var pixels_hash: UInt64
    var isOpaque: Bool

    func hash(into hasher: inout Hasher) {
      size.hash(into: &hasher)
      pixels_hash.hash(into: &hasher)
      isOpaque.hash(into: &hasher)
    }
  }

  enum PatternSource: Hashable {
    /// A image whose pixels are stored in CPU memory.
    case image(Image)
    /// Previously-rendered vector content.
    ///
    /// This value allows you to render content and then later use that content as a pattern.
    case renderTarget(
      /// The ID of the render target, including the ID of the scene it came from.
      id: Scene1.RenderTargetId,
      /// The device pixel size of the render target.
      size: SIMD2<Int32>
    )
  }

  struct PatternFlags: OptionSet, Hashable {
    let rawValue: UInt8

    /// If set, the pattern repeats in the X direction. If unset, the base color is used.
    static let REPEAT_X = PatternFlags(rawValue: 0x01)
    /// If set, the pattern repeats in the Y direction. If unset, the base color is used.
    static let REPEAT_Y = PatternFlags(rawValue: 0x02)
    /// If set, nearest-neighbor interpolation is used when compositing this pattern (i.e. the
    /// image will be pixelated). If unset, bilinear interpolation is used when compositing
    /// this pattern (i.e. the image will be smooth).
    static let NO_SMOOTHING = PatternFlags(rawValue: 0x04)
  }

  var source: PatternSource
  var transform: Transform
  var filter: PatternFilter?
  var flags: PatternFlags
}

extension Pattern1: Hashable {
  static func == (lhs: Pattern1, rhs: Pattern1) -> Bool {
    lhs.hashValue == rhs.hashValue
  }

  func hash(into hasher: inout Hasher) {
    source.hash(into: &hasher)
    transform.hash(into: &hasher)
    flags.hash(into: &hasher)
  }
}

extension Pattern1 {
  init(id: Scene1.RenderTargetId, size: SIMD2<Int32>) {
    self.init(source: .renderTarget(id: id, size: size))
  }

  init(source: PatternSource) {
    self.source = source
    self.transform = .init()
    self.filter = nil
    self.flags = PatternFlags()
  }

  var isOpaque: Bool {
    source.isOpaque
  }

  var repeatX: Bool {
    return flags.contains(.REPEAT_X)
  }

  var repeatY: Bool {
    return flags.contains(.REPEAT_Y)
  }

  var smoothingEnabled: Bool {
    get {
      return !flags.contains(.NO_SMOOTHING)
    }

    set {
      if newValue {
        flags.subtract(.NO_SMOOTHING)
      } else {
        flags.formUnion(.NO_SMOOTHING)
      }
    }
  }

  mutating func apply_transform(_ transform: Transform) {
    self.transform = transform * self.transform
  }
}

extension Pattern1.PatternSource {
  var isOpaque: Bool {
    switch self {
    case .image(let image):
      return image.isOpaque
    case .renderTarget:
      return false
    }
  }
}

enum Filter {
  /// No special filter.
  case none

  /// Converts a linear gradient to a radial one.
  case radialGradient(
    /// The line that the circles lie along.
    line: LineSegment,
    /// The radii of the circles at the two endpoints.
    radii: SIMD2<Float32>,
    /// The origin of the linearized gradient in the texture.
    uv_origin: SIMD2<Float32>
  )

  /// One of the `PatternFilter` filters.
  case patternFilter(Pattern1.PatternFilter)
}
