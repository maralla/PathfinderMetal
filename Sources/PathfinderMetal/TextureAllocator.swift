struct TextureAllocator1 {
  indirect enum TreeNode {
    case emptyLeaf
    case fullLeaf
    // Top left, top right, bottom left, and bottom right, in that order.
    case parent(TreeNode, TreeNode, TreeNode, TreeNode)
  }

  struct TextureAtlasAllocator {
    var root: TreeNode
    var size: UInt32
  }

  enum TexturePageAllocator {
    // An atlas allocated with our quadtree allocator.
    case atlas(TextureAtlasAllocator)
    // A single image.
    case image(size: SIMD2<Int32>)
  }

  struct TexturePage {
    var allocator: TexturePageAllocator
    var is_new: Bool
  }

  enum AllocationMode {
    case atlas
    case ownPage
  }

  struct TexturePageIter: Sequence, IteratorProtocol {
    typealias Element = UInt32

    let allocator: TextureAllocator1
    var nextIndex: Int

    mutating func next() -> UInt32? {
      var next_id: UInt32? = nil

      if nextIndex < allocator.pages.count {
        next_id = UInt32(nextIndex)
      }

      while true {
        nextIndex += 1

        if nextIndex >= allocator.pages.count || allocator.pages[nextIndex] != nil {
          break
        }
      }

      return next_id
    }
  }

  static let ATLAS_TEXTURE_LENGTH: UInt32 = 1024

  var pages: [TexturePage?]
}

extension TextureAllocator1.TreeNode {
  mutating func allocate(this_origin: SIMD2<Int32>, this_size: UInt32, requested_size: UInt32)
    -> PFRect<Int32>?
  {
    if case .fullLeaf = self {
      // No room here.
      return nil
    }

    if this_size < requested_size {
      // Doesn't fit.
      return nil
    }

    // Allocate here or split, as necessary.
    if case .emptyLeaf = self {
      // Do we have a perfect fit?
      if this_size == requested_size {
        self = .fullLeaf
        return .init(origin: this_origin, size: .init(repeating: Int32(this_size)))
      }

      // Split.
      self = .parent(.emptyLeaf, .emptyLeaf, .emptyLeaf, .emptyLeaf)
    }

    // Recurse into children.
    switch self {
    case .parent(var k0, var k1, var k2, var k3):
      let kid_size = this_size / 2

      if let origin = k0.allocate(
        this_origin: this_origin, this_size: kid_size, requested_size: requested_size)
      {
        self = .parent(k0, k1, k2, k3)
        return origin
      }

      if let origin = k1.allocate(
        this_origin: this_origin &+ SIMD2<Int32>(Int32(kid_size), 0),
        this_size: kid_size,
        requested_size: requested_size)
      {
        self = .parent(k0, k1, k2, k3)
        return origin
      }

      if let origin = k2.allocate(
        this_origin: this_origin &+ SIMD2<Int32>(0, Int32(kid_size)),
        this_size: kid_size,
        requested_size: requested_size)
      {
        self = .parent(k0, k1, k2, k3)
        return origin
      }

      if let origin = k3.allocate(
        this_origin: this_origin &+ SIMD2<Int32>(repeating: Int32(kid_size)),
        this_size: kid_size,
        requested_size: requested_size)
      {
        self = .parent(k0, k1, k2, k3)
        return origin
      }

      self = .parent(k0, k1, k2, k3)
      self.merge_if_necessary()
      return nil
    case .emptyLeaf, .fullLeaf: fatalError()
    }
  }

  mutating func free(
    _ this_origin: SIMD2<Int32>, _ this_size: UInt32, _ requested_origin: SIMD2<Int32>,
    _ requested_size: UInt32
  ) {
    if this_size <= requested_size {
      if this_size == requested_size && this_origin == requested_origin {
        self = .emptyLeaf
      }

      return
    }

    let child_size = this_size / 2
    let this_center = this_origin &+ SIMD2<Int32>(repeating: Int32(child_size))

    let child_index: Int
    var child_origin = this_origin

    if requested_origin.y < this_center.y {
      if requested_origin.x < this_center.x {
        child_index = 0
      } else {
        child_index = 1
        child_origin = child_origin &+ SIMD2<Int32>(Int32(child_size), 0)
      }
    } else {
      if requested_origin.x < this_center.x {
        child_index = 2
        child_origin = child_origin &+ SIMD2<Int32>(0, Int32(child_size))
      } else {
        child_index = 3
        child_origin = this_center
      }
    }

    switch self {
    case .parent(var k1, var k2, var k3, var k4):
      switch child_index {
      case 0:
        k1.free(child_origin, child_size, requested_origin, requested_size)
      case 1:
        k2.free(child_origin, child_size, requested_origin, requested_size)
      case 2:
        k3.free(child_origin, child_size, requested_origin, requested_size)
      case 3:
        k4.free(child_origin, child_size, requested_origin, requested_size)
      default:
        break
      }

      self = .parent(k1, k2, k3, k4)
      self.merge_if_necessary()
    case .emptyLeaf, .fullLeaf: fatalError()
    }
  }

  mutating func merge_if_necessary() {
    if case .parent(var k1, var k2, var k3, var k4) = self {
      if case .emptyLeaf = k1,
        case .emptyLeaf = k2,
        case .emptyLeaf = k3,
        case .emptyLeaf = k4
      {
        self = .emptyLeaf
      }
    }
  }
}

extension TextureAllocator1.TextureAtlasAllocator {
  init() {
    self.init(length: TextureAllocator1.ATLAS_TEXTURE_LENGTH)
  }

  init(length: UInt32) {
    root = .emptyLeaf
    size = length
  }

  var isEmpty: Bool {
    switch self.root {
    case .emptyLeaf: true
    default: false
    }
  }

  mutating func allocate(requested_size: SIMD2<Int32>) -> PFRect<Int32>? {
    let length = UInt32(max(requested_size.x, requested_size.y)).nextPowerOfTwo
    return self.root.allocate(this_origin: .zero, this_size: self.size, requested_size: length)
  }

  mutating func free(rect: PFRect<Int32>) {
    let requested_length = UInt32(rect.width)
    self.root.free(.zero, size, rect.origin, requested_length)
  }
}

extension TextureAllocator1 {
  init() {
    self.pages = []
  }

  mutating func allocate(requested_size: SIMD2<Int32>, mode: AllocationMode)
    -> SceneBuilder1.TextureLocation
  {
    // If requested, or if the image is too big, use a separate page.
    if mode == .ownPage || requested_size.x > Int32(TextureAllocator1.ATLAS_TEXTURE_LENGTH)
      || requested_size.y > Int32(TextureAllocator1.ATLAS_TEXTURE_LENGTH)
    {
      return self.allocate_image(requested_size: requested_size)
    }

    // Try to add to each atlas.
    for (page_index, page) in pages.enumerated() {
      if let page {
        switch page.allocator {
        case .image:
          break
        case .atlas(var allocator):
          if let rect = allocator.allocate(requested_size: requested_size) {
            pages[page_index] = .init(allocator: .atlas(allocator), is_new: page.is_new)
            return SceneBuilder1.TextureLocation(page: UInt32(page_index), rect: rect)
          }
        }
      }
    }

    // Add a new atlas.
    let page = get_first_free_page_id()
    var allocator = TextureAtlasAllocator()
    let rect = allocator.allocate(requested_size: requested_size)!

    while page >= pages.count {
      pages.append(nil)
    }

    pages[Int(page)] = TexturePage(allocator: .atlas(allocator), is_new: true)

    return .init(page: page, rect: rect)
  }

  mutating func allocate_image(requested_size: SIMD2<Int32>) -> SceneBuilder1.TextureLocation {
    let page = self.get_first_free_page_id()

    let rect = PFRect<Int32>(origin: .zero, size: requested_size)

    while page >= self.pages.count {
      self.pages.append(nil)
    }

    self.pages[Int(page)] = TexturePage(
      allocator: .image(size: rect.size),
      is_new: true
    )

    return .init(page: page, rect: rect)
  }

  func get_first_free_page_id() -> UInt32 {
    for (page_index, page) in pages.enumerated() {
      if page == nil {
        return UInt32(page_index)
      }
    }

    return UInt32(pages.count)
  }

  func page_size(_ page_id: UInt32) -> SIMD2<Int32> {
    let page = self.pages[Int(page_id)]!
    switch page.allocator {
    case .atlas(let atlas):
      return .init(repeating: Int32(atlas.size))
    case .image(let size):
      return size
    }
  }

  func page_scale(_ page_id: UInt32) -> SIMD2<Float32> {
    SIMD2<Float32>(1.0, 1.0) / SIMD2<Float32>(self.page_size(page_id))
  }

  func page_ids() -> TexturePageIter {
    var first_index = 0
    while first_index < pages.count && pages[first_index] == nil {
      first_index += 1
    }

    return TexturePageIter(allocator: self, nextIndex: first_index)
  }

  func page_is_new(_ page_id: UInt32) -> Bool {
    self.pages[Int(page_id)]!.is_new
  }

  mutating func mark_all_pages_as_allocated() {
    for (i, page) in pages.enumerated() {
      if var page {
        page.is_new = false
        pages[i] = page
      }
    }
  }

  mutating func free(location: SceneBuilder1.TextureLocation) {
    let id = Int(location.page)
    let page = pages[id]!
    switch page.allocator {
    case .image(let size):
      break
    case .atlas(var atlas_allocator):
      atlas_allocator.free(rect: location.rect)

      self.pages[id] = .init(allocator: .atlas(atlas_allocator), is_new: page.is_new)

      if !atlas_allocator.isEmpty {
        // Keep the page around.
        return
      }
    }

    // If we got here, free the page.
    // TODO(pcwalton): Actually tell the renderer to free this page!
    self.pages[id] = nil
  }
}

extension UInt32 {
  /// Get the next power of two
  var nextPowerOfTwo: UInt32 {
    guard self > 1 else { return 1 }
    var x = self
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1
  }
}

extension UInt64 {
  var nextPowerOfTwo: UInt64 {
    guard self > 1 else { return 1 }
    var n = self - 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1
  }
}
