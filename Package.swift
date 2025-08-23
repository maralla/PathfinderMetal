// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "PathfinderMetal",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "PathfinderMetal", targets: ["PathfinderMetal"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "PathfinderMetal",
            dependencies: [
                .product(name: "Collections", package: "swift-collections")
            ],
            path: "Sources/PathfinderMetal",
            exclude: ["Shaders/"],
            resources: [
                // Copy curated Resources folder (only metallib + png)
                .copy("Resources")
            ]
        ),
        .testTarget(
            name: "PathfinderMetalTests",
            dependencies: ["PathfinderMetal"]
        ),
    ]
)
