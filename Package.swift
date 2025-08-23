// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PathfinderMetal",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "PathfinderMetal", targets: ["PathfinderMetal"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "PathfinderMetal",
            dependencies: [],
            path: "Sources/PathfinderMetal",
            exclude: ["Shaders/"],
            resources: [
                // Copy curated Resources folder (only metallib + png)
                .copy("Resources")
            ]
        ),
        .testTarget(
            name: "PathfinderMetalTests",
            dependencies: [
                "PathfinderMetal"
            ]
        ),
    ]
)
