// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "PathfinderMacApp",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "PathfinderMacApp", targets: ["PathfinderMacApp"]),
    ],
    dependencies: [
        .package(path: "../../")
    ],
    targets: [
        .executableTarget(
            name: "PathfinderMacApp",
            dependencies: [
                .product(name: "PathfinderMetal", package: "PathfinderMetal")
            ],
            path: "Sources/PathfinderMacApp",
            sources: ["main.swift"],
            swiftSettings: [
                .unsafeFlags(["-O"])
            ]
        ),
    ]
) 
