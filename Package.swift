// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Macotaur",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MacotaurCore", targets: ["MacotaurCore"]),
        .executable(name: "macotaur", targets: ["macotaur"]),
        .executable(name: "macotaur-bench", targets: ["macotaur-bench"]),
    ],
    targets: [
        .target(name: "MacotaurCore"),
        .executableTarget(
            name: "macotaur",
            dependencies: ["MacotaurCore"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "macotaur-bench",
            dependencies: ["MacotaurCore"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MacotaurCoreTests",
            dependencies: ["MacotaurCore"]
        ),
    ]
)
