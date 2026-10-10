# Macotaur

GPU miner for Tari's C29 (Cuckaroo29) proof of work, written in Swift and Metal for Apple Silicon Macs.

This repo is the miner itself: a Swift package with the solver, pool client and mining loop, plus a command-line miner and a benchmark. The Mac app lives in [MacotaurGUI](https://github.com/big-chief-1889/MacotaurGUI).

## Requirements

- An Apple Silicon Mac running macOS 15 or later
- Xcode (or the Swift 6 toolchain) to build. Shaders are compiled at runtime, so the separate Metal toolchain isn't needed.

## Download

[Releases](https://github.com/big-chief-1889/Macotaur/releases) has the command-line miner as `macotaur-<version>-macos-arm64.zip`. It is ad-hoc signed and not notarized; if macOS blocks it after a browser download, clear the download flag:

```
xattr -d com.apple.quarantine macotaur
```

## Build

```
swift build -c release
```

## Mine from the command line

```
.build/release/macotaur --pool xtm-c29.kryptex.network:7040 --separator / --wallet <your XTM address> --worker mac
```

LuckyPool also works: `--pool taric29.luckypool.io:3111` (it uses the default `.` separator). Stop with Ctrl-C, or pass `--minutes N`. For apps that run it as a helper, `--json` prints one stats object per line and SIGTERM stops it cleanly. `--help` lists the other options.

## Use it as a library

```swift
.package(url: "https://github.com/big-chief-1889/Macotaur", branch: "main")
```

```swift
import MacotaurCore

let solver = try MetalSolver()
let miner = Miner(solver: solver, pool: .init(host: "xtm-c29.kryptex.network", port: 7040, user: "<address>/mac"))
miner.start()
```

`MetalSolver` on its own trims a graph and returns the leftover edges; `CycleFinder` and `MetalSolver.proofs(for:)` turn those into verified 42-cycles.

A `Miner` runs once. To pause, stop it and later start a new one on the same solver, which skips the shader compile and the allocation. The solver keeps its GPU memory until you let it go: call `stop()` and `waitUntilStopped()` on the miner, then `waitUntilIdle()` on the solver, and drop both. Miner events arrive on the miner's own queues, so hop to yours before touching app state.

## Benchmark and tests

```
.build/release/macotaur-bench --graphs 50 --profile
swift test
MACOTAUR_GPU_TESTS=1 swift test
```

The GPU tests solve a real Tari mainnet block, in the default mode and in each low-memory mode, and run a full mining round trip against a mock pool.

## How the solver works

Mean trimming on the GPU (edges sorted into buckets), then a cycle search on the CPU.

- **Seed:** hash every edge with siphash and append it to one of 4096 buckets by its first endpoint.
- **Rounds 0-17:** each bucket counts node degrees in a threadgroup bitmap, keeps edges whose node appears at least twice, and files the survivors by their other endpoint. These rounds are limited by memory bandwidth, so edges are packed into 6 bytes. The second pass reads each bucket backwards so it starts on data that's still cached.
- **Rounds 18-79:** the buckets are small enough for a threadgroup hash table.
- **CPU:** after 80 rounds about 300k edges are left. The CPU peels them down to the 2-core, finds 42-cycles with path reversal, and has the GPU recover the nonces.
- **Low-memory mode:** the seed can run in k passes. Each pass hashes every edge but keeps only 1/k of the buckets, and round 0 trims them before the next pass reuses the memory, so the same edges survive. The solver picks the fewest passes that fit in 45% of RAM and 65% of Metal's recommended working set: one pass (5.3 GB) with 16 GB or more, three (3.1 GB) on 8 GB Macs. Extra passes cost speed: on an M5, two, three and four passes run about 12%, 21% and 31% slower. `macotaur-bench --memory-plan` prints the sizes, and `--seed-splits K` forces a mode.

On an M5 MacBook Air a graph takes about 310 ms of GPU time. Speed scales with GPU cores and memory bandwidth: Pro, Max and Ultra chips are faster, and Macs with fans hold their peak speed longer.

## License

BSD 3-Clause. Parts of the solver follow John Tromp's reference code; see THIRD_PARTY_NOTICES.
