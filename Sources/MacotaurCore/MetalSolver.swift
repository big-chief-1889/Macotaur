import Foundation
import IOKit
import Metal

/// GPU Cuckaroo solver for Apple Silicon. One instance owns about 5.3 GB of
/// GPU buffers for C29 (3.0-3.7 GB in low-memory mode, see Config.seedSplits);
/// graphs run back to back on one queue, so several can be in flight while
/// the CPU searches the previous graph's leftovers for cycles.
public final class MetalSolver: @unchecked Sendable {
    public struct Config: Sendable {
        public var edgeBits = 29
        /// 80 leaves ~300k edges for the CPU cycle finder, overlapped with the
        /// next graph. Late rounds use the cheap sparse kernel.
        public var rounds = 80
        /// Buckets are keyed by low node bits. More buckets thrash the cache
        /// with scattered writes, and fewer can't fit their bitmaps in
        /// threadgroup memory (needs bucketBits >= edgeBits - 17).
        public var bucketBits = 12
        /// From each listed stage on, 2^bits buckets wherever they fit the
        /// sparse kernel (cap <= 4096). For C29 their bitmaps wouldn't fit in
        /// 32 KB, so those rounds run sparse and packing ends there. Fewer,
        /// fuller buckets make the late rounds cheaper.
        public var lateBucketBits: [Int: Int] = [18: 11]
        /// Stages before this one hold edges packed in 6 bytes instead of 8
        /// (see storePacked); nil: up to the first stage a sparse round reads.
        /// Where packing ends past round 8 hardly matters.
        public var packedUntil: Int? = nil
        public var twoPassThreads = 512
        /// Bitmap-round threads per group by round, overriding twoPassThreads.
        /// 512 is best for the packed rounds except round 1 on its grid.
        public var roundThreads: [Int: Int] = [1: 1024]
        /// Bitmap rounds that run a persistent grid of this many groups (0 =
        /// one per GPU core, < 0 = off) looping over the buckets. It caps the
        /// buckets in flight so pass 2's re-read hits cache. That pays off in
        /// round 1 only: round 0's buckets are too big for the cache either
        /// way, and from round 2 on they fit without it.
        public var roundGroups: [Int: Int] = [1: 0]
        public var seedBlocks = 4         // 64-edge siphash blocks per seed thread
        public var seedThreads = 256
        public var tailCapacity = 1 << 22
        /// Seed passes (1...maxSeedSplits); nil picks automaticSeedSplits for
        /// this Mac. With k > 1 ("low-memory mode") each pass hashes every
        /// edge but keeps only the next 1/k of stage 0's buckets, and round 0
        /// trims them before the next pass reuses the memory. The same edges
        /// survive; each extra pass costs another round of hashing.
        public var seedSplits: Int? = nil
        public init() {}
    }

    public static let maxSeedSplits = 8

    public struct Graph: Sendable {
        public var keys: Cuckaroo.Keys
        public var edges: [CycleFinder.Edge]
        public var gpuSeconds: Double
    }

    /// Geometry of one stage's output: 2^bits buckets of `cap` edges.
    struct Stage {
        var bits: Int; var cap: Int; var counterOffset: Int; var packed: Bool
        var buckets: Int { 1 << bits }
        var bytes: Int { bytes(buckets: buckets) }
        func bytes(buckets n: Int) -> Int { n * cap * (packed ? 6 : 8) }
    }

    /// How round r runs: input stage r, output stage r + 1.
    struct RoundPlan {
        /// twoPass and the packed ones are the bitmap kernel on 8- or 6-byte edges.
        enum Kernel { case twoPass, packed, packedToPairs, sparse }
        var kernel: Kernel
        var threads: Int
        var tableWords: Int        // two-pass: words per bitmap; sparse: hash slots
        var threadgroupBytes: Int
        var groups: Int? = nil     // persistent grid size; nil = one group per bucket
        var workOffset = 0         // persistent grid: counter words [next bucket, one per group]
        var workRegions = 1        // persistent grid: one such region per dispatch (per seed pass for round 0)
        var workWords: Int { groups.map { workRegions * (1 + $0) } ?? 0 }
    }

    public let config: Config
    public let device: MTLDevice
    let queue: MTLCommandQueue
    let seed, tail, recover: MTLComputePipelineState
    let roundPipelines: [RoundPlan.Kernel: MTLComputePipelineState]
    /// Low-memory mode's seed and round 0, each on one range of buckets.
    let seedSliced, round0Sliced: MTLComputePipelineState?
    /// Stage s lives in A if s is odd, else B (the seed writes B).
    let bufA, bufB: MTLBuffer
    let stages: [Stage]
    let plans: [RoundPlan]
    /// Stage 0's buckets filled by each seed pass (one range unless low-memory).
    let seedSlices: [Range<Int>]

    /// Per in-flight graph: its bucket counters and the tail output.
    final class TailSlot: @unchecked Sendable {
        let edges: MTLBuffer
        let count: MTLBuffer
        let counters: MTLBuffer
        init(_ d: MTLDevice, capacity: Int, counterWords: Int) throws {
            func buffer(_ size: Int) throws -> MTLBuffer {
                guard let b = d.makeBuffer(length: size, options: .storageModeShared) else { throw SolverError.allocation(size) }
                return b
            }
            edges = try buffer(capacity * 8)
            count = try buffer(4)
            counters = try buffer(counterWords * 4)
        }
    }
    let slots: [TailSlot]
    let slotSemaphore: DispatchSemaphore
    var nextSlot = 0
    var lastGPUEnd: CFTimeInterval = 0
    let lock = NSLock()

    public enum SolverError: Error { case noDevice, compile(String), allocation(Int), unsupported(String) }

    public init(config: Config = Config(), inFlight: Int = 2) throws {
        self.config = config
        // The shaders hold nodes in 32 bits and pad with an all-ones sentinel.
        guard (10...31).contains(config.edgeBits) else { throw SolverError.unsupported("edgeBits \(config.edgeBits)") }
        guard config.seedThreads > 0, config.seedBlocks > 0 else { throw SolverError.unsupported("seed threads/blocks") }
        // Each seed pass is followed by round 0 on its buckets.
        if let k = config.seedSplits, !(1...Self.maxSeedSplits).contains(k) || (k > 1 && config.rounds < 1) {
            throw SolverError.unsupported("seedSplits \(k) with \(config.rounds) rounds")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw SolverError.noDevice }
        self.device = device
        self.queue = queue

        let opts = MTLCompileOptions()
        opts.mathMode = .fast
        opts.preprocessorMacros = [
            "EDGEBITS": NSNumber(value: config.edgeBits),
            "SEED_BLOCKS": NSNumber(value: config.seedBlocks),
        ]
        let lib: MTLLibrary
        do { lib = try device.makeLibrary(source: Shaders.source, options: opts) }
        catch { throw SolverError.compile("\(error)") }
        func pipe(_ name: String, packed: Bool = false, pairs: Bool = false, slice: Bool = false) throws -> MTLComputePipelineState {
            let values = MTLFunctionConstantValues()
            var (packed, pairs, slice) = (packed, pairs, slice)
            values.setConstantValue(&packed, type: .bool, index: 0)
            values.setConstantValue(&pairs, type: .bool, index: 1)
            values.setConstantValue(&slice, type: .bool, index: 2)
            return try device.makeComputePipelineState(function: lib.makeFunction(name: name, constantValues: values))
        }
        let sparse = try pipe("trimSparse")
        let splits = config.seedSplits ?? Self.automaticSeedSplits(
            config: config, physicalMemory: ProcessInfo.processInfo.physicalMemory,
            recommendedWorkingSet: device.recommendedMaxWorkingSetSize, maxBufferLength: device.maxBufferLength)
        let (stages, planned) = Self.plan(config, seedSplits: splits, gpuCores: Self.gpuCoreCount(),
                                          sparseThreads: sparse.maxTotalThreadsPerThreadgroup)
        self.stages = stages
        seedSlices = Self.seedSlices(buckets: stages[0].buckets, splits: splits)
        func roundPipe(_ kernel: RoundPlan.Kernel, slice: Bool = false) throws -> MTLComputePipelineState {
            switch kernel {
            case .twoPass: try pipe("trimTwoPass", slice: slice)
            case .packed: try pipe("trimPacked", slice: slice)
            case .packedToPairs: try pipe("trimPacked", pairs: true, slice: slice)
            case .sparse: sparse
            }
        }
        seed = try pipe("seed", packed: stages[0].packed)
        roundPipelines = [
            .twoPass: try roundPipe(.twoPass), .packed: try roundPipe(.packed),
            .packedToPairs: try roundPipe(.packedToPairs), .sparse: sparse,
        ]
        let sliced = seedSlices.count > 1
        seedSliced = sliced ? try pipe("seed", packed: stages[0].packed, slice: true) : nil
        round0Sliced = sliced ? try roundPipe(planned[0].kernel, slice: true) : nil
        tail = try pipe("tail"); recover = try pipe("recover")
        // The bitmap kernels run any group size; the sparse one needs its own.
        var plans = planned
        for i in plans.indices {
            // An empty group would silently drop every edge.
            guard plans[i].threads > 0 else { throw SolverError.unsupported("round \(i) with \(plans[i].threads) threads") }
            var limit = roundPipelines[plans[i].kernel]!.maxTotalThreadsPerThreadgroup
            if i == 0, let p = round0Sliced { limit = min(limit, p.maxTotalThreadsPerThreadgroup) }
            guard plans[i].threads > limit else { continue }
            guard plans[i].kernel != .sparse else { throw SolverError.compile("trimSparse can't run \(plans[i].threads) threads per group") }
            plans[i].threads = limit
        }
        self.plans = plans

        let (sizeA, sizeB) = Self.bufferSizes(stages, seedSlices: seedSlices)
        func buffer(_ size: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: size, options: .storageModePrivate) else { throw SolverError.allocation(size) }
            return b
        }
        bufA = try buffer(sizeA); bufB = try buffer(sizeB)
        slots = try (0..<inFlight).map { _ in try TailSlot(device, capacity: config.tailCapacity, counterWords: Self.counterWords(stages, plans)) }
        slotSemaphore = DispatchSemaphore(value: inFlight)
    }

    // MARK: Memory

    /// Seed passes this solver runs per graph; more than 1 is low-memory mode.
    public var seedSplits: Int { seedSlices.count }
    public var isLowMemoryMode: Bool { seedSplits > 1 }

    /// Every GPU buffer this solver owns, in bytes. They stay allocated until
    /// the solver is released.
    public var gpuMemoryBytes: Int {
        bufA.length + bufB.length + slots.reduce(0) { $0 + $1.edges.length + $1.count.length + $1.counters.length }
    }

    public struct MemoryPlan: Sendable, Equatable {
        public var seedSplits: Int
        public var gpuMemoryBytes: Int
        public var isLowMemoryMode: Bool { seedSplits > 1 }
    }

    /// The GPU memory a solver would allocate with `seedSplits` seed passes
    /// (without compiling or allocating anything). `gpuCores` defaults to
    /// this Mac's.
    public static func memoryPlan(config: Config = Config(), seedSplits: Int, inFlight: Int = 2,
                                  gpuCores: Int? = MetalSolver.gpuCoreCount()) -> MemoryPlan {
        let m = layout(config, seedSplits: seedSplits, inFlight: inFlight, gpuCores: gpuCores)
        return MemoryPlan(seedSplits: m.passes, gpuMemoryBytes: m.total)
    }

    /// Buffer sizes for a config: A (odd stages), B (even stages and one
    /// seed pass's slice of stage 0), and all of them with the per-graph slots.
    static func layout(_ c: Config, seedSplits: Int, inFlight: Int, gpuCores: Int?) -> (passes: Int, a: Int, b: Int, total: Int) {
        let (stages, plans) = plan(c, seedSplits: seedSplits, gpuCores: gpuCores)
        let slices = seedSlices(buckets: stages[0].buckets, splits: seedSplits)
        let (a, b) = bufferSizes(stages, seedSlices: slices)
        let slot = c.tailCapacity * 8 + 4 + counterWords(stages, plans) * 4   // TailSlot's buffers
        return (slices.count, a, b, a + b + inFlight * slot)
    }

    /// What `MetalSolver(config:)` would allocate on this Mac, resolving an
    /// automatic seedSplits as the solver does.
    public static func memoryPlan(config: Config = Config(), device: MTLDevice?, inFlight: Int = 2) -> MemoryPlan {
        let k = config.seedSplits ?? automaticSeedSplits(
            config: config, physicalMemory: ProcessInfo.processInfo.physicalMemory,
            recommendedWorkingSet: device?.recommendedMaxWorkingSetSize ?? .max, maxBufferLength: device?.maxBufferLength ?? .max)
        return memoryPlan(config: config, seedSplits: k, inFlight: inFlight)
    }

    /// Bytes `init(config:inFlight:)` would allocate on this Mac, without
    /// compiling or allocating anything.
    public static func requiredBytes(config: Config = Config(), inFlight: Int = 2) -> Int {
        memoryPlan(config: config, device: MTLCreateSystemDefaultDevice(), inFlight: inFlight).gpuMemoryBytes
    }

    /// The automatic seedSplits: the fewest seed passes whose buffers fit a
    /// budget of 45% of RAM and 65% of Metal's recommended working set (what
    /// the GPU may keep resident; the rest stays with macOS, apps and the
    /// display), with no buffer over maxBufferLength. If none fits, the most
    /// memory-saving count. Solver buffers are wired while the GPU works, so
    /// they can't be compressed or swapped. For C29 this is one pass on Macs
    /// with 16 GB or more and three on 8 GB Macs. Past four passes nothing
    /// is saved, as round 1's output is then the largest even stage.
    public static func automaticSeedSplits(config: Config = Config(), physicalMemory: UInt64, recommendedWorkingSet: UInt64,
                                           maxBufferLength: Int = .max) -> Int {
        guard config.rounds > 0 else { return 1 }
        let budget = min(Double(physicalMemory) * 0.45, Double(recommendedWorkingSet) * 0.65)
        var best = (splits: 1, bytes: Int.max)
        for k in 1...maxSeedSplits {
            let m = layout(config, seedSplits: k, inFlight: 2, gpuCores: nil)
            guard m.passes == k, m.total < best.bytes else { break }   // another pass would save nothing
            best = (k, m.total)
            if Double(m.total) <= budget && max(m.a, m.b) <= maxBufferLength { break }
        }
        return best.splits
    }

    /// Stage 0's bucket ranges, one per seed pass: `splits` near-equal ranges
    /// (fewer if there are fewer buckets than that).
    static func seedSlices(buckets: Int, splits: Int) -> [Range<Int>] {
        let n = (buckets + max(1, splits) - 1) / max(1, splits)
        return stride(from: 0, to: buckets, by: n).map { $0..<min($0 + n, buckets) }
    }

    /// Bytes of buffer A (odd stages) and B (even stages). The seed only ever
    /// holds one slice of stage 0.
    static func bufferSizes(_ stages: [Stage], seedSlices: [Range<Int>]) -> (a: Int, b: Int) {
        let seed = stages[0].bytes(buckets: seedSlices.map(\.count).max()!)
        let a = stride(from: 1, to: stages.count, by: 2).map { stages[$0].bytes }.max() ?? 8
        let b = stride(from: 2, to: stages.count, by: 2).map { stages[$0].bytes }.max() ?? 0
        return (a, max(seed, b))
    }

    /// Counter words per in-flight graph: per-bucket counts, then the work
    /// counters of persistent rounds. Every round binds its work offset, which
    /// Metal requires to lie inside the buffer, so there is at least one word
    /// past each.
    static func counterWords(_ stages: [Stage], _ plans: [RoundPlan]) -> Int {
        plans.reduce(stages.last!.counterOffset + stages.last!.buckets) { max($0, $1.workOffset + max(1, $1.workWords)) }
    }

    static func roundUp(_ x: Int, _ m: Int) -> Int { (x + m - 1) / m * m }

    /// GPU core count from the IORegistry (Metal doesn't expose it).
    public static func gpuCoreCount() -> Int? {
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"))
        guard service != IO_OBJECT_NULL else { return nil }
        defer { IOObjectRelease(service) }
        let count = IORegistryEntryCreateCFProperty(service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue()
        return (count as? NSNumber)?.intValue
    }

    /// Fraction of all edges left after round r, measured on C29 graphs (past
    /// round 10 it decays ~ r^-1.8). The seed is stage "round -1" with all edges.
    static func survival(afterRound r: Int) -> Double {
        let measured: [Double] = [0.632, 0.296, 0.175, 0.117, 0.0837, 0.0631, 0.0493, 0.0396, 0.0326, 0.0273, 0.0232]
        if r < 0 { return 1 }
        return r < measured.count ? measured[r] : measured.last! * pow(11 / Double(r + 1), 1.8)
    }

    /// Pick each stage's geometry and each round's kernel.
    ///
    /// The bitmap kernels cover a bucket's 2^(edgeBits - bits) nodes in at
    /// most 32 KB of threadgroup memory. Once buckets are small, the sparse
    /// kernel's hash table (sized to the edges) is cheaper. It holds up to
    /// sparseThreads (at most 512, for a 32 KB table) times 8 edges.
    static func plan(_ c: Config, seedSplits: Int = 1, gpuCores: Int?, sparseThreads: Int = 512) -> ([Stage], [RoundPlan]) {
        let n = Double(1 << c.edgeBits)
        let ept = Shaders.sparseEdgesPerThread, sparseCap = min(512, sparseThreads) * ept
        func cap(_ s: Int, bits: Int) -> Int {
            let mean = n * survival(afterRound: s - 1) / Double(1 << bits)
            // Survival varies graph to graph, most in the long tail.
            let margin = s <= 3 ? 1.04 : 1.3
            return roundUp(Int(mean * margin + 7 * mean.squareRoot()) + 32, 32)
        }
        // Words per bitmap; the bitmap kernels keep two ("seen", "seen twice").
        func bitmapWords(_ bits: Int) -> Int { max(1, (1 << (c.edgeBits - bits)) / 32) }
        func hashSlots(_ cap: Int) -> Int { 1 << Int(log2(Double(2 * cap)).rounded(.up)) }

        let base = max(c.bucketBits, c.edgeBits - 17)
        let bits = (0...c.rounds).map { s -> Int in
            guard let late = c.lateBucketBits.filter({ $0.key <= s }).max(by: { $0.key < $1.key })?.value,
                  late < base, cap(s, bits: late) <= sparseCap else { return base }
            return late
        }
        let caps = (0...c.rounds).map { cap($0, bits: bits[$0]) }
        // A hash table sized to the edges beats clearing a bitmap sized to
        // the node range once buckets are small, and it's the only option
        // when the bitmaps don't fit.
        let sparse = (0..<c.rounds).map { r in
            caps[r] <= sparseCap && (hashSlots(caps[r]) < bitmapWords(bits[r]) || 8 * bitmapWords(bits[r]) > 32768)
        }
        // trimSparse and tail read pairs, so packing ends by the first sparse
        // round. A packed stage needs key bits, key + other node in 48 bits,
        // 32-bit indexes and (for its round) bitmaps that fit.
        let packable = (0...c.rounds).map { s in
            c.edgeBits > bits[s] && 2 * c.edgeBits - bits[s] <= 48 && (1 << bits[s]) * caps[s] < 1 << 32 && 8 * bitmapWords(bits[s]) <= 32768
        }
        let packedEnd = min(c.packedUntil ?? sparse.firstIndex(of: true) ?? c.rounds, packable.firstIndex(of: false) ?? c.rounds, c.rounds)

        var stages: [Stage] = []
        var offset = 0
        for s in 0...c.rounds {
            stages.append(Stage(bits: bits[s], cap: caps[s], counterOffset: offset, packed: s < packedEnd))
            offset += 1 << bits[s]
        }
        var plans = (0..<c.rounds).map { r -> RoundPlan in
            if sparse[r] && !stages[r].packed {
                let slots = hashSlots(caps[r])
                return RoundPlan(kernel: .sparse, threads: max(64, roundUp((caps[r] + ept - 1) / ept, 32)), tableWords: slots,
                                 threadgroupBytes: slots * 4)
            }
            // Packed stages come first, so packed output means packed input.
            let kernel: RoundPlan.Kernel = !stages[r].packed ? .twoPass : stages[r + 1].packed ? .packed : .packedToPairs
            // With the core count unknown, run one group per bucket.
            let cores = gpuCores.flatMap { $0 > 0 ? $0 : nil }
            let groups = c.roundGroups[r].flatMap { $0 > 0 ? $0 : $0 == 0 ? cores : nil }.map { min($0, stages[r].buckets) }
            return RoundPlan(kernel: kernel, threads: c.roundThreads[r] ?? c.twoPassThreads, tableWords: bitmapWords(bits[r]),
                             threadgroupBytes: roundUp(8 * bitmapWords(bits[r]), 16), groups: groups)
        }
        // Persistent rounds' work counters follow the stage counters (all are
        // zeroed per graph). In low-memory mode round 0 runs once per seed
        // pass, each with its own.
        if c.rounds > 0 { plans[0].workRegions = seedSlices(buckets: stages[0].buckets, splits: seedSplits).count }
        for r in plans.indices {
            plans[r].workOffset = offset
            offset += plans[r].workWords
        }
        return (stages, plans)
    }

    /// Buffer holding stage s (0 = seed output).
    func buffer(stage s: Int) -> MTLBuffer { s % 2 == 1 ? bufA : bufB }

    func encodeClear(_ cb: MTLCommandBuffer, slot: TailSlot) {
        let blit = cb.makeBlitCommandEncoder()!
        blit.fill(buffer: slot.counters, range: 0..<slot.counters.length, value: 0)
        blit.fill(buffer: slot.count, range: 0..<4, value: 0)
        blit.endEncoding()
    }

    /// Seed pass `pass`: hash every edge, keep those in seedSlices[pass].
    func encodeSeed(_ enc: MTLComputeCommandEncoder, keys: Cuckaroo.Keys, slot: TailSlot, pass: Int = 0) {
        var k = keys
        var p = SIMD2<UInt32>(UInt32(stages[0].cap), UInt32(stages[0].buckets - 1))
        let slice = seedSlices[pass]
        enc.setComputePipelineState(seedSliced ?? seed)
        enc.setBytes(&k, length: 32, index: 0)
        enc.setBuffer(buffer(stage: 0), offset: 0, index: 1)
        enc.setBuffer(slot.counters, offset: (stages[0].counterOffset + slice.lowerBound) * 4, index: 2)
        enc.setBytes(&p, length: 8, index: 3)
        if seedSliced != nil {
            var s = SIMD2<UInt32>(UInt32(slice.lowerBound), UInt32(slice.count))
            enc.setBytes(&s, length: 8, index: 4)
        }
        enc.dispatchThreads(MTLSize(width: (1 << config.edgeBits) / 64 / config.seedBlocks, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: config.seedThreads, height: 1, depth: 1))
    }

    /// Round r; in low-memory mode round 0 runs once per seed pass, on just
    /// that pass's buckets (`pass`).
    func encodeRound(_ enc: MTLComputeCommandEncoder, _ r: Int, slot: TailSlot, pass: Int? = nil) {
        let (input, output, plan) = (stages[r], stages[r + 1], plans[r])
        let slice = pass.map { seedSlices[$0] } ?? 0..<input.buckets
        var p: [UInt32] = [UInt32(input.cap), UInt32(output.cap), UInt32(input.bits), UInt32(output.bits), UInt32(plan.tableWords),
                           UInt32(slice.lowerBound), UInt32(slice.count)]
        enc.setComputePipelineState(pass != nil ? round0Sliced! : roundPipelines[plan.kernel]!)
        enc.setBuffer(buffer(stage: r), offset: 0, index: 0)
        enc.setBuffer(slot.counters, offset: (input.counterOffset + slice.lowerBound) * 4, index: 1)
        enc.setBuffer(buffer(stage: r + 1), offset: 0, index: 2)
        enc.setBuffer(slot.counters, offset: output.counterOffset * 4, index: 3)
        enc.setBytes(&p, length: p.count * 4, index: 4)
        enc.setBuffer(slot.counters, offset: (plan.workOffset + (pass ?? 0) * (plan.workWords / plan.workRegions)) * 4, index: 5)
        enc.setThreadgroupMemoryLength(plan.threadgroupBytes, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: min(plan.groups ?? slice.count, slice.count), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: plan.threads, height: 1, depth: 1))
    }

    func encodeTail(_ enc: MTLComputeCommandEncoder, slot: TailSlot) {
        // Each round swaps edge orientation, so after an even number of
        // rounds edges are (u, v) again.
        let r = config.rounds
        var p = SIMD3<UInt32>(UInt32(stages[r].cap), UInt32(config.tailCapacity), r % 2 == 0 ? 0 : 1)
        enc.setComputePipelineState(tail)
        enc.setBuffer(buffer(stage: r), offset: 0, index: 0)
        enc.setBuffer(slot.counters, offset: stages[r].counterOffset * 4, index: 1)
        enc.setBuffer(slot.edges, offset: 0, index: 2)
        enc.setBuffer(slot.count, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<SIMD3<UInt32>>.size, index: 4)   // uint3 is 16 bytes
        enc.dispatchThreadgroups(MTLSize(width: stages[r].buckets, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// Encode a whole graph into one command buffer, in order. If `split` is
    /// given it is called after each stage and returns the encoder to continue
    /// with (profiling puts each stage in its own command buffer).
    func encodeTrim(_ cb: MTLCommandBuffer, keys: Cuckaroo.Keys, slot: TailSlot,
                    split: ((String) -> MTLComputeCommandEncoder)? = nil) {
        encodeClear(cb, slot: slot)
        var enc = cb.makeComputeCommandEncoder()!
        func boundary(_ name: String) {
            guard let split else { return }
            enc.endEncoding()
            enc = split(name)
        }
        // Low-memory mode: each seed pass is trimmed by round 0 before the
        // next one overwrites stage 0.
        let sliced = seedSliced != nil
        for pass in seedSlices.indices {
            encodeSeed(enc, keys: keys, slot: slot, pass: pass)
            boundary("seed")
            guard sliced else { break }
            encodeRound(enc, 0, slot: slot, pass: pass)
            boundary("round 0")
        }
        for r in (sliced ? 1 : 0)..<config.rounds {
            encodeRound(enc, r, slot: slot)
            boundary("round \(r)")
        }
        encodeTail(enc, slot: slot)
        enc.endEncoding()
    }

    /// Trim one graph; the completion runs on a Metal callback thread with
    /// the surviving edges. Blocks while `inFlight` graphs are already queued.
    public func trim(keys: Cuckaroo.Keys, completion: @escaping @Sendable (Graph) -> Void) {
        slotSemaphore.wait()
        lock.lock()
        let slot = slots[nextSlot]
        nextSlot = (nextSlot + 1) % slots.count
        lock.unlock()

        let cb = queue.makeCommandBuffer()!
        encodeTrim(cb, keys: keys, slot: slot)
        let cap = config.tailCapacity
        cb.addCompletedHandler { [slotSemaphore, weak self] cb in
            // With two graphs queued, gpuStartTime can predate the previous
            // graph's end (its counter clear starts early), so count only the
            // time this graph had the GPU to itself. The Miner's power modes
            // rest the GPU in proportion to this.
            let busy = self?.lock.withLock {
                defer { self?.lastGPUEnd = cb.gpuEndTime }
                return cb.gpuEndTime - max(cb.gpuStartTime, self?.lastGPUEnd ?? 0)
            } ?? (cb.gpuEndTime - cb.gpuStartTime)
            let n = min(Int(slot.count.contents().load(as: UInt32.self)), cap)
            let p = slot.edges.contents().bindMemory(to: CycleFinder.Edge.self, capacity: n)
            let edges = Array(UnsafeBufferPointer(start: p, count: n))
            slotSemaphore.signal()
            completion(Graph(keys: keys, edges: edges, gpuSeconds: busy))
        }
        cb.commit()
    }

    /// Run one graph with every stage in its own command buffer and return
    /// per-stage GPU time in seconds (in low-memory mode, "seed" and "round 0"
    /// sum all seed passes).
    public func profile(keys: Cuckaroo.Keys) -> [(stage: String, seconds: Double)] {
        var cbs: [(String, MTLCommandBuffer)] = []
        var cb = queue.makeCommandBuffer()!
        encodeTrim(cb, keys: keys, slot: slots[0]) { name in
            cbs.append((name, cb))
            cb.commit()
            cb = self.queue.makeCommandBuffer()!
            return cb.makeComputeCommandEncoder()!
        }
        cbs.append(("tail", cb))
        cb.commit()
        cb.waitUntilCompleted()
        var times: [(stage: String, seconds: Double)] = []
        for (name, cb) in cbs {
            let t = cb.gpuEndTime - cb.gpuStartTime
            if let i = times.firstIndex(where: { $0.stage == name }) { times[i].seconds += t } else { times.append((name, t)) }
        }
        return times
    }

    /// Synchronous trim, for tests and the benchmark's single-graph mode.
    public func trim(keys: Cuckaroo.Keys) -> Graph {
        let done = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: Graph?
        trim(keys: keys) { result = $0; done.signal() }
        done.wait()
        return result!
    }

    /// Map a cycle's endpoint pairs back to sorted edge nonces on the GPU.
    public func recover(keys: Cuckaroo.Keys, cycle: [CycleFinder.Edge]) -> [UInt64] {
        let found = device.makeBuffer(length: 4, options: .storageModeShared)!
        let nonces = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
        found.contents().storeBytes(of: 0, as: UInt32.self)
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        var k = keys
        var cyc = cycle
        var len = UInt32(cycle.count)
        enc.setComputePipelineState(recover)
        enc.setBytes(&k, length: 32, index: 0)
        enc.setBytes(&cyc, length: cycle.count * 8, index: 1)
        enc.setBytes(&len, length: 4, index: 2)
        enc.setBuffer(found, offset: 0, index: 3)
        enc.setBuffer(nonces, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: (1 << config.edgeBits) / 64, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let n = min(Int(found.contents().load(as: UInt32.self)), 64)
        let p = nonces.contents().bindMemory(to: UInt32.self, capacity: n)
        return (0..<n).map { UInt64(p[$0]) }.sorted()
    }

    /// Trim, find cycles, recover and verify. Returns verified 42-cycles.
    public func solve(keys: Cuckaroo.Keys) -> [[UInt64]] {
        let g = trim(keys: keys)
        return proofs(for: g)
    }

    public func proofs(for g: Graph) -> [[UInt64]] {
        proofs(for: g, cycles: CycleFinder.findCycles(g.edges))
    }

    /// Recover and verify cycles already found in `g`.
    public func proofs(for g: Graph, cycles: [[CycleFinder.Edge]]) -> [[UInt64]] {
        cycles.compactMap { cycle in
            let nonces = recover(keys: g.keys, cycle: cycle)
            guard (try? Cuckaroo.verify(nonces, keys: g.keys, edgeBits: config.edgeBits)) != nil else { return nil }
            return nonces
        }
    }

    /// Fill level of the last trimmed graph, per stage (seed, then each round):
    /// (largest bucket, total edges, bucket capacity, bucket count).
    public func stageCounts() -> [(max: Int, total: Int, capacity: Int, buckets: Int)] {
        let c = lock.withLock { slots[(nextSlot + slots.count - 1) % slots.count].counters }
        let p = c.contents().bindMemory(to: UInt32.self, capacity: c.length / 4)
        return stages.map { s in
            let v = (0..<s.buckets).map { Int(p[s.counterOffset + $0]) }
            return (v.max()!, v.reduce(0, +), s.cap, s.buckets)
        }
    }

    /// Per-round kernel choice, for diagnostics.
    public var roundPlans: [(kernel: String, threads: Int, threadgroupBytes: Int)] {
        plans.map { ("\($0.kernel)", $0.threads, $0.threadgroupBytes) }
    }

    /// Blocks until no graph is in flight on the GPU. Call only after the
    /// Miner using this solver has stopped (otherwise new graphs keep coming).
    public func waitUntilIdle() {
        for _ in slots { slotSemaphore.wait() }
        for _ in slots { slotSemaphore.signal() }
    }
}
