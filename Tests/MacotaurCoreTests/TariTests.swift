import Foundation
import Testing
@testable import MacotaurCore

// Vectors from tari-project/tari base_layer/core/src/proof_of_work/cuckaroo_pow.rs
let tariMiningHash = [UInt8](hex: "4dbfee3eb7b9a6a27d2a4a8d754eb77cc5493006945c1246e50e9beb4de5ffa5")!
let tariNonce: UInt64 = 0x9d589ed597ed42d0
let tariPacked = [UInt8](hex: "ab3c742104de5808220f0ebd1d24e2a279489c9fa8c9264f754181433226655286c69a08f166e523283813e5eceabbec042598193013a34cd966601e064d5f3dbb911efe39536e7847f3180071865023e18c6c1c627696aece2ff401938abacc8ed2f446b8ba71785b074a086d36d1d61d638dc0eab156b8883214cbff9962f199b96c92349df3d1d7b647ed0cdf6dfde1e49077bcfb74b503")!
let tariEdges: [UInt64] = [
    24394923, 46592033, 58968194, 71842682, 75995694, 81022926, 97860763, 105410600, 106063142, 138729012,
    150559164, 170291280, 197045966, 202539638, 206356582, 215689620, 218504800, 232385274, 243238820,
    250666150, 268537652, 296239928, 296891268, 315542595, 338677422, 341088271, 346272558, 359697897,
    364349211, 377758979, 391857369, 403811427, 408334826, 413242437, 413564914, 426980322, 433277222,
    460056825, 473150750, 473935291, 477597924, 497788893,
]

@Test func blake2bVectors() {
    // Cross-checked with Python hashlib.blake2b(digest_size=...)
    #expect(Blake2b.hash([]).hex == "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8")
    #expect(Blake2b.hash((0..<200).map { UInt8($0) }).hex == "63c3d97a9f8894d5e043a707b0fee7f7ec4c049a23bbf1079df20b4165f9e22d")
    #expect(Blake2b.hash((0..<256).map { UInt8($0) }, outLength: 64).hex ==
        "1ecc896f34d3f9cac484c73f75f6a5fb58ee6784be41b35f46067b9c65c63a6794d3d744112c653f73dd7deb6666204c5a9bfa5b46081fc10fdbe7884fa5cbf8")
}

@Test func tariHeaderHash() {
    // test_header_hash: Blake2b-256(nonce BE || mining_hash)
    var b = Blake2b()
    b.update([UInt8](hex: "9d589ed597ed42d1")!)
    b.update(tariMiningHash)
    #expect(b.finalize().hex == "49b48f77df94943cf3a422c5a0b528c737cc38a7b6c36076e81abcede5b2be3a")
    let k = Tari.sipKeys(miningHash: tariMiningHash, nonce: 0x9d589ed597ed42d1)
    #expect(k == Cuckaroo.Keys(hash32: [UInt8](hex: "49b48f77df94943cf3a422c5a0b528c737cc38a7b6c36076e81abcede5b2be3a")!))
}

@Test func tariPacking() {
    #expect(Tari.packedProofSize == 153)
    #expect(Tari.unpack(tariPacked) == tariEdges)
    #expect(Tari.pack(tariEdges) == tariPacked)
}

@Test func tariRealSolution() throws {
    let keys = Tari.sipKeys(miningHash: tariMiningHash, nonce: tariNonce)
    try Cuckaroo.verify(tariEdges, keys: keys)
    #expect(Tari.powHash(tariEdges).hex == "06d52b90ccfd4db1a52cc133a46dac0dc4577343c1a27d200865a81d717a60e1")
    #expect(Tari.difficulty(nonces: tariEdges) == 37)
}

@Test func tariDifficultyEdges() {
    func d(_ prefix: String, _ fill: String) -> UInt64 {
        Tari.difficulty(powHash: [UInt8](hex: prefix + String(repeating: fill, count: 24))!)
    }
    // Reference values from Python: min((2**256 - 1) // h, 2**64 - 1)
    #expect(d("00000000ffffffff", "ff") == 4294967296)
    #expect(d("0000000100000000", "00") == 4294967295)
    #expect(d("ffffffffffffffff", "ff") == 1)
    #expect(d("0000000000000000", "01") == .max)
}

/// End to end on the GPU: solving the real block's graph must rediscover its
/// proof, in the default mode and in every low-memory mode (seed split into
/// passes), which must also trim to exactly the same edges. The solvers run
/// one at a time, so the test never holds more than one's GPU memory.
@Test(.enabled(if: ProcessInfo.processInfo.environment["MACOTAUR_GPU_TESTS"] != nil))
func gpuSolvesRealTariBlock() throws {
    let keys = Tari.sipKeys(miningHash: tariMiningHash, nonce: tariNonce)
    var reference: (edges: [UInt64], totals: [Int], memory: Int)?
    for splits in [1, 2, 3, 4] {
        var config = MetalSolver.Config()
        config.seedSplits = splits
        let solver = try MetalSolver(config: config)
        #expect(solver.seedSplits == splits && solver.isLowMemoryMode == (splits > 1))
        #expect(solver.gpuMemoryBytes == MetalSolver.memoryPlan(config: config, seedSplits: splits).gpuMemoryBytes)
        let graph = solver.trim(keys: keys)
        #expect(solver.proofs(for: graph).contains(tariEdges), "seed passes: \(splits)")
        let run = (edges: graph.edges.map { UInt64($0.u) << 32 | UInt64($0.v) }.sorted(),
                   totals: solver.stageCounts().map(\.total), memory: solver.gpuMemoryBytes)
        if let reference {
            #expect(run.edges == reference.edges && run.totals == reference.totals, "seed passes: \(splits)")
            #expect(run.memory < reference.memory)
        }
        reference = run
    }
}

@Test func gpuSolvesCuckaroo19Vector() throws {
    // Low-memory modes too, also with round 0 on a persistent grid (its
    // work counters are per seed pass).
    for (splits, round0Groups) in [(1, nil), (2, nil), (3, 0), (4, 3)] as [(Int, Int?)] {
        var config = MetalSolver.Config()
        config.edgeBits = 19
        config.seedSplits = splits
        config.roundGroups[0] = round0Groups
        let solver = try MetalSolver(config: config)
        #expect(solver.solve(keys: v1Keys19).contains(v1Sol19), "seed passes: \(splits)")
        #expect(solver.solve(keys: v2Keys19).contains(v2Sol19), "seed passes: \(splits)")
    }
}

/// Low-memory mode's buffers, and the automatic choice of seed passes by
/// RAM size (Metal recommends ~2/3 of RAM on 8 and 16 GB Macs).
@Test func seedSplitsMemoryPlan() {
    let plans = (1...4).map { MetalSolver.memoryPlan(seedSplits: $0) }
    #expect(plans.map(\.seedSplits) == [1, 2, 3, 4] && plans.map(\.isLowMemoryMode) == [false, true, true, true])
    for i in 1..<plans.count { #expect(plans[i].gpuMemoryBytes < plans[i - 1].gpuMemoryBytes) }
    #expect(plans[0].gpuMemoryBytes > 5_500_000_000 && plans[3].gpuMemoryBytes < 3_300_000_000)
    // The default layout: stage 0 and round 0's output, each in full.
    let (stages, _) = MetalSolver.plan(MetalSolver.Config(), gpuCores: 10)
    #expect(MetalSolver.bufferSizes(stages, seedSlices: [0..<4096]) == (stages[1].bytes, stages[0].bytes))
    // Passes cover every bucket once, in order.
    for (buckets, k) in [(4096, 1), (4096, 3), (4096, 4), (2048, 8), (5, 4)] {
        let slices = MetalSolver.seedSlices(buckets: buckets, splits: k)
        #expect(slices.first?.lowerBound == 0 && slices.last?.upperBound == buckets && slices.count <= k)
        #expect(zip(slices, slices.dropFirst()).allSatisfy { $0.upperBound == $1.lowerBound && !$0.isEmpty })
    }
    func auto(_ gb: UInt64, workingSet: UInt64? = nil) -> Int {
        MetalSolver.automaticSeedSplits(physicalMemory: gb << 30, recommendedWorkingSet: workingSet ?? (gb << 30) * 2 / 3)
    }
    #expect(auto(8) == 3 && auto(16) == 1 && auto(24) == 1 && auto(192) == 1)
    #expect(auto(4) == 4)   // nothing fits: save the most
    #expect(auto(16, workingSet: 4 << 30) == 4)
    #expect(MetalSolver.automaticSeedSplits(physicalMemory: 16 << 30, recommendedWorkingSet: 11 << 30, maxBufferLength: 3 << 30) == 2)
    var c = MetalSolver.Config()
    c.rounds = 0
    #expect(MetalSolver.automaticSeedSplits(config: c, physicalMemory: 1 << 30, recommendedWorkingSet: 1 << 30) == 1)
}
