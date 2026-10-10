import Testing
@testable import MacotaurCore

typealias Edge = CycleFinder.Edge

/// Deterministic generator for reproducible random graphs.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// The 2-core, plainly: drop every edge with a degree-1 endpoint until none is left.
func referencePeel(_ edges: [Edge]) -> [Edge] {
    var alive = [Bool](repeating: true, count: edges.count)
    var changed = true
    while changed {
        changed = false
        var du: [UInt32: Int] = [:], dv: [UInt32: Int] = [:]
        for (i, e) in edges.enumerated() where alive[i] { du[e.u, default: 0] += 1; dv[e.v, default: 0] += 1 }
        for (i, e) in edges.enumerated() where alive[i] && (du[e.u] == 1 || dv[e.v] == 1) { alive[i] = false; changed = true }
    }
    return edges.indices.filter { alive[$0] }.map { edges[$0] }
}

/// tromp's path-reversal search over the reference 2-core, with a Dictionary forest.
func referenceCycles(_ edges: [Edge], length: Int) -> [[Edge]] {
    var next: [UInt64: UInt64] = [:]
    var cycles: [[Edge]] = []
    func path(_ start: UInt64) -> [UInt64] {
        var p = [start]
        while let n = next[p.last!] { p.append(n) }
        return p
    }
    func edge(_ a: UInt64, _ b: UInt64) -> Edge {
        a & 1 == 0 ? Edge(u: UInt32(a >> 1), v: UInt32(b >> 1)) : Edge(u: UInt32(b >> 1), v: UInt32(a >> 1))
    }
    for e in referencePeel(edges) {
        let u0 = UInt64(e.u) << 1, v0 = UInt64(e.v) << 1 | 1
        let us = path(u0), vs = path(v0)
        if us.last == vs.last {
            var nu = us.count - 1, nv = vs.count - 1
            let m = min(nu, nv)
            nu -= m; nv -= m
            while us[nu] != vs[nv] { nu += 1; nv += 1 }
            if nu + nv + 1 == length {
                cycles.append([e] + (0..<nu).map { edge(us[$0], us[$0 + 1]) } + (0..<nv).map { edge(vs[$0], vs[$0 + 1]) })
            }
        } else if us.count < vs.count {
            for i in stride(from: us.count - 1, to: 0, by: -1) { next[us[i]] = us[i - 1] }
            next[u0] = v0
        } else {
            for i in stride(from: vs.count - 1, to: 0, by: -1) { next[vs[i]] = vs[i - 1] }
            next[v0] = u0
        }
    }
    return cycles
}

/// Small dense graphs (many short cycles, duplicate edges, ids at both ends of
/// the 32-bit range) must give exactly the reference 2-core and cycles. One
/// scratch serves them all after a big graph has grown its tables, so its
/// generation tag wraps along the way.
@Test func cycleFinderMatchesReference() {
    var rng = SplitMix64(state: 42)
    let scratch = CycleFinder.Scratch()
    // 2^19 + 1 edges size the tables at 2^20 slots, leaving 11 generation
    // bits: the tables are cleared after 2047 graphs.
    let big = (0..<(1 << 19) + 1).map { _ in Edge(u: .random(in: 0..<1 << 19, using: &rng), v: .random(in: 0..<1 << 19, using: &rng)) }
    let core = scratch.peel(big)
    var du: [UInt32: Int] = [:], dv: [UInt32: Int] = [:]
    for e in core { du[e.u, default: 0] += 1; dv[e.v, default: 0] += 1 }
    #expect(!core.isEmpty && core.allSatisfy { du[$0.u]! >= 2 && dv[$0.v]! >= 2 })

    var found = 0
    for g in 0..<2500 {
        let nodes = UInt32.random(in: 2...256, using: &rng)
        let base = [0, UInt32.max - nodes + 1, .random(in: 0...UInt32.max - nodes, using: &rng)][g % 3]
        let edges = (0..<Int.random(in: 1...400, using: &rng)).map { _ in
            Edge(u: base + .random(in: 0..<nodes, using: &rng), v: base + .random(in: 0..<nodes, using: &rng))
        }
        let length = [2, 4, 6, 8, 10][g % 5]
        let peeled = scratch.peel(edges)
        #expect(peeled == referencePeel(edges))
        let cycles = CycleFinder.searchCycles(peeled, length: length, next: &scratch.next)
        #expect(cycles == referenceCycles(edges, length: length))
        found += cycles.count
        if g % 100 == 0 {
            #expect(CycleFinder.peel(edges) == peeled)
            #expect(CycleFinder.findCycles(edges, length: length) == referenceCycles(edges, length: length))
        }
    }
    #expect(found > 1000)
    #expect(CycleFinder.peel([]).isEmpty && CycleFinder.findCycles([]).isEmpty)
}

/// A 42-cycle with dead-end chains hanging off it, hidden in a big sparse
/// random forest, must be found and be all that's left after peeling.
@Test func cycleFinderFindsPlantedCycle() {
    var rng = SplitMix64(state: 7)
    var edges = (0..<100_000).map { _ in Edge(u: .random(in: 0..<1 << 24, using: &rng), v: .random(in: 0..<1 << 24, using: &rng)) }
    // Nodes from 2^25 up are unused by the forest.
    let cycle = (0..<21).flatMap { i -> [Edge] in
        let u = UInt32(1 << 25 + i), v = UInt32(1 << 25 + i)
        return [Edge(u: u, v: v), Edge(u: UInt32(1 << 25 + (i + 1) % 21), v: v)]
    }
    var chains: [Edge] = []
    for i in 0..<21 {
        // Alternate sides along each chain, starting from cycle node u_i.
        var u = UInt32(1 << 25 + i)
        for j in 0..<5 {
            let v = UInt32(1 << 26 + i * 8 + j)
            chains.append(Edge(u: u, v: v))
            u = UInt32(1 << 26 + i * 8 + j)
            chains.append(Edge(u: u, v: v))
        }
    }
    edges += cycle + chains
    edges.shuffle(using: &rng)
    #expect(Set(CycleFinder.peel(edges)) == Set(cycle))
    let found = CycleFinder.findCycles(edges)
    #expect(found.count == 1 && Set(found[0]) == Set(cycle))
}

/// Every plan must fit its kernels: threadgroup memory, sparse capacity,
/// packing limits and the counter layout. Larger graphs than C29 (whose late
/// buckets don't fit the sparse kernel as early) and odd configs must plan too.
/// (MetalSolver rejects edgeBits > 31: the shaders hold nodes in 32 bits.)
@Test func planFitsKernels() {
    var configs: [MetalSolver.Config] = []
    for edgeBits in [10, 19, 29, 30, 31] {
        var c = MetalSolver.Config()
        c.edgeBits = edgeBits
        configs.append(c)
    }
    var c = MetalSolver.Config()
    c.lateBucketBits = [10: 11, 40: 10]
    configs.append(c)
    c.packedUntil = 80
    configs.append(c)
    c = MetalSolver.Config()
    c.roundGroups[0] = 0   // round 0 on a persistent grid, once per seed pass
    configs.append(c)
    for c in configs {
        for (cores, sparseThreads, splits) in [(nil, 512, 1), (0, 1024, 1), (10, 256, 1), (10, 512, 3)] {
            let (stages, plans) = MetalSolver.plan(c, seedSplits: splits, gpuCores: cores, sparseThreads: sparseThreads)
            // Work counters follow the stage counters, without overlapping.
            var end = stages.last!.counterOffset + stages.last!.buckets
            for (r, p) in plans.enumerated() {
                #expect(p.workOffset == end && p.workRegions == (r == 0 ? splits : 1))
                end += p.workWords
            }
            // Metal needs every bound offset inside the buffer.
            let words = MetalSolver.counterWords(stages, plans)
            #expect(words >= end && plans.allSatisfy { $0.workOffset < words })
            for (r, p) in plans.enumerated() {
                #expect(p.threadgroupBytes <= 32768)
                if p.kernel == .sparse {
                    #expect(!stages[r].packed && p.threads <= min(512, sparseThreads))
                    #expect(stages[r].cap <= p.threads * Shaders.sparseEdgesPerThread && p.tableWords >= 2 * stages[r].cap)
                } else {
                    #expect(p.kernel == (!stages[r].packed ? .twoPass : stages[r + 1].packed ? .packed : .packedToPairs))
                    #expect(p.groups == (cores == 10 && c.roundGroups[r] == 0 ? 10 : nil))
                }
            }
            for (s, st) in stages.enumerated() {
                if st.packed {
                    #expect(c.edgeBits > st.bits && 2 * c.edgeBits - st.bits <= 48 && st.buckets * st.cap < 1 << 32)
                    #expect(s == 0 || stages[s - 1].packed)
                }
                if s > 0 { #expect(st.counterOffset == stages[s - 1].counterOffset + stages[s - 1].buckets) }
            }
        }
    }
    let (stages, plans) = MetalSolver.plan(MetalSolver.Config(), gpuCores: 10)
    #expect(stages[0].packed && stages[17].packed && !stages[18].packed)
    #expect(stages[17].buckets == 4096 && stages[18].buckets == 2048 && stages[80].buckets == 2048)
    #expect(plans[0].kernel == .packed && plans[17].kernel == .packedToPairs && plans[18].kernel == .sparse)
    #expect(plans[1].groups == 10 && plans[1].threads == 1024 && plans[0].groups == nil)
}
