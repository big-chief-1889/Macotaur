/// Slow, straightforward Cuckaroo solver. It's a reference for testing (small
/// graphs or one-off checks) and not meant for mining.
public enum CPUSolver {
    /// All edges of the graph, indexed by nonce.
    public static func edges(keys: Cuckaroo.Keys, edgeBits: Int) -> [CycleFinder.Edge] {
        let n = 1 << edgeBits
        let mask = UInt64(n - 1)
        var out = [CycleFinder.Edge](repeating: .init(u: 0, v: 0), count: n)
        var block = [UInt64](repeating: 0, count: Cuckaroo.edgeBlockSize)
        for b in stride(from: 0, to: n, by: Cuckaroo.edgeBlockSize) {
            var s = Cuckaroo.SipState(keys)
            for i in 0..<Cuckaroo.edgeBlockSize {
                s.hash24(UInt64(b + i))
                block[i] = s.digest
            }
            let last = block[Cuckaroo.edgeBlockSize - 1]
            for i in 0..<Cuckaroo.edgeBlockSize {
                let h = i == Cuckaroo.edgeBlockSize - 1 ? last : block[i] ^ last
                out[b + i] = .init(u: UInt32(h & mask), v: UInt32((h >> 32) & mask))
            }
        }
        return out
    }

    /// Indices of edges surviving `rounds` of leaf trimming (alternating sides).
    public static func trim(_ edges: [CycleFinder.Edge], edgeBits: Int, rounds: Int) -> [Int] {
        var alive = Array(edges.indices)
        var degree = [UInt8](repeating: 0, count: 1 << edgeBits)
        for r in 0..<rounds {
            for i in degree.indices { degree[i] = 0 }
            let side = r & 1
            for i in alive {
                let n = Int(side == 0 ? edges[i].u : edges[i].v)
                if degree[n] < 2 { degree[n] += 1 }
            }
            alive = alive.filter { degree[Int(side == 0 ? edges[$0].u : edges[$0].v)] >= 2 }
        }
        return alive
    }

    /// Full solve: trim, find cycles, map back to sorted nonces.
    public static func solve(keys: Cuckaroo.Keys, edgeBits: Int, rounds: Int = 80) -> [[UInt64]] {
        let all = edges(keys: keys, edgeBits: edgeBits)
        let alive = trim(all, edgeBits: edgeBits, rounds: rounds)
        let cycles = CycleFinder.findCycles(alive.map { all[$0] })
        return cycles.compactMap { cycle in
            let wanted = Set(cycle)
            let nonces = alive.filter { wanted.contains(all[$0]) }.map(UInt64.init).sorted()
            return nonces.count == Cuckaroo.proofSize ? nonces : nil
        }
    }
}
