import Foundation

/// Finds 42-cycles among the edges left after GPU trimming, using tromp's
/// "cuckoo" path-reversal forest. Cycles are returned as endpoint pairs; the
/// GPU recovery pass then maps those back to edge nonces.
public struct CycleFinder {
    public struct Edge: Hashable, Sendable {
        public var u: UInt32
        public var v: UInt32
        public init(u: UInt32, v: UInt32) { self.u = u; self.v = v }
    }

    static let maxPathLength = 8192

    public static func findCycles(_ edges: [Edge], length: Int = Cuckaroo.proofSize) -> [[Edge]] {
        Scratch.withPooled { searchCycles($0.peel(edges), length: length, next: &$0.next) }
    }

    /// Reduce to the graph's 2-core: repeatedly drop edges with an endpoint of
    /// degree 1, which can't be on any cycle. Linear time, and it removes the
    /// long dead-end chains that make the path walk below slow. Survivors keep
    /// their input order.
    public static func peel(_ edges: [Edge]) -> [Edge] {
        Scratch.withPooled { $0.peel(edges) }
    }

    /// tromp's path-reversal cycle search over (ideally peeled) edges.
    static func searchCycles(_ edges: [Edge], length: Int, next: inout NodeMap) -> [[Edge]] {
        // Node ids: u side even, v side odd, so both sides share one map.
        next.reset(capacity: edges.count * 2)
        var us = [UInt64](), vs = [UInt64]()
        us.reserveCapacity(maxPathLength)
        vs.reserveCapacity(maxPathLength)
        var cycles: [[Edge]] = []

        func path(from start: UInt64, into p: inout [UInt64]) -> Bool {
            p.removeAll(keepingCapacity: true)
            var node = start
            p.append(node)
            while let n = next[node] {
                node = n
                p.append(node)
                if p.count >= maxPathLength { return false }
            }
            return true
        }

        func edge(_ a: UInt64, _ b: UInt64) -> Edge {
            a & 1 == 0 ? Edge(u: UInt32(a >> 1), v: UInt32(b >> 1)) : Edge(u: UInt32(b >> 1), v: UInt32(a >> 1))
        }

        for e in edges {
            let u0 = UInt64(e.u) << 1, v0 = UInt64(e.v) << 1 | 1
            guard path(from: u0, into: &us), path(from: v0, into: &vs) else { continue }
            var nu = us.count - 1, nv = vs.count - 1
            if us[nu] == vs[nv] {
                // Same tree: walk both paths back to where they join.
                let m = min(nu, nv)
                nu -= m; nv -= m
                while us[nu] != vs[nv] { nu += 1; nv += 1 }
                if nu + nv + 1 == length {
                    var cycle = [e]
                    for i in 0..<nu { cycle.append(edge(us[i], us[i + 1])) }
                    for i in 0..<nv { cycle.append(edge(vs[i], vs[i + 1])) }
                    cycles.append(cycle)
                }
                continue
            }
            // Reverse the shorter path so the new edge can be attached at its root.
            if nu < nv {
                while nu > 0 { next[us[nu]] = us[nu - 1]; nu -= 1 }
                next[u0] = v0
            } else {
                while nv > 0 { next[vs[nv]] = vs[nv - 1]; nv -= 1 }
                next[v0] = u0
            }
        }
        return cycles
    }

    /// Working memory kept from graph to graph. Allocating and zeroing ~32 MB
    /// per graph cost page faults and memory traffic; here every table is
    /// generation-tagged, so a new graph needs neither. Callers borrow one
    /// from a pool, so concurrent searches each get their own.
    final class Scratch: @unchecked Sendable {
        /// Two open-addressing tables, u side then v side, of 2^tableBits
        /// entries each: generation | dense node id | node.
        private var table = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        private var tableBits = 0
        private var generation: UInt64 = 0
        /// Per node id: degree << 32 | XOR of its live neighbours' ids.
        private var info = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        /// Per edge: the ids of its u and v endpoints.
        private var ends = UnsafeMutablePointer<UInt32>.allocate(capacity: 1)
        private var edgeCapacity = 0
        private var touched: UInt64 = 0
        var next = NodeMap()

        private static let lock = NSLock()
        nonisolated(unsafe) private static var pool: [Scratch] = []

        static func withPooled<R>(_ body: (Scratch) -> R) -> R {
            let s = lock.withLock { pool.popLast() } ?? Scratch()
            defer { lock.withLock { pool.append(s) } }
            return body(s)
        }

        deinit { table.deallocate(); info.deallocate(); ends.deallocate() }

        /// Size the tables for m edges and start a new generation.
        private func reserve(edges m: Int) {
            precondition(m <= 1 << 30, "too many edges for the cycle finder")
            // Up to m nodes per side; ~m / 2 is typical after trimming, a load
            // of ~0.3 (it nears 1, and probing slows, only for near-matchings).
            var bits = 4
            while 1 << bits < m { bits += 1 }
            if bits > tableBits {
                table.deallocate()
                table = .allocate(capacity: 2 << bits)
                table.initialize(repeating: 0, count: 2 << bits)
                tableBits = bits
                generation = 0
            }
            if 2 * m > edgeCapacity {
                info.deallocate(); ends.deallocate()
                edgeCapacity = 2 * m
                info = .allocate(capacity: edgeCapacity)
                ends = .allocate(capacity: edgeCapacity)
            }
            // Ids take tableBits + 1 bits (< 2m), nodes 32: the generation
            // gets the rest and wraps every few thousand graphs.
            generation += 1
            if generation >> (31 - tableBits) != 0 {
                table.update(repeating: 0, count: 2 << tableBits)
                generation = 1
            }
        }

        func peel(_ edges: [Edge]) -> [Edge] {
            let m = edges.count
            guard m > 0 else { return [] }
            reserve(edges: m)
            let bits = tableBits, table = table, info = info, ends = ends
            let idMask = (UInt64(1) << UInt64(bits + 1)) - 1
            let base = generation << UInt64(33 + bits)
            let keyMask = ~(idMask << 32)
            let slotMask = (1 << bits) - 1
            let vTable = table + (1 << bits)
            var nodes = 0

            @inline(__always) func slot(_ node: UInt32) -> Int {
                Int(truncatingIfNeeded: (UInt64(node) &* 0x9E37_79B9_7F4A_7C15) >> UInt64(64 - bits))
            }
            // Dense id of a node, assigned on first sight. Entries from older
            // generations compare below `base` and count as empty.
            @inline(__always) func id(_ t: UnsafeMutablePointer<UInt64>, _ node: UInt32) -> Int {
                let key = base | UInt64(node)
                var i = slot(node)
                while true {
                    let e = t[i]
                    if e & keyMask == key { return Int(truncatingIfNeeded: (e >> 32) & idMask) }
                    if e < base {
                        t[i] = key | UInt64(nodes) << 32
                        info[nodes] = 0
                        nodes += 1
                        return nodes - 1
                    }
                    i = (i + 1) & slotMask
                }
            }

            edges.withUnsafeBufferPointer { ep in
                var touched: UInt64 = 0
                for batch in stride(from: 0, to: m, by: 32) {
                    let end = min(m, batch + 32)
                    // Load the batch's table slots first, with no branches on
                    // the loaded data, so their cache misses overlap.
                    for j in batch..<end { touched &+= table[slot(ep[j].u)] &+ vTable[slot(ep[j].v)] }
                    for j in batch..<end {
                        let a = id(table, ep[j].u), b = id(vTable, ep[j].v)
                        ends[2 * j] = UInt32(truncatingIfNeeded: a)
                        ends[2 * j + 1] = UInt32(truncatingIfNeeded: b)
                        info[a] = (info[a] &+ 1 << 32) ^ UInt64(b)
                        info[b] = (info[b] &+ 1 << 32) ^ UInt64(a)
                    }
                }
                self.touched = touched   // keeps the loads above alive
            }

            // A degree-1 node's XOR is its only neighbour: drop that edge and
            // follow the chain while the neighbour falls to degree 1. Sixteen
            // chains advance in turn so their dependent loads overlap; each
            // step re-checks the degree, as another chain may have got there.
            var start = 0
            func nextStart() -> Int {
                while start < nodes {
                    start += 1
                    if info[start - 1] >> 32 == 1 { return start - 1 }
                }
                return -1
            }
            var cursors = (0..<16).map { _ in nextStart() }
            var live = cursors.reduce(0) { $0 + ($1 >= 0 ? 1 : 0) }
            while live > 0 {
                for c in cursors.indices where cursors[c] >= 0 {
                    let n = cursors[c], x = info[n]
                    if x >> 32 == 1 {
                        let other = Int(truncatingIfNeeded: x & 0xFFFF_FFFF)
                        info[n] = 0
                        info[other] = (info[other] &- 1 << 32) ^ UInt64(n)
                        cursors[c] = other
                    } else {
                        cursors[c] = nextStart()
                        if cursors[c] < 0 { live -= 1 }
                    }
                }
            }

            // Every node left has degree 0 (info 0) or >= 2; an edge survives
            // iff neither endpoint was peeled.
            var core: [Edge] = []
            edges.withUnsafeBufferPointer { ep in
                for i in 0..<m where info[Int(ends[2 * i])] != 0 && info[Int(ends[2 * i + 1])] != 0 {
                    core.append(ep[i])
                }
            }
            return core
        }
    }
}

/// Open-addressing node -> node map; ~10x faster than Dictionary here.
/// Entries are tagged with a generation, so reset() empties the map without
/// clearing it and one map serves graph after graph. Keys are < 2^33.
struct NodeMap {
    private var keys: [UInt64] = []
    private var values: [UInt64] = []
    private var shift: UInt64 = 0
    private var mask = 0
    private var base: UInt64 = 0

    mutating func reset(capacity: Int) {
        // Load factor <= 1/2 for the expected number of nodes.
        var size = 16
        while size < capacity * 2 { size <<= 1 }
        if size > keys.count {
            keys = [UInt64](repeating: 0, count: size)
            values = [UInt64](repeating: 0, count: size)
            shift = UInt64(64 - size.trailingZeroBitCount)
            mask = size - 1
            base = 0
        }
        base &+= 1 << 33
        if base == 0 {
            for i in keys.indices { keys[i] = 0 }
            base = 1 << 33
        }
    }

    @inline(__always) private func slot(_ key: UInt64) -> Int {
        Int(truncatingIfNeeded: (key &* 0x9E37_79B9_7F4A_7C15) >> shift)
    }

    subscript(key: UInt64) -> UInt64? {
        get {
            let k = base | key
            var i = slot(key)
            while true {
                let s = keys[i]
                if s == k { return values[i] }
                if s < base { return nil }
                i = (i + 1) & mask
            }
        }
        set {
            let k = base | key
            var i = slot(key)
            while keys[i] != k && keys[i] >= base { i = (i + 1) & mask }
            keys[i] = k
            values[i] = newValue!
        }
    }
}
