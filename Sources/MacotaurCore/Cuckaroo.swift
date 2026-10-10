// CPU reference implementation of Cuckaroo (Grin's original "cuckaroo", as
// ported from github.com/tromp/cuckoo). Used to verify GPU output and in tests;
// it is not on the hot path.

public enum Cuckaroo {
    public static let proofSize = 42
    public static let edgeBlockBits = 6
    public static let edgeBlockSize = 1 << edgeBlockBits
    public static let edgeBlockMask = UInt64(edgeBlockSize - 1)

    /// The four 64-bit siphash keys derived from a header (blake2b output read
    /// as little-endian words).
    public struct Keys: Equatable, Sendable {
        public var k0, k1, k2, k3: UInt64
        public init(_ k0: UInt64, _ k1: UInt64, _ k2: UInt64, _ k3: UInt64) {
            (self.k0, self.k1, self.k2, self.k3) = (k0, k1, k2, k3)
        }
        public init(hash32 bytes: [UInt8]) {
            precondition(bytes.count == 32)
            func word(_ i: Int) -> UInt64 {
                var w: UInt64 = 0
                for b in 0..<8 { w |= UInt64(bytes[i * 8 + b]) << (8 * b) }
                return w
            }
            self.init(word(0), word(1), word(2), word(3))
        }
    }

    public enum VerifyError: Error, Equatable {
        case wrongSize, edgeTooBig, notAscending, endpointsDontMatch, branch, deadEnd, shortCycle
    }

    // MARK: SipHash-2-4 (tromp variant: keys are the initial state directly)

    struct SipState {
        var v0, v1, v2, v3: UInt64
        init(_ k: Keys) { (v0, v1, v2, v3) = (k.k0, k.k1, k.k2, k.k3) }

        @inline(__always) mutating func round() {
            v0 &+= v1; v2 &+= v3
            v1 = v1.rotl(13); v3 = v3.rotl(16)
            v1 ^= v0; v3 ^= v2
            v0 = v0.rotl(32)
            v2 &+= v1; v0 &+= v3
            v1 = v1.rotl(17); v3 = v3.rotl(21)
            v1 ^= v2; v3 ^= v0
            v2 = v2.rotl(32)
        }

        @inline(__always) mutating func hash24(_ nonce: UInt64) {
            v3 ^= nonce
            round(); round()
            v0 ^= nonce
            v2 ^= 0xff
            round(); round(); round(); round()
        }

        var digest: UInt64 { (v0 ^ v1) ^ (v2 ^ v3) }
    }

    /// Single siphash-2-4 of `nonce`.
    public static func siphash24(_ keys: Keys, _ nonce: UInt64) -> UInt64 {
        var s = SipState(keys)
        s.hash24(nonce)
        return s.digest
    }

    /// Cuckaroo's block hash: hash the whole 64-nonce block containing `nonce`
    /// with a chained state, then xor the requested output with the block's last.
    public static func sipBlock(_ keys: Keys, _ nonce: UInt64) -> UInt64 {
        let nonce0 = nonce & ~edgeBlockMask
        let idx = Int(nonce & edgeBlockMask)
        var s = SipState(keys)
        var out: UInt64 = 0
        for i in 0..<edgeBlockSize {
            s.hash24(nonce0 + UInt64(i))
            if i == idx { out = s.digest }
        }
        return idx == edgeBlockSize - 1 ? out : out ^ s.digest
    }

    /// Endpoints of edge `nonce` in a graph with `edgeBits`-bit node ids.
    public static func edge(_ keys: Keys, _ nonce: UInt64, edgeBits: Int = 29) -> (u: UInt32, v: UInt32) {
        let mask = (UInt64(1) << edgeBits) - 1
        let h = sipBlock(keys, nonce)
        return (UInt32(h & mask), UInt32((h >> 32) & mask))
    }

    // MARK: Verification

    /// Port of tromp/Grin `verify`: edges must be strictly ascending and form a
    /// single cycle of length `proofSize`.
    public static func verify(_ nonces: [UInt64], keys: Keys, edgeBits: Int = 29) throws(VerifyError) {
        guard nonces.count == proofSize else { throw .wrongSize }
        let edgeMask = (UInt64(1) << edgeBits) - 1
        var uvs = [UInt64](repeating: 0, count: 2 * proofSize)
        var xor0: UInt64 = 0, xor1: UInt64 = 0
        for n in 0..<proofSize {
            if nonces[n] > edgeMask { throw .edgeTooBig }
            if n > 0 && nonces[n] <= nonces[n - 1] { throw .notAscending }
            let e = edge(keys, nonces[n], edgeBits: edgeBits)
            uvs[2 * n] = UInt64(e.u)
            uvs[2 * n + 1] = UInt64(e.v)
            xor0 ^= UInt64(e.u)
            xor1 ^= UInt64(e.v)
        }
        if xor0 | xor1 != 0 { throw .endpointsDontMatch }
        var n = 0, i = 0
        repeat {
            var j = i
            var k = i
            while true {
                k = (k + 2) % (2 * proofSize)
                if k == i { break }
                if uvs[k] == uvs[i] {
                    if j != i { throw .branch }
                    j = k
                }
            }
            if j == i { throw .deadEnd }
            i = j ^ 1
            n += 1
        } while i != 0
        if n != proofSize { throw .shortCycle }
    }
}

extension UInt64 {
    @inline(__always) func rotl(_ b: UInt64) -> UInt64 { (self << b) | (self >> (64 - b)) }
}
