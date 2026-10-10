import Foundation

/// Tari (XTM) specifics for Cuckaroo29, from tari-project/tari
/// base_layer/core/src/proof_of_work/cuckaroo_pow.rs.
public enum Tari {
    public static let edgeBits = 29
    public static let cycleLength = 42
    /// ceil(42 * 29 / 8): the pow_data length (after the algorithm byte 3).
    public static let packedProofSize = (cycleLength * edgeBits + 7) / 8

    /// siphash keys = Blake2b-256(nonce as 8 big-endian bytes || mining_hash).
    public static func sipKeys(miningHash: [UInt8], nonce: UInt64) -> Cuckaroo.Keys {
        precondition(miningHash.count == 32)
        var b = Blake2b(outLength: 32)
        b.update((0..<8).map { UInt8(truncatingIfNeeded: nonce >> (56 - 8 * $0)) })
        b.update(miningHash)
        return Cuckaroo.Keys(hash32: b.finalize())
    }

    /// Pack edge nonces LSB-first, `edgeBits` each, into a little-endian bitstream.
    public static func pack(_ nonces: [UInt64], edgeBits: Int = edgeBits) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: (nonces.count * edgeBits + 7) / 8)
        for (i, n) in nonces.enumerated() {
            for bit in 0..<edgeBits where (n >> bit) & 1 == 1 {
                let p = i * edgeBits + bit
                out[p >> 3] |= 1 << (p & 7)
            }
        }
        return out
    }

    public static func unpack(_ bytes: [UInt8], count: Int = cycleLength, edgeBits: Int = edgeBits) -> [UInt64] {
        (0..<count).map { i in
            var n: UInt64 = 0
            for bit in 0..<edgeBits {
                let p = i * edgeBits + bit
                if bytes[p >> 3] >> (p & 7) & 1 == 1 { n |= 1 << bit }
            }
            return n
        }
    }

    /// The PoW hash Tari scores: Blake2b-256 over the packed proof.
    public static func powHash(_ nonces: [UInt64]) -> [UInt8] {
        Blake2b.hash(pack(nonces))
    }

    /// Achieved difficulty: min((2^256 - 1) / H, u64::MAX), H read big-endian.
    public static func difficulty(powHash h: [UInt8]) -> UInt64 {
        precondition(h.count == 32)
        // Big-endian limbs, most significant first.
        let limbs = (0..<4).map { l in (0..<8).reduce(UInt64(0)) { $0 << 8 | UInt64(h[l * 8 + $1]) } }
        if limbs == [0, 0, 0, 0] { return .max }
        // If H < 2^192 the quotient is at least 2^64, which caps.
        if limbs[0] == 0 { return .max }
        // Quotient fits in 64 bits: binary long division of (2^256 - 1) by H,
        // tracking a 256-bit remainder. Only 64 quotient bits can be set.
        var q: UInt64 = 0
        var rem: [UInt64] = [0, 0, 0, 0]
        // Bring down the 192 high bits of the dividend first (all ones); they
        // are < H since H >= 2^192, so they produce no quotient bits.
        rem = [0, .max, .max, .max]
        for _ in 0..<64 {
            // rem = rem << 1 | 1
            let carry = rem[0] >> 63
            rem[0] = rem[0] << 1 | rem[1] >> 63
            rem[1] = rem[1] << 1 | rem[2] >> 63
            rem[2] = rem[2] << 1 | rem[3] >> 63
            rem[3] = rem[3] << 1 | 1
            q <<= 1
            if carry == 1 || !less(rem, limbs) {
                rem = sub(rem, limbs)
                q |= 1
            }
        }
        return q
    }

    public static func difficulty(nonces: [UInt64]) -> UInt64 { difficulty(powHash: powHash(nonces)) }

    static func less(_ a: [UInt64], _ b: [UInt64]) -> Bool {
        for i in 0..<4 where a[i] != b[i] { return a[i] < b[i] }
        return false
    }

    static func sub(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
        var out = [UInt64](repeating: 0, count: 4)
        var borrow: UInt64 = 0
        for i in (0..<4).reversed() {
            let (d1, o1) = a[i].subtractingReportingOverflow(b[i])
            let (d2, o2) = d1.subtractingReportingOverflow(borrow)
            out[i] = d2
            borrow = (o1 || o2) ? 1 : 0
        }
        return out
    }
}

extension Array where Element == UInt8 {
    public init?(hex: String) {
        let chars = Array(hex.utf8)
        guard chars.count % 2 == 0 else { return nil }
        var out = [UInt8]()
        out.reserveCapacity(chars.count / 2)
        for i in stride(from: 0, to: chars.count, by: 2) {
            guard let b = UInt8(String(decoding: chars[i..<i + 2], as: UTF8.self), radix: 16) else { return nil }
            out.append(b)
        }
        self = out
    }

    public var hex: String { map { String(format: "%02x", $0) }.joined() }
}
