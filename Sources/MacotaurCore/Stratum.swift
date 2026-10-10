import Foundation

/// Tari C29 pool protocol: XMRig-style JSON-RPC, one JSON object per line over
/// TCP (as used by LuckyPool / Kryptex and implemented by TARI.Miner).
///
///   -> {"id":1,"method":"login","params":{"login":"WALLET.worker","pass":"x","agent":..}}
///   <- {"id":1,"result":{"id":SESSION,"job":JOB,"status":"OK"}}
///   <- {"method":"job","params":JOB}
///   -> {"id":N,"method":"submit","params":{"id":SESSION,"job_id":..,"nonce":"<16 hex BE>",
///       "pow":[42 edges],"result":"<blake2b-256 of packed proof>"}}
///   <- {"id":N,"result":true} or {"status":"OK"}; an "error" means rejected
///
/// JOB = {"job_id":..,"blob":"<32-byte mining hash>","target":"<u64 LE hex>",
///        "height":N,"xn":"<optional nonce prefix, high bytes>"}
public enum Stratum {
    public struct Job: Equatable, Sendable {
        public var id: String
        public var miningHash: [UInt8]
        /// Share difficulty: u64::MAX / target.
        public var difficulty: UInt64
        public var height: UInt64
        /// Nonces are `noncePrefix | (counter & nonceMask)`.
        public var noncePrefix: UInt64
        public var nonceMask: UInt64

        public init(id: String, miningHash: [UInt8], difficulty: UInt64, height: UInt64,
                    noncePrefix: UInt64 = 0, nonceMask: UInt64 = .max) {
            self.id = id; self.miningHash = miningHash; self.difficulty = difficulty
            self.height = height; self.noncePrefix = noncePrefix; self.nonceMask = nonceMask
        }
    }

    public enum Message: Equatable, Sendable {
        case loginResult(session: String, job: Job?, keepalive: Bool)
        case job(Job)
        /// Reply to request `id`; `accepted` is nil when it's neither a clear
        /// accept nor reject.
        case response(id: Int, accepted: Bool?, error: String?)
        case error(String)
        case other
    }

    public enum ParseError: Error, Equatable { case notJSON, badJob(String) }

    public static let loginRequestID = 1

    // MARK: Outgoing

    public static func login(user: String, pass: String, agent: String) -> String {
        line(["id": loginRequestID, "jsonrpc": "2.0", "method": "login",
              "params": ["login": user, "pass": pass, "agent": agent]])
    }

    public static func submit(requestID: Int, session: String, jobID: String, nonce: UInt64, edges: [UInt64]) -> String {
        line(["id": requestID, "jsonrpc": "2.0", "method": "submit",
              "params": ["id": session, "job_id": jobID, "nonce": nonceHex(nonce),
                         "pow": edges.map { Int($0) }, "result": Tari.powHash(edges).hex] as [String: Any]])
    }

    public static func keepalive(requestID: Int, session: String) -> String {
        line(["id": requestID, "jsonrpc": "2.0", "method": "keepalived", "params": ["id": session]])
    }

    static func line(_ object: [String: Any]) -> String {
        let data = try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
        return String(decoding: data, as: UTF8.self) + "\n"
    }

    /// Nonce as 8 big-endian bytes in hex.
    public static func nonceHex(_ nonce: UInt64) -> String {
        (0..<8).map { UInt8(truncatingIfNeeded: nonce >> (56 - 8 * $0)) }.hex
    }

    // MARK: Incoming

    public static func parse(_ line: String) throws(ParseError) -> Message {
        guard let data = line.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw .notJSON }

        if let method = obj["method"] as? String {
            if method == "job", let params = obj["params"] as? [String: Any] {
                return .job(try job(params))
            }
            return .other
        }

        let id = (obj["id"] as? NSNumber)?.intValue
        let error = errorText(obj["error"])
        let result = obj["result"]

        if let r = result as? [String: Any], let session = r["id"] as? String, id == loginRequestID || r["job"] != nil {
            let j = try (r["job"] as? [String: Any]).map { params throws(ParseError) in try job(params) }
            let ext = (r["extensions"] as? [String]) ?? []
            return .loginResult(session: session, job: j, keepalive: ext.contains("keepalive"))
        }
        if let id {
            var accepted: Bool?
            if error != nil { accepted = false }
            else if let b = result as? Bool { accepted = b }
            else if let r = result as? [String: Any], (r["status"] as? String)?.uppercased() == "OK" { accepted = true }
            return .response(id: id, accepted: accepted, error: error)
        }
        if let error { return .error(error) }
        return .other
    }

    static func errorText(_ e: Any?) -> String? {
        switch e {
        case nil, is NSNull: return nil
        case let s as String: return s
        case let d as [String: Any]: return (d["message"] as? String) ?? "\(d)"
        case let other?: return "\(other)"
        }
    }

    static func job(_ p: [String: Any]) throws(ParseError) -> Job {
        guard let id = p["job_id"] as? String ?? (p["job_id"] as? NSNumber)?.stringValue else { throw .badJob("missing job_id") }
        guard let blobHex = p["blob"] as? String, let blob = [UInt8](hex: blobHex), blob.count == 32
        else { throw .badJob("blob must be 32 bytes of hex") }
        guard let targetHex = p["target"] as? String, let diff = difficulty(targetHex: targetHex)
        else { throw .badJob("target must be 8 bytes of hex") }
        let height = (p["height"] as? NSNumber)?.uint64Value ?? 0
        var prefix: UInt64 = 0, mask: UInt64 = .max
        if let xn = p["xn"] as? String, !xn.isEmpty {
            guard let bytes = [UInt8](hex: xn), bytes.count <= 8 else { throw .badJob("xn must be at most 8 bytes of hex") }
            let free = 64 - 8 * bytes.count
            let value = bytes.reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
            prefix = free == 64 ? 0 : value << free
            mask = free == 0 ? 0 : (free == 64 ? .max : (1 << free) - 1)
        }
        return Job(id: id, miningHash: blob, difficulty: diff, height: height, noncePrefix: prefix, nonceMask: mask)
    }

    /// Target is a little-endian u64; share difficulty = u64::MAX / target.
    public static func difficulty(targetHex: String) -> UInt64? {
        guard let bytes = [UInt8](hex: targetHex), bytes.count == 8 else { return nil }
        let target = bytes.reversed().reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
        return target == 0 ? .max : max(1, UInt64.max / target)
    }

    public static func targetHex(difficulty: UInt64) -> String {
        let t = UInt64.max / max(1, difficulty)
        return (0..<8).map { UInt8(truncatingIfNeeded: t >> (8 * $0)) }.hex
    }
}
