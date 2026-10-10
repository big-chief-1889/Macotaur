import Foundation
import Network
import Testing
@testable import MacotaurCore

@Test func targetDifficultyRoundTrip() {
    #expect(Stratum.difficulty(targetHex: Stratum.targetHex(difficulty: 20)) == 20)
    #expect(Stratum.difficulty(targetHex: "ffffffffffffffff") == 1)
    #expect(Stratum.difficulty(targetHex: "0000000000000000") == .max)
    // u64::MAX / 0x0ccccccccccccccc = 20, little-endian on the wire
    #expect(Stratum.difficulty(targetHex: "cccccccccccccc0c") == 20)
    #expect(Stratum.difficulty(targetHex: "abc") == nil)
}

@Test func parseLoginResultWithJob() throws {
    let line = #"{"id":1,"jsonrpc":"2.0","error":null,"result":{"id":"sess-1","job":{"blob":"4dbfee3eb7b9a6a27d2a4a8d754eb77cc5493006945c1246e50e9beb4de5ffa5","job_id":"77","target":"cccccccccccccc0c","height":341600,"xn":"beef"},"extensions":["keepalive"],"status":"OK"}}"#
    guard case let .loginResult(session, job?, keepalive) = try Stratum.parse(line) else {
        Issue.record("not a login result"); return
    }
    #expect(session == "sess-1")
    #expect(keepalive)
    #expect(job.id == "77")
    #expect(job.miningHash == tariMiningHash)
    #expect(job.difficulty == 20)
    #expect(job.height == 341600)
    #expect(job.noncePrefix == 0xbeef_0000_0000_0000)
    #expect(job.nonceMask == 0x0000_ffff_ffff_ffff)
}

@Test func parseJobNotifyAndResponses() throws {
    let job = #"{"jsonrpc":"2.0","method":"job","params":{"blob":"4dbfee3eb7b9a6a27d2a4a8d754eb77cc5493006945c1246e50e9beb4de5ffa5","job_id":"78","target":"cccccccccccccc0c","height":341601}}"#
    guard case let .job(j) = try Stratum.parse(job) else { Issue.record("not a job"); return }
    #expect(j.id == "78" && j.nonceMask == .max && j.noncePrefix == 0)

    #expect(try Stratum.parse(#"{"id":5,"jsonrpc":"2.0","result":true}"#) == .response(id: 5, accepted: true, error: nil))
    #expect(try Stratum.parse(#"{"id":6,"jsonrpc":"2.0","error":null,"result":{"status":"OK"}}"#) == .response(id: 6, accepted: true, error: nil))
    #expect(try Stratum.parse(#"{"id":7,"jsonrpc":"2.0","error":{"code":-1,"message":"Low difficulty share"}}"#)
        == .response(id: 7, accepted: false, error: "Low difficulty share"))
    #expect(throws: Stratum.ParseError.self) { try Stratum.parse("not json") }
    #expect(throws: Stratum.ParseError.self) { try Stratum.parse(#"{"method":"job","params":{"blob":"00","job_id":"1","target":"cccccccccccccc0c"}}"#) }
}

@Test func submitMessageFormat() throws {
    let line = Stratum.submit(requestID: 9, session: "s", jobID: "77", nonce: tariNonce, edges: tariEdges)
    #expect(line.hasSuffix("\n"))
    let obj = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
    #expect(obj["method"] as? String == "submit")
    #expect(obj["id"] as? Int == 9)
    let p = obj["params"] as! [String: Any]
    #expect(p["id"] as? String == "s")
    #expect(p["job_id"] as? String == "77")
    #expect(p["nonce"] as? String == "9d589ed597ed42d0")
    #expect((p["pow"] as? [Int])?.map(UInt64.init) == tariEdges)
    #expect(p["result"] as? String == "06d52b90ccfd4db1a52cc133a46dac0dc4577343c1a27d200865a81d717a60e1")
}

/// Minimal pool on localhost that serves the real Tari block as its job and
/// verifies submitted shares the way a pool would.
final class MockPool: @unchecked Sendable {
    let listener: NWListener
    let queue = DispatchQueue(label: "mockpool")
    let jobLine: String
    var submissions: [(nonce: UInt64, edges: [UInt64], valid: Bool)] = []
    var logins: [String] = []

    init(job: [String: Any]) throws {
        listener = try NWListener(using: .tcp, on: .any)
        let r: [String: Any] = ["id": 1, "jsonrpc": "2.0", "error": NSNull(),
                                "result": ["id": "mock-session", "job": job, "status": "OK"]]
        jobLine = String(decoding: try JSONSerialization.data(withJSONObject: r), as: UTF8.self) + "\n"
        listener.newConnectionHandler = { [unowned self] c in
            c.start(queue: self.queue)
            self.read(c, buffer: Data())
        }
        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { if case .ready = $0 { ready.signal() } }
        listener.start(queue: queue)
        ready.wait()
    }

    var port: UInt16 { listener.port!.rawValue }

    func read(_ c: NWConnection, buffer: Data) {
        c.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [unowned self] data, _, done, _ in
            var buf = buffer + (data ?? Data())
            while let nl = buf.firstIndex(of: 10) {
                let line = buf[buf.startIndex..<nl]
                buf.removeSubrange(buf.startIndex...nl)
                handle(line, c)
            }
            if !done { read(c, buffer: buf) }
        }
    }

    func handle(_ line: Data, _ c: NWConnection) {
        guard let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
              let method = obj["method"] as? String, let p = obj["params"] as? [String: Any] else { return }
        if method == "login" {
            logins.append(p["login"] as? String ?? "")
            c.send(content: Data(jobLine.utf8), completion: .idempotent)
        } else if method == "submit" {
            let nonce = UInt64(p["nonce"] as! String, radix: 16)!
            let edges = (p["pow"] as! [Int]).map(UInt64.init)
            let keys = Tari.sipKeys(miningHash: tariMiningHash, nonce: nonce)
            let valid = (try? Cuckaroo.verify(edges, keys: keys)) != nil
                && Tari.powHash(edges).hex == p["result"] as? String
                && Tari.difficulty(nonces: edges) >= 20
            submissions.append((nonce, edges, valid))
            let reply: [String: Any] = valid
                ? ["id": obj["id"]!, "jsonrpc": "2.0", "error": NSNull(), "result": ["status": "OK"]]
                : ["id": obj["id"]!, "jsonrpc": "2.0", "error": ["code": -1, "message": "invalid share"]]
            c.send(content: try! JSONSerialization.data(withJSONObject: reply) + Data([10]), completion: .idempotent)
        }
    }
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["MACOTAUR_GPU_TESTS"] != nil))
func minerSubmitsRealShareToMockPool() async throws {
    // xn is the full 8-byte nonce of the real block, so the miner can only
    // ever mine that graph and must find its known proof (difficulty 37 >= 20).
    let pool = try MockPool(job: ["blob": tariMiningHash.hex, "job_id": "1", "height": 1,
                                  "target": Stratum.targetHex(difficulty: 20), "xn": Stratum.nonceHex(tariNonce)])
    var opts = PoolClient.Options(host: "127.0.0.1", port: pool.port, user: "wallet.worker")
    opts.reconnectDelay = 0.5
    let solver = try MetalSolver()
    // Mine, stop, then mine again with a fresh Miner on the same solver (what
    // the app does on Stop/Start).
    for round in 0..<2 {
        let miner = Miner(solver: solver, pool: opts)
        miner.dutyCycle = round == 0 ? 1 : 0.5
        miner.start()
        let deadline = Date().addingTimeInterval(30)
        while miner.stats.accepted == 0 && Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
        let s = miner.stats
        miner.stop()
        miner.waitUntilStopped()
        #expect(s.accepted >= 1)
        #expect(s.rejected == 0)
    }
    let subs = pool.queue.sync { pool.submissions }
    #expect(subs.first?.nonce == tariNonce)
    #expect(subs.first?.edges == tariEdges)
    #expect(subs.allSatisfy { $0.valid })
    #expect(pool.queue.sync { pool.logins } == ["wallet.worker", "wallet.worker"])
}
