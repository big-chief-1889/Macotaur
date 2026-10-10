import Foundation
import Network

/// One pool connection: connects, logs in, tracks the current job, submits
/// shares, and reconnects with backoff when the connection drops. All state is
/// confined to `queue`; events are delivered on it too, so handlers must not
/// call `currentJob` (it syncs onto the queue).
public final class PoolClient: @unchecked Sendable {
    public struct Options: Sendable {
        public var host: String
        public var port: UInt16
        public var tls = false
        public var user: String
        public var pass = "x"
        public var agent = "macotaur/0.1"
        public var reconnectDelay: TimeInterval = 5
        public var keepaliveInterval: TimeInterval = 60

        public init(host: String, port: UInt16, user: String) {
            self.host = host; self.port = port; self.user = user
        }
    }

    public enum Event: Sendable {
        case connected
        case loggedIn(session: String)
        case newJob(Stratum.Job)
        case shareResult(accepted: Bool, error: String?)
        case loginFailed(String)
        case disconnected(String)
        case log(String)
    }

    public let options: Options
    let queue = DispatchQueue(label: "macotaur.pool")
    let onEvent: @Sendable (Event) -> Void

    private var connection: NWConnection?
    private var buffer = Data()
    private var session: String?
    private var job: Stratum.Job?
    private var nextRequestID = 2
    private var pendingSubmits: Set<Int> = []
    private var keepaliveEnabled = false
    private var keepaliveTimer: DispatchSourceTimer?
    private var running = false
    private var generation = 0
    static let maxLineBytes = 64 * 1024

    public init(options: Options, onEvent: @escaping @Sendable (Event) -> Void) {
        self.options = options
        self.onEvent = onEvent
    }

    public func start() { queue.async { self.running = true; self.connect() } }

    public func stop() {
        queue.async {
            self.running = false
            self.teardown()
        }
    }

    /// The job shares should currently be computed for.
    public var currentJob: Stratum.Job? { queue.sync { job } }

    /// Submit a share for `job`; dropped (with a log event) if not logged in.
    public func submit(job: Stratum.Job, nonce: UInt64, edges: [UInt64]) {
        queue.async {
            guard let session = self.session, let connection = self.connection else {
                return self.onEvent(.log("share dropped: not logged in"))
            }
            let id = self.nextRequestID
            self.nextRequestID += 1
            self.pendingSubmits.insert(id)
            self.send(Stratum.submit(requestID: id, session: session, jobID: job.id, nonce: nonce, edges: edges), on: connection)
        }
    }

    // MARK: Connection lifecycle (on queue)

    private func connect() {
        guard running else { return }
        generation += 1
        let gen = generation
        let params: NWParameters = options.tls ? .tls : .tcp
        let conn = NWConnection(host: .init(options.host), port: .init(rawValue: options.port)!, using: params)
        connection = conn
        conn.stateUpdateHandler = { [weak self] state in
            guard let self, gen == self.generation else { return }
            switch state {
            case .ready:
                self.onEvent(.connected)
                self.send(Stratum.login(user: self.options.user, pass: self.options.pass, agent: self.options.agent), on: conn)
                self.receive(on: conn, generation: gen)
            case .failed(let e):
                self.dropped("connection failed: \(e)")
            case .waiting(let e):
                self.dropped("connection waiting: \(e)")
            default:
                break
            }
        }
        conn.start(queue: queue)
    }

    private func teardown() {
        generation += 1
        keepaliveTimer?.cancel()
        keepaliveTimer = nil
        connection?.cancel()
        connection = nil
        session = nil
        job = nil
        buffer.removeAll()
        pendingSubmits.removeAll()
    }

    private func dropped(_ reason: String, retry: Bool = true) {
        teardown()
        onEvent(.disconnected(reason))
        guard running, retry else { return }
        queue.asyncAfter(deadline: .now() + options.reconnectDelay) { [weak self] in self?.connect() }
    }

    private func send(_ line: String, on conn: NWConnection) {
        conn.send(content: Data(line.utf8), completion: .contentProcessed { [weak self] error in
            if let error { self?.onEvent(.log("send failed: \(error)")) }
        })
    }

    private func receive(on conn: NWConnection, generation gen: Int) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, complete, error in
            guard let self, gen == self.generation else { return }
            if let data { self.buffer.append(data) }
            while let nl = self.buffer.firstIndex(of: UInt8(ascii: "\n")) {
                let line = String(decoding: self.buffer[self.buffer.startIndex..<nl], as: UTF8.self)
                self.buffer.removeSubrange(self.buffer.startIndex...nl)
                if !line.trimmingCharacters(in: .whitespaces).isEmpty { self.handle(line) }
                guard gen == self.generation else { return }
            }
            if self.buffer.count > Self.maxLineBytes { return self.dropped("pool sent an oversized line") }
            if let error { return self.dropped("receive error: \(error)") }
            if complete { return self.dropped("pool closed the connection") }
            self.receive(on: conn, generation: gen)
        }
    }

    private func handle(_ line: String) {
        let message: Stratum.Message
        do { message = try Stratum.parse(line) } catch {
            onEvent(.log("unparseable pool message (\(error)): \(line.prefix(200))"))
            return
        }
        switch message {
        case .loginResult(let s, let j, let keepalive):
            session = s
            onEvent(.loggedIn(session: s))
            if keepalive { startKeepalive() }
            if let j { setJob(j) }
        case .job(let j):
            setJob(j)
        case .response(let id, let accepted, let error):
            if pendingSubmits.remove(id) != nil {
                onEvent(.shareResult(accepted: accepted ?? false, error: error))
            } else if id == Stratum.loginRequestID, session == nil, accepted == false {
                onEvent(.loginFailed(error ?? "login rejected"))
                dropped("login rejected", retry: true)
            } else if let error {
                onEvent(.log("pool error: \(error)"))
            }
        case .error(let e):
            onEvent(.log("pool error: \(e)"))
        case .other:
            break
        }
    }

    private func setJob(_ j: Stratum.Job) {
        job = j
        onEvent(.newJob(j))
    }

    private func startKeepalive() {
        keepaliveTimer?.cancel()
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + options.keepaliveInterval, repeating: options.keepaliveInterval)
        t.setEventHandler { [weak self] in
            guard let self, let session = self.session, let conn = self.connection else { return }
            let id = self.nextRequestID
            self.nextRequestID += 1
            self.send(Stratum.keepalive(requestID: id, session: session), on: conn)
        }
        t.resume()
        keepaliveTimer = t
    }
}
