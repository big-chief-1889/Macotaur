import Foundation

/// Mines the pool's current job on the GPU: picks nonces, trims graphs
/// (two in flight), searches leftovers for 42-cycles on the CPU, and submits
/// proofs that meet the share difficulty.
///
/// A Miner runs once; to resume after stop(), make a new one on the same
/// solver. The Miner keeps its solver, and with it the GPU memory, alive. To
/// give the memory back: stop(), waitUntilStopped(), solver.waitUntilIdle(),
/// then drop both. The last graphs' cycle search still holds the Miner for a
/// few hundred ms after that; the memory is back when the solver is freed.
public final class Miner: @unchecked Sendable {
    public struct Stats: Sendable {
        public var graphs = 0
        public var cycles = 0            // verified 42-cycles, any difficulty
        public var submitted = 0
        public var accepted = 0
        public var rejected = 0
        public var stale = 0             // found for a job whose block was already superseded
        public var graphsPerSecond = 0.0 // rolling, last ~30 graphs
        public var bestDifficulty: UInt64 = 0
        public var job: Stratum.Job?
        public var connected = false
        public var startTime = Date()
        public init() {}
    }

    public enum Event: Sendable {
        case pool(PoolClient.Event)
        /// A verified 42-cycle: submitted when it meets the share difficulty.
        case share(difficulty: UInt64, target: UInt64, submitted: Bool)
        /// A share found for a job whose block was already superseded (not submitted).
        case stale(difficulty: UInt64, target: UInt64)
    }

    public let solver: MetalSolver
    private(set) public var pool: PoolClient!
    private let lock = NSLock()
    private var _stats = Stats()
    private var recentGraphs: [Date] = []
    private var running = false
    private var started = false
    private var thread: Thread?
    private let exited = DispatchSemaphore(value: 0)
    private var _dutyCycle = 1.0
    private var lastGPUSeconds = 0.0
    /// Cycle finding is off the GPU's critical path (a graph's slot is freed
    /// before its edges are handed over), so it runs at utility QoS, where it
    /// draws less from the power budget the GPU shares. Only the rare graph
    /// with a cycle (~2%) hops to `proofQueue` for recovery and submission.
    private let cycleQueue = DispatchQueue(label: "macotaur.cycles", qos: .utility, attributes: .concurrent)
    private let proofQueue = DispatchQueue(label: "macotaur.proofs", qos: .userInitiated, attributes: .concurrent)
    private let onEvent: @Sendable (Event) -> Void

    /// `onEvent` runs on the Miner's own queues (pool events on the pool
    /// client's), so hop to your own queue before touching shared state, and
    /// don't read `pool.currentJob` from it.
    public init(solver: MetalSolver, pool options: PoolClient.Options, onEvent: @escaping @Sendable (Event) -> Void = { _ in }) {
        self.solver = solver
        self.onEvent = onEvent
        pool = PoolClient(options: options) { [weak self] e in self?.poolEvent(e) }
    }

    public var stats: Stats {
        lock.withLock {
            var s = _stats
            s.graphsPerSecond = rate(at: Date())
            return s
        }
    }

    /// Graphs per second over the recent graphs. Once the next graph is
    /// overdue the wait counts too, so the rate falls when mining stalls.
    private func rate(at now: Date) -> Double {
        guard recentGraphs.count > 1, let first = recentGraphs.first, let last = recentGraphs.last else { return 0 }
        let n = Double(recentGraphs.count - 1)
        let interval = last.timeIntervalSince(first) / n
        let span = (now.timeIntervalSince(last) > 2 * interval ? now : last).timeIntervalSince(first)
        return span > 0 ? n / span : 0
    }

    /// Fraction of time the GPU is kept busy (0.01...1). Below 1 the miner
    /// idles between graphs, trading speed for heat and power.
    public var dutyCycle: Double {
        get { lock.withLock { _dutyCycle } }
        set { lock.withLock { _dutyCycle = min(1, max(0.01, newValue)) } }
    }

    /// Does nothing after the first call: one start, one mining thread.
    public func start() {
        let first = lock.withLock {
            guard !started else { return false }
            started = true
            running = true
            _stats.startTime = Date()
            return true
        }
        guard first else { return }
        pool.start()
        let t = Thread { [weak self] in self?.loop() }
        t.name = "macotaur.miner"
        t.qualityOfService = .userInitiated
        thread = t
        t.start()
    }

    public func stop() {
        lock.withLock { running = false }
        pool.stop()
    }

    /// Block until the mining thread has exited (in-flight graphs may still
    /// finish on the GPU afterwards; their shares are dropped).
    public func waitUntilStopped() {
        guard thread != nil else { return }
        exited.wait()
        exited.signal()
    }

    private func poolEvent(_ e: PoolClient.Event) {
        lock.withLock {
            switch e {
            case .loggedIn: _stats.connected = true
            case .disconnected, .loginFailed: _stats.connected = false; _stats.job = nil
            case .newJob(let j): _stats.job = j
            case .shareResult(let ok, _): if ok { _stats.accepted += 1 } else { _stats.rejected += 1 }
            default: break
            }
        }
        onEvent(.pool(e))
    }

    private func loop() {
        defer { exited.signal() }
        // Random start so restarts and multiple rigs don't repeat nonces when
        // the pool gives no extranonce.
        var counter = UInt64.random(in: 0 ... .max)
        while lock.withLock({ running }) {
            guard let job = pool.currentJob else {
                Thread.sleep(forTimeInterval: 0.1)
                continue
            }
            let nonce = job.noncePrefix | (counter & job.nonceMask)
            counter &+= 1
            let keys = Tari.sipKeys(miningHash: job.miningHash, nonce: nonce)
            // Blocks while two graphs are already queued on the GPU.
            solver.trim(keys: keys) { [weak self] graph in
                guard let self else { return }
                self.lock.withLock { self.lastGPUSeconds = graph.gpuSeconds }
                self.cycleQueue.async { self.process(graph, job: job, nonce: nonce) }
            }
            // The graph just queued runs during this wait, so the wait spans
            // its GPU time plus the rest: gpu / duty in all.
            let (duty, gpu) = lock.withLock { (_dutyCycle, lastGPUSeconds) }
            if duty < 1 && gpu > 0 { rest(gpu / duty) }
        }
    }

    /// Sleeps in short steps, so a long rest doesn't hold up stop().
    private func rest(_ seconds: TimeInterval) {
        let end = Date().addingTimeInterval(seconds)
        while lock.withLock({ running }) {
            let left = end.timeIntervalSinceNow
            guard left > 0 else { return }
            Thread.sleep(forTimeInterval: min(left, 0.25))
        }
    }

    private func process(_ graph: MetalSolver.Graph, job: Stratum.Job, nonce: UInt64) {
        let cycles = CycleFinder.findCycles(graph.edges)
        lock.withLock {
            _stats.graphs += 1
            recentGraphs.append(Date())
            if recentGraphs.count > 30 { recentGraphs.removeFirst(recentGraphs.count - 30) }
        }
        guard !cycles.isEmpty else { return }
        proofQueue.async { self.submit(self.solver.proofs(for: graph, cycles: cycles), job: job, nonce: nonce) }
    }

    private func submit(_ proofs: [[UInt64]], job: Stratum.Job, nonce: UInt64) {
        lock.withLock { _stats.cycles += proofs.count }
        for edges in proofs {
            let diff = Tari.difficulty(nonces: edges)
            lock.withLock { _stats.bestDifficulty = max(_stats.bestDifficulty, diff) }
            guard diff >= job.difficulty else {
                onEvent(.share(difficulty: diff, target: job.difficulty, submitted: false))
                continue
            }
            // A share for an older block is worthless; same-height job
            // refreshes are still accepted by pools.
            if let latest = pool.currentJob, latest.height > job.height {
                lock.withLock { _stats.stale += 1 }
                onEvent(.stale(difficulty: diff, target: job.difficulty))
                continue
            }
            pool.submit(job: job, nonce: nonce, edges: edges)
            lock.withLock { _stats.submitted += 1 }
            onEvent(.share(difficulty: diff, target: job.difficulty, submitted: true))
        }
    }
}
