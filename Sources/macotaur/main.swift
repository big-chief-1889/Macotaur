import Foundation
import MacotaurCore
import Metal

// macotaur --pool HOST:PORT --wallet ADDRESS [options]; see `usage`.
setlinebuf(stdout)
let version = "0.2.0"
let usage = """
usage: macotaur --pool HOST:PORT --wallet ADDRESS [options]
       macotaur --list-devices [--json]
  --worker NAME         worker name (default: mac)
  --separator S         joins wallet and worker in the login (default "."; Kryptex uses "/")
                        (--login-separator is the same)
  --pass P              pool password (default "x")
  --tls                 connect with TLS
  --intensity N         percent of the time the GPU works, 1-100 (default 100)
  --minutes N           stop after N minutes
  --seed-splits K       seed passes, 1-8: more need less GPU memory but mine slower
                        (default: the fewest that fit this Mac)
  --json                stats as one JSON object per line on stdout, log lines on stderr
  --quiet               log only problems
  --stats-interval S    seconds between stats lines (default 30)
  --list-devices        print the GPU and whether it can mine, then exit
  --version             print the version
pools:
  Kryptex    xtm-c29.kryptex.network:7040   --separator /
  LuckyPool  taric29.luckypool.io:3111
"""

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

let args = Array(CommandLine.arguments.dropFirst())
var values: [String: String] = [:]
var flags: Set<String> = []
do {
    let takesValue: Set = ["--pool", "--wallet", "--worker", "--separator", "--pass", "--intensity", "--minutes", "--seed-splits",
                           "--stats-interval"]
    let switches: Set = ["--tls", "--json", "--quiet", "--list-devices", "--version", "--help", "-h"]
    let aliases = ["--login-separator": "--separator"]
    var i = 0
    while i < args.count {
        let a = aliases[args[i]] ?? args[i]
        if takesValue.contains(a) {
            guard i + 1 < args.count else { fail("\(args[i]) needs a value\n\n\(usage)") }
            values[a] = args[i + 1]
            i += 2
        } else if switches.contains(a) {
            flags.insert(a)
            i += 1
        } else {
            fail("unknown option \(args[i])\n\n\(usage)")
        }
    }
}
if flags.contains("--help") || flags.contains("-h") { print(usage); exit(0) }
if flags.contains("--version") { print("macotaur \(version)"); exit(0) }
var config = MetalSolver.Config()
if let s = values["--seed-splits"] {
    guard let k = Int(s), (1...MetalSolver.maxSeedSplits).contains(k) else { fail("--seed-splits must be 1-\(MetalSolver.maxSeedSplits)") }
    config.seedSplits = k
}
let json = flags.contains("--json")

/// One --list-devices entry. A Mac has one GPU, so the list has one.
struct DeviceLine: Encodable {
    var id: Int
    var name: String
    /// What Metal lets the GPU keep resident, out of the Mac's shared memory.
    var memoryBytes: UInt64
    var requiredBytes: Int
    var seedSplits: Int
    var lowMemory: Bool
    var mineable: Bool
    var reason: String?
}
if flags.contains("--list-devices") {
    var devices: [DeviceLine] = []
    if let device = MTLCreateSystemDefaultDevice() {
        let plan = MetalSolver.memoryPlan(config: config, device: device)
        let fits = UInt64(plan.gpuMemoryBytes) <= device.recommendedMaxWorkingSetSize
        devices.append(DeviceLine(id: 0, name: device.name, memoryBytes: device.recommendedMaxWorkingSetSize,
                                  requiredBytes: plan.gpuMemoryBytes, seedSplits: plan.seedSplits, lowMemory: plan.isLowMemoryMode,
                                  mineable: fits, reason: fits ? nil : "not enough GPU memory"))
    }
    if json {
        let e = JSONEncoder()
        e.outputFormatting = .sortedKeys
        if let data = try? e.encode(["devices": devices]) { print(String(decoding: data, as: UTF8.self)) }
    } else if devices.isEmpty {
        print("no GPU found")
    } else {
        func gb(_ bytes: UInt64) -> String { String(format: "%.1f GB", Double(bytes) / Double(1 << 30)) }
        for d in devices {
            let mode = d.lowMemory ? ", low-memory mode with \(d.seedSplits) seed passes" : ""
            let verdict = d.mineable ? "" : ", can't mine: \(d.reason ?? "unsupported")"
            print("\(d.id): \(d.name), \(gb(d.memoryBytes)) for the GPU, needs \(gb(UInt64(d.requiredBytes)))\(mode)\(verdict)")
        }
    }
    exit(0)
}
guard let poolArg = values["--pool"], let wallet = values["--wallet"] else { fail(usage) }
let parts = poolArg.replacingOccurrences(of: "stratum+tcp://", with: "").split(separator: ":")
guard parts.count == 2, let port = UInt16(parts[1]) else { fail("--pool must be HOST:PORT") }
// A fixed default: the Mac's name would show on the pool's stats page.
let worker = values["--worker"] ?? "mac"
var options = PoolClient.Options(host: String(parts[0]), port: port, user: wallet + (values["--separator"] ?? ".") + worker)
options.tls = flags.contains("--tls")
options.agent = "macotaur/\(version)"
if let p = values["--pass"] { options.pass = p }
func positive(_ name: String) -> Double? {
    guard let s = values[name] else { return nil }
    guard let v = Double(s), v > 0 else { fail("\(name) must be a positive number") }
    return v
}
let minutes = positive("--minutes")
var intensity = 100
if let s = values["--intensity"] {
    guard let n = Int(s), (1...100).contains(n) else { fail("--intensity must be 1-100") }
    intensity = n
}
let statsInterval = positive("--stats-interval") ?? 30

@Sendable func timestamp() -> String {
    let f = DateFormatter()
    f.dateFormat = "HH:mm:ss"
    return f.string(from: Date())
}
/// With --json, stdout carries only the stats objects.
@Sendable func log(_ s: String) {
    let line = "[\(timestamp())] \(s)"
    if json { FileHandle.standardError.write(Data((line + "\n").utf8)) } else { print(line) }
}
/// Routine progress, dropped with --quiet.
let quiet = flags.contains("--quiet")
@Sendable func note(_ s: String) {
    if !quiet { log(s) }
}

let solver: MetalSolver
do { solver = try MetalSolver(config: config) } catch { fail("GPU init failed: \(error)") }
let memory = String(format: "%.1f GB", Double(solver.gpuMemoryBytes) / Double(1 << 30))
let mode = solver.isLowMemoryMode ? " (low-memory mode, \(solver.seedSplits) seed passes)" : ""
let pace = intensity < 100 ? ", intensity \(intensity)%" : ""
note("GPU \(solver.device.name), \(memory)\(mode)\(pace), connecting to \(options.host):\(options.port) as \(worker)")

let miner = Miner(solver: solver, pool: options) { event in
    switch event {
    case .pool(.connected): note("connected")
    case .pool(.loggedIn): note("logged in")
    case .pool(.newJob(let j)): note("new job height \(j.height) diff \(j.difficulty)")
    case .pool(.shareResult(true, _)): note("share accepted")
    case .pool(.shareResult(false, let err)): log("share rejected: \(err ?? "?")")
    case .pool(.loginFailed(let e)): log("login failed: \(e)")
    case .pool(.disconnected(let why)): log("disconnected: \(why); reconnecting")
    case .pool(.log(let s)): log(s)
    case .share(let d, let t, let submitted):
        note(submitted ? "found share diff \(d) (target \(t)), submitting" : "found 42-cycle diff \(d) (target \(t)), below target")
    case .stale(let d, let t):
        note("found 42-cycle diff \(d) (target \(t)), stale")
    }
}
miner.dutyCycle = Double(intensity) / 100
miner.start()

/// One --json line.
struct StatsLine: Encodable {
    var final: Bool
    var uptimeSeconds: Double
    var graphsPerSecond: Double
    var graphs, cycles, submitted, accepted, rejected, stale: Int
    var bestDifficulty: UInt64
    var connected: Bool
    var height: UInt64?
    var difficulty: UInt64?
    var gpuMemoryBytes: Int
    var seedSplits: Int
    var lowMemory: Bool
}
func printStats(final: Bool) {
    let s = miner.stats
    let line = StatsLine(final: final, uptimeSeconds: Date().timeIntervalSince(s.startTime), graphsPerSecond: s.graphsPerSecond,
                         graphs: s.graphs, cycles: s.cycles, submitted: s.submitted, accepted: s.accepted, rejected: s.rejected,
                         stale: s.stale, bestDifficulty: s.bestDifficulty, connected: s.connected, height: s.job?.height,
                         difficulty: s.job?.difficulty, gpuMemoryBytes: solver.gpuMemoryBytes, seedSplits: solver.seedSplits,
                         lowMemory: solver.isLowMemoryMode)
    let e = JSONEncoder()
    e.outputFormatting = .sortedKeys
    if let data = try? e.encode(line), let text = String(data: data, encoding: .utf8) { print(text) }
}

let stopAt = minutes.map { Date().addingTimeInterval($0 * 60) }
func finish() -> Never {
    let s = miner.stats
    miner.stop()
    if json { printStats(final: true) }
    note(String(format: "stopping: %d graphs in %.0fs (%.2f g/s), %d cycles, %d submitted, %d accepted, %d rejected, %d stale",
               s.graphs, Date().timeIntervalSince(s.startTime), Double(s.graphs) / Date().timeIntervalSince(s.startTime),
               s.cycles, s.submitted, s.accepted, s.rejected, s.stale))
    exit(0)
}
// Ctrl-C, or a host app stopping the process.
let signalSources = [SIGINT, SIGTERM].map { sig in
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler { finish() }
    source.resume()
    return source
}

let report = DispatchSource.makeTimerSource(queue: .main)
report.schedule(deadline: .now() + statsInterval, repeating: statsInterval)
report.setEventHandler {
    if json {
        printStats(final: false)
    } else {
        let s = miner.stats
        log(String(format: "%.2f g/s | graphs %d | cycles %d | shares %d/%d accepted | best diff %llu",
                   s.graphsPerSecond, s.graphs, s.cycles, s.accepted, s.submitted, s.bestDifficulty))
    }
    if let stopAt, Date() >= stopAt { finish() }
}
report.resume()
dispatchMain()
