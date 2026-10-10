import Foundation
import MacotaurCore
import Metal

let usage = """
usage: macotaur-bench [--graphs N] [--rounds R] [--edgebits B] [--stats] [--profile [--profile-only]]
  plan overrides: [--packed-until S] [--late S:BITS,...|none] [--round-threads R:T,...] [--round-groups R:G,...]
                  [--seed-threads T] [--seed-blocks B]
  memory: [--seed-splits K] (default: automatic) [--memory-plan] (print each mode's GPU memory and exit)
  correctness: [--edge-hash N [--key-salt S]] (hash each graph's sorted tail edges and stage totals)
  sustained: [--schedule DUTY:SECONDS,...] (mine in phases at these duty cycles, one line a minute)
"""
setlinebuf(stdout)
let args = Array(CommandLine.arguments.dropFirst())
// Every run reserves the full GPU buffers, so a typo stops here instead.
do {
    let takesValue: Set = ["--graphs", "--rounds", "--edgebits", "--packed-until", "--late", "--round-threads", "--round-groups",
                           "--seed-threads", "--seed-blocks", "--seed-splits", "--edge-hash", "--key-salt", "--schedule"]
    let switches: Set = ["--stats", "--profile", "--profile-only", "--memory-plan"]
    var i = 0
    while i < args.count {
        if args[i] == "--help" || args[i] == "-h" { print(usage); exit(0) }
        if takesValue.contains(args[i]), i + 1 < args.count {
            i += 2
        } else if switches.contains(args[i]) {
            i += 1
        } else {
            FileHandle.standardError.write(Data("unknown or incomplete option \(args[i])\n\n\(usage)\n".utf8))
            exit(2)
        }
    }
}
func option(_ name: String) -> Int? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return Int(args[i + 1])
}
var config = MetalSolver.Config()
if let r = option("--rounds") { config.rounds = r }
if let b = option("--edgebits") { config.edgeBits = b }
if let t = option("--seed-threads") { config.seedThreads = t }
if let b = option("--seed-blocks") { config.seedBlocks = b }
if let p = option("--packed-until") { config.packedUntil = p }
if let k = option("--seed-splits") { config.seedSplits = k }
// "key:value,..." options, e.g. --round-threads 0:256,1:1024 ("none" = empty)
func mapOption(_ name: String) -> [Int: Int]? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    var m: [Int: Int] = [:]
    for kv in args[i + 1].split(separator: ",") {
        let p = kv.split(separator: ":")
        if p.count == 2, let k = Int(p[0]), let v = Int(p[1]) { m[k] = v }
    }
    return m
}
if let m = mapOption("--late") { config.lateBucketBits = m }
if let m = mapOption("--round-threads") { config.roundThreads.merge(m) { $1 } }
if let m = mapOption("--round-groups") { config.roundGroups.merge(m) { $1 } }
let graphs = max(1, option("--graphs") ?? 20)
let showStats = args.contains("--stats")

// --memory-plan: GPU memory per seed split count, and the automatic choice
// for some Mac sizes (assuming Metal recommends ~2/3 of RAM, as on small Macs).
if args.contains("--memory-plan") {
    let gb = { (b: Int) in String(format: "%.2f GB", Double(b) / Double(1 << 30)) }
    for k in 1...4 {
        let m = MetalSolver.memoryPlan(config: config, seedSplits: k)
        print("seed passes \(m.seedSplits): \(gb(m.gpuMemoryBytes))")
    }
    for ram in [8, 16, 24, 32] {
        let bytes = UInt64(ram) << 30
        print("\(ram) GB Mac: \(MetalSolver.automaticSeedSplits(config: config, physicalMemory: bytes, recommendedWorkingSet: bytes * 2 / 3)) pass(es)")
    }
    let here = MetalSolver.memoryPlan(config: config, device: MTLCreateSystemDefaultDevice())
    print("this Mac: \(here.seedSplits) pass(es), \(gb(here.gpuMemoryBytes))")
    exit(0)
}

let solver = try MetalSolver(config: config)

// --edge-hash N [--key-salt S]: for N graphs, an FNV-1a hash of the sorted
// tail edges and of the stage totals. Trimming is exact, so these must match
// across plans and memory modes that keep the same rounds.
if let n = option("--edge-hash") {
    let salt = UInt64(option("--key-salt") ?? 0)
    var worst = 0.0
    for g in 0..<n {
        let keys = Cuckaroo.Keys(UInt64(g) &* 0x9E3779B97F4A7C15 ^ salt, 0xabcdef &+ salt, UInt64(g) &+ 1000, 99 &+ salt)
        let r = solver.trim(keys: keys)
        func fnv(_ values: [UInt64]) -> UInt64 { values.reduce(1469598103934665603) { ($0 ^ $1) &* 1099511628211 } }
        let counts = solver.stageCounts()
        let fill = counts.map { Double($0.max) / Double($0.capacity) }.max()!
        worst = max(worst, fill)
        print(String(format: "graph %d edges %d edgehash %016llx totalshash %016llx worstfill %.4f%@", g, r.edges.count,
                     fnv(r.edges.map { UInt64($0.u) << 32 | UInt64($0.v) }.sorted()), fnv(counts.map { UInt64($0.total) }), fill,
                     counts.contains { $0.max > $0.capacity } ? " OVERFLOW" : ""))
    }
    print(String(format: "worst fill over all graphs: %.4f", worst))
    exit(0)
}

// --schedule "duty:seconds,...": sustained mining in phases, mirroring the
// Miner's loop (2 graphs in flight, CPU cycle search, at most one graph
// every gpuTime / duty below full duty). Prints one line per minute.
if let sched = args.firstIndex(of: "--schedule").map({ args[$0 + 1] }) {
    let phases = sched.split(separator: ",").map { p -> (Double, Double) in
        let kv = p.split(separator: ":"); return (Double(kv[0])!, Double(kv[1])!)
    }
    let lock = NSLock()
    var graphs = 0, cycles = 0, lastGPU = 0.0
    let cpu = DispatchQueue(label: "cycles", qos: .utility, attributes: .concurrent)
    var nonce: UInt64 = 0
    let t0 = Date()
    let thermal = ["nominal", "fair", "serious", "critical"]
    for (i, (duty, seconds)) in phases.enumerated() {
        let phaseStart = Date()
        var lastReport = phaseStart, lastGraphs = lock.withLock { graphs }
        while Date().timeIntervalSince(phaseStart) < seconds {
            nonce &+= 1
            solver.trim(keys: .init(nonce &* 0x9E3779B97F4A7C15, 0x5eed, nonce, 7)) { g in
                cpu.async {
                    let found = CycleFinder.findCycles(g.edges).count
                    lock.withLock { graphs += 1; cycles += found; lastGPU = g.gpuSeconds }
                }
            }
            let gpu = lock.withLock { lastGPU }
            if duty < 1 && gpu > 0 { Thread.sleep(forTimeInterval: gpu / duty) }
            if Date().timeIntervalSince(lastReport) >= 60 {
                let now = Date(), n = lock.withLock { graphs }
                print(String(format: "phase=%d duty=%.2f t=%.0f elapsed=%.0f gps=%.3f thermal=%@",
                             i, duty, now.timeIntervalSince1970, now.timeIntervalSince(phaseStart),
                             Double(n - lastGraphs) / now.timeIntervalSince(lastReport),
                             thermal[ProcessInfo.processInfo.thermalState.rawValue]))
                lastReport = now; lastGraphs = n
            }
        }
    }
    print(String(format: "done: %d graphs, %d cycles in %.0fs", graphs, cycles, Date().timeIntervalSince(t0)))
    exit(0)
}

print("GPU: \(solver.device.name)  buffers: \(String(format: "%.2f", Double(solver.gpuMemoryBytes) / Double(1 << 30))) GB  rounds: \(config.rounds)"
      + "  seed passes: \(solver.seedSplits)\(solver.isLowMemoryMode ? " (low-memory mode)" : "")")

if config.edgeBits == 19 {
    // Grin's cuckaroo19 test vector: must find this exact cycle.
    let keys = Cuckaroo.Keys(0x23796193872092ea, 0xf1017d8a68c4b745, 0xd312bd53d2cd307b, 0x840acce5833ddc52)
    let proofs = solver.solve(keys: keys)
    print("cuckaroo19 vector:", proofs.isEmpty ? "NOT FOUND" : "found \(proofs.count) proof(s): \(proofs[0].prefix(4).map { String($0, radix: 16) })...")
    exit(proofs.isEmpty ? 1 : 0)
}

// Warm-up graph, also used for per-stage fill diagnostics.
let warm = solver.trim(keys: .init(0, 1, 2, 3))
if showStats {
    let plans = solver.roundPlans
    for (i, s) in solver.stageCounts().enumerated() {
        let name = i == 0 ? "seed" : "round \(i - 1)"
        let kernel = i == 0 ? "" : " \(plans[i - 1].kernel)"
        print("\(name.padding(toLength: 9, withPad: " ", startingAt: 0)) \(s.buckets) buckets\(kernel)  total \(s.total)  max \(s.max) / cap \(s.capacity)\(s.max > s.capacity ? "  OVERFLOW" : "")")
    }
}
if args.contains("--profile") {
    // Best of 5 per stage: thermal and scheduling noise is large on a laptop.
    _ = solver.profile(keys: .init(7, 8, 9, 10))
    var prof = solver.profile(keys: .init(7, 8, 9, 10))
    for _ in 0..<4 {
        for (i, p) in solver.profile(keys: .init(7, 8, 9, 10)).enumerated() where p.seconds < prof[i].seconds { prof[i] = p }
    }
    let total = prof.reduce(0) { $0 + $1.seconds }
    var rounds = 0.0
    for p in prof {
        if p.stage.hasPrefix("round") { rounds += p.seconds }
        let r = p.stage.hasPrefix("round ") ? Int(p.stage.dropFirst(6))! : -1
        if r < 6 || [10, 20, 40, 80].contains(r) {
            print(String(format: "  %-10@ %7.2f ms", p.stage as NSString, p.seconds * 1000))
        }
    }
    // Later rounds are too short to compare one by one, so also print sums.
    let sums = [(6, 17), (18, 79)].map { lo, hi in
        let t = prof.filter { $0.stage.hasPrefix("round ") && (lo...hi).contains(Int($0.stage.dropFirst(6))!) }
            .reduce(0) { $0 + $1.seconds }
        return String(format: "r%d-%d %.2f", lo, hi, t * 1000)
    }
    print("  round sums (ms): " + sums.joined(separator: "  "))
    print(String(format: "  all rounds %7.2f ms   total %7.2f ms", rounds * 1000, total * 1000))
    if args.contains("--profile-only") { exit(0) }
}
var bestCPU = Double.infinity
for _ in 0..<5 {
    let cpuStart = Date()
    _ = CycleFinder.findCycles(warm.edges)
    bestCPU = min(bestCPU, Date().timeIntervalSince(cpuStart))
}
print(String(format: "cycle finding on warm-up graph: %.1f ms CPU (best of 5), %d edges after peeling",
             bestCPU * 1000, CycleFinder.peel(warm.edges).count))
print("warm-up: \(warm.edges.count) edges left, \(String(format: "%.1f", warm.gpuSeconds * 1000)) ms GPU")

// Pipelined run: GPU trims graph n+1 while the CPU cycle-finds graph n, at
// the Miner's QoS (search at utility, recovery of found cycles above it).
let lock = NSLock()
var totalCycles = 0, verified = 0, gpuTime = 0.0, leftover = 0
let group = DispatchGroup()
let cycleQueue = DispatchQueue(label: "cycles", qos: .utility, attributes: .concurrent)
let proofQueue = DispatchQueue(label: "proofs", qos: .userInitiated)
let start = Date()
for n in 0..<graphs {
    group.enter()
    let keys = Cuckaroo.Keys(UInt64(n) &* 0x9E3779B97F4A7C15, 0xdeadbeef, UInt64(n), 42)
    solver.trim(keys: keys) { g in
        cycleQueue.async {
            let cycles = CycleFinder.findCycles(g.edges)
            func record(_ proofs: [[UInt64]]) {
                lock.lock()
                totalCycles += proofs.count
                gpuTime += g.gpuSeconds
                leftover += g.edges.count
                for p in proofs where (try? Cuckaroo.verify(p, keys: g.keys)) != nil { verified += 1 }
                lock.unlock()
                group.leave()
            }
            if cycles.isEmpty { record([]) } else { proofQueue.async { record(solver.proofs(for: g, cycles: cycles)) } }
        }
    }
}
group.wait()
let wall = Date().timeIntervalSince(start)
print(String(format: "%d graphs in %.2fs: %.3f graphs/s (GPU %.1f ms/graph, avg %d edges left), %d verified 42-cycles",
             graphs, wall, Double(graphs) / wall, gpuTime / Double(graphs) * 1000, leftover / graphs, verified))
