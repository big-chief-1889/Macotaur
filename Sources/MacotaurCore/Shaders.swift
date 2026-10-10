// Metal source for the Cuckaroo mean-trimming solver. It's compiled at runtime
// (MTLDevice.makeLibrary(source:)) so the build needs no Metal toolchain and
// the graph geometry can be injected as preprocessor macros.
//
// Pipeline per graph (C29 geometry, chosen by MetalSolver.plan):
//   seed    siphash every edge, scatter into 4096 buckets by low bits of u
//           (low-memory mode: k passes, each keeping 1/k of the buckets and
//           followed by round 0 on them, so stage 0 needs 1/k the memory)
//   round   per bucket: count key-node degrees in threadgroup memory, keep
//           edges whose key node has degree >= 2 and write them out keyed by
//           the other endpoint (so sides alternate every round). Up to round
//           17, edges are packed into 6 bytes and degrees go in a bitmap over
//           the bucket's node range; from round 18, 2048 buckets of a few
//           thousand edges each count them in a small hash table instead.
//   tail    gather the survivors into one list for the CPU cycle finder
//   recover recompute all edges and pick out the nonces of a found cycle

enum Shaders {
    /// Edges per thread held in registers by the sparse trimming kernel.
    static let sparseEdgesPerThread = 8

    static let source = #"""
#include <metal_stdlib>
using namespace metal;

#define NODEMASK      ((1u << EDGEBITS) - 1)
#define SPARSE_EPT    8
#define NO_EDGE       0xffffffffu

struct SipKeys { ulong k0, k1, k2, k3; };

#define ROTL(x, b) rotate((x), (ulong)(b))
// Rotating by 32 swaps the 32-bit halves and rotating by 16 permutes the
// 16-bit quarters. Written as swizzles (the 16-bit one fused with the xor
// that follows it) they compile to cheaper code than 64-bit rotates.
#define ROT32(x) as_type<ulong>(as_type<uint2>(x).yx)
#define SIPROUND do { \
    v0 += v1; v2 += v3; v1 = ROTL(v1, 13); \
    v1 ^= v0; v3 = as_type<ulong>(as_type<ushort4>(v3).wxyz ^ as_type<ushort4>(v2)); \
    v0 = ROT32(v0); v2 += v1; v0 += v3; \
    v1 = ROTL(v1, 17); v3 = ROTL(v3, 21); \
    v1 ^= v2; v3 ^= v0; v2 = ROT32(v2); \
} while (0)

// Hash the 64-edge block starting at nonce0 into sip[] (pre-xor values).
static inline void sipblock(constant SipKeys& keys, ulong nonce0, thread ulong* sip) {
    ulong v0 = keys.k0, v1 = keys.k1, v2 = keys.k2, v3 = keys.k3;
    for (uint i = 0; i < 64; i++) {
        ulong n = nonce0 + i;
        v3 ^= n;
        SIPROUND; SIPROUND;
        v0 ^= n;
        v2 ^= 0xff;
        SIPROUND; SIPROUND; SIPROUND; SIPROUND;
        sip[i] = (v0 ^ v1) ^ (v2 ^ v3);
    }
}

static inline uint2 blockEdge(thread const ulong* sip, uint i) {
    ulong h = i == 63 ? sip[63] : sip[i] ^ sip[63];
    return uint2(uint(h) & NODEMASK, uint(h >> 32) & NODEMASK);
}

// Pipeline variants, set per pipeline state (see MetalSolver.init).
constant bool OUT_PACKED [[function_constant(0)]];   // seed: write packed edges
constant bool OUT_PAIRS  [[function_constant(1)]];   // trimPacked: write uint2 edges
// Low-memory mode (MetalSolver.Config.seedSplits): the seed keeps one range of
// stage 0's buckets and round 0 trims just that range. Both index the range
// from 0, in the edge buffer and in the counters (bound at the range's first
// counter); only the packed edge keys still need the global bucket. (trimSparse
// runs one group per bucket on whole nodes, so it needs no variant.)
constant bool IN_SLICE   [[function_constant(2)]];

// Early stages store edges in 6 bytes instead of 8, since the seed and the
// big rounds are bound by DRAM bytes.
// An edge (x, y) in the bucket of x's low bits is stored as the 46-bit value
// key | y << kb, where key = x >> bucket bits has kb = EDGEBITS - bucket bits
// bits. Indexes are 32-bit: the host keeps buckets * cap < 2^32.
static inline void storePacked(device packed_ushort3* d, uint j, uint key, uint y, uint kb) {
    uint lo = key | y << kb;
    d[j] = packed_ushort3(ushort(lo), ushort(lo >> 16), ushort(y >> (32 - kb)));
}

// (key, y) of packed edge j.
static inline uint2 loadPacked(device const packed_ushort3* d, uint j, uint kb) {
    packed_ushort3 v = d[j];
    uint lo = uint(v.x) | uint(v.y) << 16;
    return uint2(lo & ((1u << kb) - 1), lo >> kb | uint(v.z) << (32 - kb));
}

// Siphash every edge and write it straight to its bucket (keyed by the low
// bits of u) with one device atomic per edge. Apart from the hashing, the
// cost is the scattered stores, which scales with bytes (hence OUT_PACKED).
// Staging through threadgroup memory needs a second full pass and doesn't
// pay off.
kernel void seed(constant SipKeys& keys      [[buffer(0)]],
                 device uint2* out           [[buffer(1)]],
                 device atomic_uint* counts  [[buffer(2)]],
                 constant uint2& p           [[buffer(3)]],   // x: bucket cap, y: bucket mask
                 constant uint2& slice       [[buffer(4), function_constant(IN_SLICE)]],   // x: first bucket, y: count
                 uint gid    [[thread_position_in_grid]])
{
    ulong sip[64];
    uint bits = popcount(p.y);
    for (uint blk = 0; blk < SEED_BLOCKS; blk++) {
        ulong nonce0 = ((ulong)gid * SEED_BLOCKS + blk) * 64;
        sipblock(keys, nonce0, sip);
        for (uint i = 0; i < 64; i++) {
            uint2 e = blockEdge(sip, i);
            uint b = e.x & p.y;
            if (IN_SLICE) { b -= slice.x; if (b >= slice.y) continue; }   // another pass's bucket
            uint pos = atomic_fetch_add_explicit(&counts[b], 1, memory_order_relaxed);
            if (pos >= p.x) continue;
            if (OUT_PACKED) storePacked((device packed_ushort3*)out, b * p.x + pos, e.x >> bits, e.y, EDGEBITS - bits);
            else out[(ulong)b * p.x + pos] = e;
        }
    }
}

struct RoundParams {
    uint srcCap, dstCap;
    uint inBits;      // input buckets are keyed by the low inBits of edge.x
    uint outBits;     // output buckets: low outBits of edge.y
    uint tableWords;  // two-pass: words per bitmap; sparse: hash slots (power of 2)
    uint inBase, inBuckets;   // IN_SLICE: input buckets [inBase, inBase + inBuckets)
};

// Load this bucket's edges into registers, NO_EDGE-padded. The host keeps
// every bucket's capacity <= threads * EPT.
template <uint EPT>
static inline void loadEdges(thread uint2 (&e)[EPT], device const uint2* src, device const uint* srcCount,
                             constant RoundParams& p, uint bucket, uint tid, uint T) {
    uint n = min(srcCount[bucket], p.srcCap);
    device const uint2* in = src + (ulong)bucket * p.srcCap;
    for (uint j = 0; j < EPT; j++) {
        uint i = tid + j * T;
        e[j] = i < n ? in[i] : uint2(NO_EDGE);
    }
}

static inline void emit(uint2 e, device uint2* dst, device atomic_uint* dstCount, constant RoundParams& p) {
    uint b = e.y & ((1u << p.outBits) - 1);
    uint pos = atomic_fetch_add_explicit(&dstCount[b], 1, memory_order_relaxed);
    if (pos < p.dstCap) dst[(ulong)b * p.dstCap + pos] = e.yx;
}

// Count key-node degrees into two bitmaps ("seen", "seen twice") covering the
// bucket's whole node range, then keep edges whose key was seen twice.
static inline void markKey(uint k, threadgroup atomic_uint* bm, constant RoundParams& p) {
    uint m = 1u << (k & 31);
    uint old = atomic_fetch_or_explicit(&bm[k >> 5], m, memory_order_relaxed);
    if (old & m) atomic_fetch_or_explicit(&bm[p.tableWords + (k >> 5)], m, memory_order_relaxed);
}

static inline bool seenTwiceKey(uint k, threadgroup atomic_uint* bm, constant RoundParams& p) {
    return atomic_load_explicit(&bm[p.tableWords + (k >> 5)], memory_order_relaxed) & (1u << (k & 31));
}

static inline void markBitmap(uint2 e, threadgroup atomic_uint* bm, constant RoundParams& p) { markKey(e.x >> p.inBits, bm, p); }
static inline bool seenTwice(uint2 e, threadgroup atomic_uint* bm, constant RoundParams& p) { return seenTwiceKey(e.x >> p.inBits, bm, p); }

// Persistent grid (the host dispatches fewer groups than buckets): a group
// that finishes a bucket takes the next one from work[0]. Threadgroup memory
// is all bitmap, so tid 0 broadcasts it through the group's slot work[1 + tg].
// The first barrier also keeps the next bucket's bitmap clear behind this
// bucket's pass 2.
static inline uint nextBucket(device atomic_uint* work, uint tg, uint G, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) atomic_store_explicit(&work[1 + tg], G + atomic_fetch_add_explicit(&work[0], 1, memory_order_relaxed), memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_device);
    return atomic_load_explicit(&work[1 + tg], memory_order_relaxed);
}

// Most rounds: count, then filter. Pass 2 re-reads the bucket, which is what
// costs in the big rounds: a bucket times the resident groups overflows the
// few MB the GPU caches re-reads in. Two things make the re-read hit:
//  - pass 2 walks the bucket newest-first, so it starts on the lines pass 1
//    read last (a forward rescan of a region larger than the cache is LRU's
//    worst case);
//  - a persistent grid (see nextBucket) caps the buckets in flight. Taking
//    buckets dynamically rather than striding keeps a group on a busy core
//    from holding up the round.
kernel void trimTwoPass(device const uint2* src       [[buffer(0)]],
                        device const uint* srcCount   [[buffer(1)]],
                        device uint2* dst             [[buffer(2)]],
                        device atomic_uint* dstCount  [[buffer(3)]],
                        constant RoundParams& p       [[buffer(4)]],
                        device atomic_uint* work      [[buffer(5)]],   // persistent grid only
                        threadgroup atomic_uint* bm   [[threadgroup(0)]],
                        uint tg     [[threadgroup_position_in_grid]],
                        uint G      [[threadgroups_per_grid]],
                        uint tid    [[thread_position_in_threadgroup]],
                        uint T      [[threads_per_threadgroup]])
{
    uint buckets = IN_SLICE ? p.inBuckets : 1u << p.inBits;
    for (uint bucket = tg; bucket < buckets; bucket = G == buckets ? buckets : nextBucket(work, tg, G, tid)) {
        for (uint i = tid; i < 2 * p.tableWords; i += T) atomic_store_explicit(&bm[i], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = min(srcCount[bucket], p.srcCap);
        device const uint2* in = src + (ulong)bucket * p.srcCap;
        for (uint i = tid; i < n; i += T) markBitmap(in[i], bm, p);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = tid; k < n; k += T) {
            uint2 e = in[n - 1 - k];
            if (seenTwice(e, bm, p)) emit(e, dst, dstCount, p);
        }
    }
}

// trimTwoPass on packed edges, writing packed edges (or pairs, before the
// sparse rounds).
kernel void trimPacked(device const packed_ushort3* src [[buffer(0)]],
                       device const uint* srcCount      [[buffer(1)]],
                       device packed_ushort3* dst       [[buffer(2)]],   // OUT_PAIRS: uint2 edges
                       device atomic_uint* dstCount     [[buffer(3)]],
                       constant RoundParams& p          [[buffer(4)]],
                       device atomic_uint* work         [[buffer(5)]],   // persistent grid only
                       threadgroup atomic_uint* bm      [[threadgroup(0)]],
                       uint tg     [[threadgroup_position_in_grid]],
                       uint G      [[threadgroups_per_grid]],
                       uint tid    [[thread_position_in_threadgroup]],
                       uint T      [[threads_per_threadgroup]])
{
    uint buckets = IN_SLICE ? p.inBuckets : 1u << p.inBits, kb = EDGEBITS - p.inBits;
    for (uint bucket = tg; bucket < buckets; bucket = G == buckets ? buckets : nextBucket(work, tg, G, tid)) {
        for (uint i = tid; i < 2 * p.tableWords; i += T) atomic_store_explicit(&bm[i], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = min(srcCount[bucket], p.srcCap), base = bucket * p.srcCap;
        uint low = IN_SLICE ? p.inBase + bucket : bucket;   // the key node's low bits
        for (uint i = tid; i < n; i += T) markKey(loadPacked(src, base + i, kb).x, bm, p);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = tid; k < n; k += T) {
            uint2 e = loadPacked(src, base + n - 1 - k, kb);
            if (!seenTwiceKey(e.x, bm, p)) continue;
            uint x = e.x << p.inBits | low;
            if (OUT_PAIRS) { emit(uint2(x, e.y), (device uint2*)dst, dstCount, p); continue; }
            uint b = e.y & ((1u << p.outBits) - 1);
            uint pos = atomic_fetch_add_explicit(&dstCount[b], 1, memory_order_relaxed);
            if (pos < p.dstCap) storePacked(dst, b * p.dstCap + pos, e.y >> p.outBits, x, EDGEBITS - p.outBits);
        }
    }
}

// Late rounds: buckets hold at most a few thousand edges, so clearing a bitmap
// for the whole node range would dominate (and with 2048 buckets it wouldn't
// fit). Instead count degrees in a small open-addressing table:
// slot = (key + 1) << 1 | seenTwice, 0 = empty.
kernel void trimSparse(device const uint2* src       [[buffer(0)]],
                       device const uint* srcCount   [[buffer(1)]],
                       device uint2* dst             [[buffer(2)]],
                       device atomic_uint* dstCount  [[buffer(3)]],
                       constant RoundParams& p       [[buffer(4)]],
                       threadgroup atomic_uint* tbl  [[threadgroup(0)]],
                       uint bucket [[threadgroup_position_in_grid]],
                       uint tid    [[thread_position_in_threadgroup]],
                       uint T      [[threads_per_threadgroup]])
{
    uint mask = p.tableWords - 1;
    for (uint i = tid; i < p.tableWords; i += T) atomic_store_explicit(&tbl[i], 0, memory_order_relaxed);
    uint2 e[SPARSE_EPT];
    loadEdges(e, src, srcCount, p, bucket, tid, T);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Node bits are uniformly random, so the key's low bits are a fine hash.
    for (uint j = 0; j < SPARSE_EPT; j++) {
        if (e[j].x == NO_EDGE) continue;
        uint key = e[j].x >> p.inBits;
        uint tag = (key + 1) << 1;
        for (uint h = key & mask; ; ) {
            uint cur = 0;
            if (atomic_compare_exchange_weak_explicit(&tbl[h], &cur, tag, memory_order_relaxed, memory_order_relaxed)) break;
            if (cur == 0) continue;                       // spurious failure: retry this slot
            if ((cur >> 1) == key + 1) { atomic_fetch_or_explicit(&tbl[h], 1u, memory_order_relaxed); break; }
            h = (h + 1) & mask;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < SPARSE_EPT; j++) {
        if (e[j].x == NO_EDGE) continue;
        uint key = e[j].x >> p.inBits;
        uint h = key & mask, cur;
        while (((cur = atomic_load_explicit(&tbl[h], memory_order_relaxed)) >> 1) != key + 1) h = (h + 1) & mask;
        if (cur & 1) emit(e[j], dst, dstCount, p);
    }
}

kernel void tail(device const uint2* src       [[buffer(0)]],
                 device const uint* srcCount   [[buffer(1)]],
                 device uint2* out             [[buffer(2)]],
                 device atomic_uint* outCount  [[buffer(3)]],
                 constant uint3& p             [[buffer(4)]],   // x: src cap, y: out cap, z: swap to (u,v)
                 uint bucket [[threadgroup_position_in_grid]],
                 uint tid    [[thread_position_in_threadgroup]],
                 uint tgSize [[threads_per_threadgroup]])
{
    uint n = min(srcCount[bucket], p.x);
    device const uint2* in = src + (ulong)bucket * p.x;
    for (uint i = tid; i < n; i += tgSize) {
        uint pos = atomic_fetch_add_explicit(outCount, 1, memory_order_relaxed);
        if (pos < p.y) out[pos] = p.z ? in[i].yx : in[i];
    }
}

// Cycle edges arrive as (u, v); a 4096-bit filter on u skips the exact
// compare for ~99% of edges.
kernel void recover(constant SipKeys& keys     [[buffer(0)]],
                    constant uint2* cycle      [[buffer(1)]],
                    constant uint& cycleLen    [[buffer(2)]],
                    device atomic_uint* found  [[buffer(3)]],
                    device uint* nonces        [[buffer(4)]],
                    uint tid    [[thread_position_in_threadgroup]],
                    uint gid    [[thread_position_in_grid]],
                    uint tgSize [[threads_per_threadgroup]])
{
    threadgroup atomic_uint filter[128];
    for (uint i = tid; i < 128; i += tgSize) atomic_store_explicit(&filter[i], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < cycleLen; i += tgSize) {
        uint f = cycle[i].x & 4095;
        atomic_fetch_or_explicit(&filter[f >> 5], 1u << (f & 31), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong sip[64];
    ulong nonce0 = (ulong)gid * 64;
    sipblock(keys, nonce0, sip);
    for (uint i = 0; i < 64; i++) {
        uint2 e = blockEdge(sip, i);
        uint f = e.x & 4095;
        if (!(atomic_load_explicit(&filter[f >> 5], memory_order_relaxed) & (1u << (f & 31)))) continue;
        for (uint j = 0; j < cycleLen; j++) {
            if (all(cycle[j] == e)) {
                uint pos = atomic_fetch_add_explicit(found, 1, memory_order_relaxed);
                if (pos < 64) nonces[pos] = uint(nonce0 + i);
            }
        }
    }
}
"""#
}
