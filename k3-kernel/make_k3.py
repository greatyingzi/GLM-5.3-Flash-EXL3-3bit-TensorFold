#!/usr/bin/env python3
"""Generate exl3k3.cu / exl3k3.cpp (3-bit, 24-word tiles, mcg) from the family's 4-bit exl3.cu / exl3.cpp.

Only what depends on the tile format changes: words per tile (32 -> TW = 24), the lane decode (the generic
module's LaneMap for K2 = 6, the same states and codebook, so the same values as tensorfold.cuda.exl3.experts),
and the loads of a tile's words. Scheduling, fusion, sums and epilogues are the 4-bit kernels' as they are."""
import re, sys
src, dst_cu, cpp_src, dst_cpp = sys.argv[1:5]
s = open(src).read()

def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, f"expected {count} x {old!r}, found {n}"
    s = s.replace(old, new)

# --- decode helpers: 3-bit lane map and decodes (register words via shuffles, or straight from shared memory)
rep("""// decode_tile's values with fewer integer instructions""", """// ---- 3-bit tiles (tf-3bit-kernel): K2 = 6 half-bits a value, TW = 24 words a 16x16 tile. Lane L's eight states are
// one 64-bit merge of words lo:hi of its tile (tensorfold.cuda.exl3 experts_grouped.cuh Fmt<6> / LaneMap<6>, GV 8):
// the same states and the same mcg values as the generic module, decoded without its runtime width dispatch.
constexpr int TW = 24;
struct Map3 {
    int hi, lo, sh;
};
__device__ __forceinline__ Map3 map3(int lane) {
    const int last_end = 3 * (8 * lane + 8) + 128 * 6;      // Fmt<6>::end(8 lane + 7) + 128 K2
    const int hr = (last_end - 1) >> 5;
    return Map3{hr % TW, (hr + TW - 1) % TW, (hr + 1) * 32 - last_end};
}
__device__ __forceinline__ void decode3(uint32_t whi, uint32_t wlo, int sh, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> sh;
    const uint32_t lo32 = (uint32_t)mm, hi32 = (uint32_t)(mm >> 32);
    // state j = (mm >> (21 - 3 j)) & 0xffff
    b0[0] = mcg2((uint32_t)(mm >> 21) & 0xffffu, (uint32_t)(mm >> 18) & 0xffffu);
    b0[1] = mcg2((uint32_t)(mm >> 15) & 0xffffu, (uint32_t)(mm >> 12) & 0xffffu);
    b1[0] = mcg2((uint32_t)(mm >> 9) & 0xffffu, (lo32 >> 6) & 0xffffu);
    b1[1] = mcg2((lo32 >> 3) & 0xffffu, lo32 & 0xffffu);
    (void)hi32;
}
// a tile's words held one a lane (lane < TW; lanes TW.. hold 0)
__device__ __forceinline__ void decode_tile3(uint32_t w, const Map3& m, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    const uint32_t whi = __shfl_sync(0xffffffffu, w, m.hi);
    const uint32_t wlo = __shfl_sync(0xffffffffu, w, m.lo);
    decode3(whi, wlo, m.sh, b0, b1);
}
__device__ __forceinline__ uint32_t ldg_word(const uint32_t* p, int lane) { return lane < TW ? __ldg(p) : 0u; }

// decode_tile's values with fewer integer instructions""")

# --- grouped_kernel (decode-order and SEQ prompt fallback)
rep("""    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if (PF) {""", """    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * TW + lane;
    const Map3 m3 = map3(lane);

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if (PF) {""")
rep("        for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);\n        {\n            const int k = kt0 * 16;",
    "        for (int i = 0; i < NT; ++i) words[i] = ldg_word(tile + i * TW, lane);\n        {\n            const int k = kt0 * 16;")
rep("""                const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
                for (int i = 0; i < NT; ++i) nw[i] = __ldg(next + i * 32);""", """                const uint32_t* next = tile + (size_t)NTILES * TW;
#pragma unroll
                for (int i = 0; i < NT; ++i) nw[i] = ldg_word(next + i * TW, lane);""")
rep("""                decode_tile(words[i], lane, b0, b1);
                mma16816(acc[i][0], a, b0);
                mma16816(acc[i][1], a, b1);
            }
            if (more) {""", """                decode_tile3(words[i], m3, b0, b1);
                mma16816(acc[i][0], a, b0);
                mma16816(acc[i][1], a, b1);
            }
            if (more) {""")
rep("""            tile += (size_t)NTILES * 32;
        }
    } else {
        for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
            uint32_t words[NT];
#pragma unroll
            for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);""", """            tile += (size_t)NTILES * TW;
        }
    } else {
        for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
            uint32_t words[NT];
#pragma unroll
            for (int i = 0; i < NT; ++i) words[i] = ldg_word(tile + i * TW, lane);""")
rep("""                decode_tile(words[i], lane, b0, b1);
                mma16816(acc[i][0], a, b0);
                mma16816(acc[i][1], a, b1);
            }
            tile += (size_t)NTILES * 32;
        }
    }""", """                decode_tile3(words[i], m3, b0, b1);
                mma16816(acc[i][0], a, b0);
                mma16816(acc[i][1], a, b1);
            }
            tile += (size_t)NTILES * TW;
        }
    }""")

# --- dec_kernel
rep("""    constexpr int NV = NT / 4;                           // LD 1: 16-byte loads a lane a k step""",
    """    constexpr int NV = (NT * TW / 4 + 31) / 32;          // LD 1: 16-byte loads a lane a k step (the last partial)
    constexpr int NU = NT * TW / 4;                      // 16-byte pieces a k step""")
rep("""    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * STEPS) * NTILES + nblock * NT) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next""",
    """    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * STEPS) * NTILES + nblock * NT) * TW;
    const size_t kstep = (size_t)NTILES * TW;            // words from one k tile to the next""")
rep("""            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);""",
    """            for (int v = 0; v < NV; ++v)
                if (32 * v + lane < NU) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);""")
rep("""    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (LD == 1) {""", """    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * TW + lane;
    const Map3 m3 = map3(lane);

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (LD == 1) {""")
rep("""                uint32_t a[4], words[NT];""", """                uint32_t a[4];""")
rep("""                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];""",
    """                for (int v = 0; v < NV; ++v)
                    if (32 * v + lane < NU) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];""")
rep("""                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + PD) * kstep + 128 * v + 4 * lane);""",
    """                    for (int v = 0; v < NV; ++v)
                        if (32 * v + lane < NU) rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + PD) * kstep + 128 * v + 4 * lane);""")
rep("""                for (int i = 0; i < NT; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);""", """                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode3(stage[i * TW + m3.hi], stage[i * TW + m3.lo], m3.sh, b0, b1);   // words from the stage""")
rep("""    for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;""", """    for (int i = 0; i < NT; ++i) words[i] = ldg_word(tile + i * TW, lane);
    {
        const int k = kt0 * 16;""")
rep("""            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < NT; ++i) nw[i] = __ldg(next + i * 32);""", """            const uint32_t* next = tile + (size_t)NTILES * TW;
#pragma unroll
            for (int i = 0; i < NT; ++i) nw[i] = ldg_word(next + i * TW, lane);""")
rep("""            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < NT; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;""", """            decode_tile3(words[i], m3, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < NT; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * TW;""")

# --- prompt kernels: stage TW words a tile, decode straight from shared memory
rep("constexpr int PWORDS = 16 * PKT * 32;          // a stage's trellis words (16 m tiles x PKT k tiles x 32)",
    "constexpr int PWORDS = 16 * PKT * TW;          // a stage's trellis words (16 m tiles x PKT k tiles x TW)")
rep("""        for (int idx = threadIdx.x; idx < 16 * PKT * 8; idx += C::T) {       // 16 B pieces of 128-B tile words
            const int q = idx / (PKT * 8), kk = (idx / 8) % PKT, part = idx % 8;""",
    """        constexpr int PP = TW / 4;                                          // 16 B pieces of a tile's words
        for (int idx = threadIdx.x; idx < 16 * PKT * PP; idx += C::T) {
            const int q = idx / (PKT * PP), kk = (idx / PP) % PKT, part = idx % PP;""")
rep("""            const uint32_t* src = (wm ? T1 : T0) + (((size_t)e * KT + chunk * PKT + kk) * NTILES + nt) * 32 + part * 4;
            cp_async16(wdst + ((size_t)q * PKT + kk) * 32 + part * 4, src, true);""",
    """            const uint32_t* src = (wm ? T1 : T0) + (((size_t)e * KT + chunk * PKT + kk) * NTILES + nt) * TW + part * 4;
            cp_async16(wdst + ((size_t)q * PKT + kk) * TW + part * 4, src, true);""")
rep("""    const int nblocks = (cnt + 15) >> 4;                                            // 16-member blocks in use
""", """    const int nblocks = (cnt + 15) >> 4;                                            // 16-member blocks in use
    const Map3 m3 = map3(lane);
""")
rep("""                decode_tile_p(ws[((size_t)(warp * C::MT + i) * PKT + kk) * 32 + lane], lane, b0, b1);""",
    """                decode3(whi[kk][i], wlo[kk][i], m3.sh, b0, b1);   // words read at the stage's start""")
# the stage's words for this warp's tiles, all read before the first decode (their latency behind each other)
rep("""#pragma unroll
        for (int kk = 0; kk < PKT; ++kk) {
            uint32_t a[C::MT][4];""", """        uint32_t whi[PKT][C::MT], wlo[PKT][C::MT];
#pragma unroll
        for (int kk = 0; kk < PKT; ++kk)
#pragma unroll
            for (int i = 0; i < C::MT; ++i) {
                const uint32_t* tw = ws + ((size_t)(warp * C::MT + i) * PKT + kk) * TW;
                whi[kk][i] = tw[m3.hi];
                wlo[kk][i] = tw[m3.lo];
            }
#pragma unroll
        for (int kk = 0; kk < PKT; ++kk) {
            uint32_t a[C::MT][4];""")


# --- prompt pipeline depth / k width per stage: compile-time knobs (defaults = the 4-bit kernels')
rep("constexpr int PBK = 32;              // k values a shared-memory stage (2 k tiles)",
    "#ifndef K3_PBK\n#define K3_PBK 32\n#endif\n#ifndef K3_PSTAGES\n#define K3_PSTAGES 3\n#endif\nconstexpr int PBK = K3_PBK;          // k values a shared-memory stage")
rep("constexpr int PSTAGES = 3;", "constexpr int PSTAGES = K3_PSTAGES;")

# --- host entry points get their own names (a second extension next to the 4-bit one)
s = re.sub(r"\bexl3_(\w+_cuda)\b", r"exl3k3_\1", s)
assert "decode_tile(" not in s.split("decode_tile_p(uint32_t w")[1], "a 4-bit decode is left"
s = s.replace("// GLM-5.3-Flash's EXL3 routed experts on CUDA:", "// [tf-3bit-kernel] 3-bit (24-word tiles) build of GLM-5.3-Flash's EXL3 routed experts on CUDA:", 1)
open(dst_cu, "w").write(s)

c = open(cpp_src).read()
c = re.sub(r"\bexl3_(\w+_cuda)\b", r"exl3k3_\1", c)
open(dst_cpp, "w").write(c)
print("ok", len(s), len(c))
