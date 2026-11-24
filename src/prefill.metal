// ================================ batched prefill kernels ================================
// Activations are [T][dim] row-major f32 in the arena A. Included after kernels.metal.

#define GM_BM 64
#define GM_BN 32
#define GM_BK 32

// dequantize 16 consecutive weights (k0 % 16 == 0) of one row into half
// 4 nibbles (one per byte of u, values 0..15) -> 4 exact halves via the 0x6400|n trick (= 1024 + n), minus 1024
inline half4 nib4(uint u) {
    const uint lo = (u & 0xFFu) | ((u & 0xFF00u) << 8), hi = ((u >> 16) & 0xFFu) | ((u >> 8) & 0xFF0000u);
    return half4(as_type<half2>(lo | 0x64006400u), as_type<half2>(hi | 0x64006400u)) - half4(1024.0h);
}

// fast Q4_K dequant of 16 weights: vectorized nibble -> half, w = q*d - m in fp16 FMA (single rounding)
inline void deq16_q4k(device const uchar * row, uint k0, thread half * out) {
    device const block_q4_K * b = (device const block_q4_K *) row + k0 / 256;
    const uint w = k0 % 256, j = w / 32, hp = (w % 32) / 16;
    float sc, m;
    scale_min_k4(j, b->scales, sc, m);
    // fp32 scale/offset, single rounding to half at the end (same numerics as the scalar path)
    const float df = float(b->d) * sc, mf = float(b->dmin) * m;
    const uint4 v = *(device const uint4 *) (b->qs + (j / 2) * 32 + hp * 16);
    const uint sh = (j & 1) * 4;
    thread half4 * o = (thread half4 *) out;
    o[0] = half4(fma(float4(nib4((v.x >> sh) & 0x0F0F0F0Fu)), df, -mf));
    o[1] = half4(fma(float4(nib4((v.y >> sh) & 0x0F0F0F0Fu)), df, -mf));
    o[2] = half4(fma(float4(nib4((v.z >> sh) & 0x0F0F0F0Fu)), df, -mf));
    o[3] = half4(fma(float4(nib4((v.w >> sh) & 0x0F0F0F0Fu)), df, -mf));
}

inline void deq16(device const uchar * row, uint wtype, uint k0, thread half * out) {
    if (wtype == 12) { deq16_q4k(row, k0, out); return; }
    if (wtype == 200) {         // NV4: fp4 * e4m3 scale is exact in half (|w| <= 6 * 448); the expert scale comes later
        device const uchar * b = row + (k0 / 32) * 18;
        const uint hp = (k0 % 32) / 16;
        const float s = e4m3f(b[hp]) * NV4_K;
        device const packed_uchar4 * q4 = (device const packed_uchar4 *) (b + 2);
        thread half4 * o = (thread half4 *) out;
        for (int i = 0; i < 4; i++) { const uchar4 q = uchar4(q4[i]); o[i] = half4(fp4f4(hp ? q >> 4 : q & 0xF) * s); }
        return;
    }
    if (wtype == 201) {         // MX8: e4m3 * a power of two
        device const uchar * b = row + (k0 / 32) * 34;
        const float d = float(*(device const half *) b) * MX8_K;
        device const packed_uchar4 * q4 = (device const packed_uchar4 *) (b + 2 + (k0 % 32));
        thread half4 * o = (thread half4 *) out;
        for (int i = 0; i < 4; i++) o[i] = half4(e4m3f4(uchar4(q4[i])) * d);
        return;
    }
    if (wtype == 8) {           // Q8_0: int8 * d. |q| <= 127 is exact in half, so the half product is the correctly rounded w
        device const block_q8_0 * b = (device const block_q8_0 *) row + k0 / 32;
        const half d = b->d;
        device const packed_char4 * q = (device const packed_char4 *) (b->qs + (k0 % 32));
        thread half4 * o = (thread half4 *) out;
        for (int i = 0; i < 4; i++) o[i] = half4(char4(q[i])) * d;
        return;
    }
    if (wtype == 30) {          // BF16: the upper half of a float
        device const packed_ushort4 * p = (device const packed_ushort4 *) ((device const ushort *) row + k0);
        thread half4 * o = (thread half4 *) out;
        for (int i = 0; i < 4; i++) o[i] = half4(as_type<float4>(uint4(ushort4(p[i])) << 16));
        return;
    }
    if (wtype == 6) {           // Q5_0: (nibble | high bit << 4) - 16, * d; SWAR unpack of 4 weights per word
        device const block_q5_0 * b = (device const block_q5_0 *) row + k0 / 32;
        const half d = b->d;
        const bool hi = (k0 % 32) != 0;
        // (qs is only 2-byte aligned: load it as ushorts)
        const ushort4 qa = ushort4(((device const packed_ushort4 *) b->qs)[0]), qb = ushort4(((device const packed_ushort4 *) b->qs)[1]);
        const uint4 v = uint4(uint(qa.x) | uint(qa.y) << 16, uint(qa.z) | uint(qa.w) << 16, uint(qb.x) | uint(qb.y) << 16, uint(qb.z) | uint(qb.w) << 16);
        const uint hb = ((device const ushort *) b->qh)[hi ? 1 : 0];     // the 16 high bits of this half-block
        const uint sh = hi ? 4 : 0;
        thread half4 * o = (thread half4 *) out;
        for (int i = 0; i < 4; i++) {
            const uint n = (v[i] >> sh) & 0x0F0F0F0Fu;
            const uint h = (((hb >> (4 * i)) & 0xFu) * 0x00204081u) & 0x01010101u;     // bit j -> byte j
            // q - 16 as int8: nibble | 0xF0 when the high bit is clear
            const uint qb = n | ((0x01010101u - h) * 0xF0u);
            o[i] = half4(as_type<char4>(qb)) * d;
        }
        return;
    }
    if (wtype == 12) {          // Q4_K
        device const block_q4_K * b = (device const block_q4_K *) row + k0 / 256;
        const uint w = k0 % 256, j = w / 32, hp = (w % 32) / 16;
        float sc, m;
        scale_min_k4(j, b->scales, sc, m);
        const float d = float(b->d) * sc, dm = float(b->dmin) * m;
        const uint4 q = *(device const uint4 *) (b->qs + (j / 2) * 32 + hp * 16);
        const uint sh = (j & 1) * 4;
        thread const uchar * qb = (thread const uchar *) &q;
        for (int i = 0; i < 16; i++) out[i] = half(d * float((qb[i] >> sh) & 0xF) - dm);
    } else if (wtype == 14) {   // Q6_K
        device const block_q6_K * b = (device const block_q6_K *) row + k0 / 256;
        const uint w = k0 % 256, n = w / 128, g = (w % 128) / 32, l0 = w % 32;
        const float d = float(b->d) * float(b->scales[8 * n + l0 / 16 + 2 * g]);
        device const uchar * ql = b->ql + 64 * n + (g & 1) * 32 + l0;
        device const uchar * qh = b->qh + 32 * n + l0;
        const uint s4 = (g >> 1) * 4, s2 = 2 * g;
        for (int i = 0; i < 16; i++) out[i] = half(d * float(int(((ql[i] >> s4) & 0xF) | (((qh[i] >> s2) & 3) << 4)) - 32));
    } else if (wtype == 8) {    // Q8_0
        device const block_q8_0 * b = (device const block_q8_0 *) row + k0 / 32;
        const float d = b->d;
        device const char * q = b->qs + (k0 % 32);
        for (int i = 0; i < 16; i++) out[i] = half(d * float(q[i]));
    } else if (wtype == 6) {    // Q5_0
        device const block_q5_0 * b = (device const block_q5_0 *) row + k0 / 32;
        const float d = b->d;
        const uint qh = uint(b->qh[0]) | (uint(b->qh[1]) << 8) | (uint(b->qh[2]) << 16) | (uint(b->qh[3]) << 24);
        const bool hi = (k0 % 32) != 0;
        for (int i = 0; i < 16; i++) {
            const uint q = hi ? (b->qs[i] >> 4) | (((qh >> (i + 16)) & 1) << 4) : (b->qs[i] & 0xF) | (((qh >> i) & 1) << 4);
            out[i] = half(d * float(int(q) - 16));
        }
    } else {                    // BF16
        device const ushort * p = (device const ushort *) row + k0;
        for (int i = 0; i < 16; i++) out[i] = half(bf16f(p[i]));
    }
}

struct GSeg {
    uint w_lo, w_hi, row_bytes, wtype;
    uint nrows, K, tile_start, in_off;
    uint out_off, ldo, wsrc, in2;    // in2 != 0: columns [K/2, K) come from a second [T][K/2] input
};
struct GemmArgs { uint nseg, T, pad0, pad1; GSeg seg[10]; };

// core tile loop: C[32 tokens x 64 rows] += X[tokens][K] * W[rows][K]^T
// xrow(t) gives the device pointer to token t's input row (or null for padding)
struct GemmTile {
    simdgroup_float8x8 mc[8];
};

// Y[t][n] = sum_k X[t][k] W[n][k]; grid: (row tiles over all segments, token tiles); 128 threads
kernel void k_gemm(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                   constant GemmArgs & args [[buffer(5)]],
                   uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half sa[GM_BM * GM_BK];
    threadgroup half sb[GM_BN * GM_BK];
    threadgroup float sc[GM_BN * GM_BM];
    uint si = 0;
    while (si + 1 < args.nseg && tgp.x >= args.seg[si + 1].tile_start) si++;
    constant GSeg & S = args.seg[si];
    const uint r0 = (tgp.x - S.tile_start) * GM_BM, t0 = tgp.y * GM_BN;
    const uint T = args.T, K = S.K;
    device const uchar * wb = (S.wsrc ? PB : W) + (ulong(S.w_hi) << 32 | S.w_lo);
    const uint wr = tid / 2, wh = tid % 2;
    const uint xt = tid / 4, xk = (tid % 4) * 8;
    const bool wvalid = r0 + wr < S.nrows, xvalid = t0 + xt < T;
    device const uchar * wrow = wb + (ulong) min(r0 + wr, S.nrows - 1) * S.row_bytes;
    // split input (in2 != 0): columns [K/2, K) come from a second [T][K/2] array
    const uint Kx = S.in2 ? K / 2 : K, xtc = min(t0 + xt, T - 1);
    device const float * xrow = A + S.in_off + (ulong) xtc * Kx;
    device const float * xrow2 = A + S.in2 + (ulong) xtc * Kx;
    simdgroup_float8x8 mc[8];
    for (int i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += GM_BK) {
        half wv[16];
        deq16(wrow, S.wtype, k0 + 16 * wh, wv);
        for (int i = 0; i < 16; i++) sa[wr * GM_BK + 16 * wh + i] = wvalid ? wv[i] : half(0);
        device const float * xs = k0 < Kx ? xrow + k0 + xk : xrow2 + (k0 - Kx) + xk;
        const float4 x0 = *(device const float4 *) xs, x1 = *(device const float4 *) (xs + 4);
        for (int i = 0; i < 4; i++) { sb[xt * GM_BK + xk + i] = xvalid ? half(x0[i]) : half(0); sb[xt * GM_BK + xk + 4 + i] = xvalid ? half(x1[i]) : half(0); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < GM_BK / 8; kk++) {
            simdgroup_half8x8 ma[4], mb[2];
            for (int t = 0; t < 4; t++) simdgroup_load(ma[t], sb + t * 8 * GM_BK + kk * 8, GM_BK);
            for (int r = 0; r < 2; r++) simdgroup_load(mb[r], sa + (sg * 16 + r * 8) * GM_BK + kk * 8, GM_BK, ulong2(0, 0), true);
            for (int t = 0; t < 4; t++) for (int r = 0; r < 2; r++) simdgroup_multiply_accumulate(mc[t * 2 + r], ma[t], mb[r], mc[t * 2 + r]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int t = 0; t < 4; t++) for (int r = 0; r < 2; r++)
        simdgroup_store(mc[t * 2 + r], sc + (t * 8) * GM_BM + sg * 16 + r * 8, GM_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < GM_BN * GM_BM; i += 128) {
        const uint t = i / GM_BM, r = i % GM_BM;
        if (t0 + t < T && r0 + r < S.nrows) A[S.out_off + (t0 + t) * S.ldo + r0 + r] = sc[i];
    }
}

// ---- MoE grouped GEMM over a tile list (expert, assignment start, count<=32).
struct MoeGemmArgs {
    uint w_lo, w_hi, expert_bytes, row_bytes;
    uint wtype, nrows, K, x_off;
    uint gather, perm_off, tiles_off, ntiles_off;
    uint out_off, ldo, situ, gs_lo;
    float beta, lbeta; uint gs_hi, pad2;
};

kernel void k_moe_gemm(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant MoeGemmArgs & a [[buffer(5)]],
                       uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half sa[GM_BM * GM_BK];
    threadgroup half sb[GM_BN * GM_BK];
    threadgroup float sc[GM_BN * GM_BM];
    const uint ntiles = as_type<uint>(A[a.ntiles_off]);
    if (tgp.y >= ntiles) return;
    device const uint * tl = (device const uint *) (A + a.tiles_off) + tgp.y * 4;
    const uint e = tl[0], as0 = tl[1], cnt = tl[2];
    device const uint * perm = (device const uint *) (A + a.perm_off);
    // row tile: situ -> 32 gate rows (i0..i0+31) + 32 up rows (FF+i0..); else 64 consecutive rows
    const uint wr = tid / 2, wh = tid % 2;
    uint row;
    if (a.situ) row = (wr < 32) ? tgp.x * 32 + wr : FF + tgp.x * 32 + (wr - 32);
    else row = tgp.x * GM_BM + wr;
    device const uchar * wrow = W + (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) e * a.expert_bytes + (ulong) row * a.row_bytes;
    const uint xt = tid / 4, xk = (tid % 4) * 8;
    const bool xvalid = xt < cnt;
    const uint arow = as0 + min(xt, cnt - 1);
    const uint xsrc = a.gather ? perm[arow] : arow;
    device const float * xrow = A + a.x_off + (ulong) xsrc * a.K;
    simdgroup_float8x8 mc[8];
    for (int i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < a.K; k0 += GM_BK) {
        half wv[16];
        deq16(wrow, a.wtype, k0 + 16 * wh, wv);
        for (int i = 0; i < 16; i++) sa[wr * GM_BK + 16 * wh + i] = wv[i];
        const float4 x0 = *(device const float4 *) (xrow + k0 + xk), x1 = *(device const float4 *) (xrow + k0 + xk + 4);
        for (int i = 0; i < 4; i++) { sb[xt * GM_BK + xk + i] = xvalid ? half(x0[i]) : half(0); sb[xt * GM_BK + xk + 4 + i] = xvalid ? half(x1[i]) : half(0); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < GM_BK / 8; kk++) {
            simdgroup_half8x8 ma[4], mb[2];
            for (int t = 0; t < 4; t++) simdgroup_load(ma[t], sb + t * 8 * GM_BK + kk * 8, GM_BK);
            for (int r = 0; r < 2; r++) simdgroup_load(mb[r], sa + (sg * 16 + r * 8) * GM_BK + kk * 8, GM_BK, ulong2(0, 0), true);
            for (int t = 0; t < 4; t++) for (int r = 0; r < 2; r++) simdgroup_multiply_accumulate(mc[t * 2 + r], ma[t], mb[r], mc[t * 2 + r]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int t = 0; t < 4; t++) for (int r = 0; r < 2; r++)
        simdgroup_store(mc[t * 2 + r], sc + (t * 8) * GM_BM + sg * 16 + r * 8, GM_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a.wtype == 200) {
        const float g = gscale_at(W, a.gs_lo, a.gs_hi, e);
        for (uint i = tid; i < GM_BN * GM_BM; i += 128) sc[i] *= g;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (a.situ) {
        for (uint i = tid; i < GM_BN * 32; i += 128) {
            const uint t = i / 32, r = i % 32;
            if (t < cnt) {
                const float gt = sc[t * GM_BM + r], up = sc[t * GM_BM + 32 + r];
                const float act = a.beta * tanh(gt / a.beta) * sigm(gt) * (a.lbeta > 0 ? a.lbeta * tanh(up / a.lbeta) : up);
                A[a.out_off + (as0 + t) * a.ldo + tgp.x * 32 + r] = act;
            }
        }
    } else {
        for (uint i = tid; i < GM_BN * GM_BM; i += 128) {
            const uint t = i / GM_BM, r = i % GM_BM;
            if (t < cnt) A[a.out_off + (as0 + t) * a.ldo + tgp.x * GM_BM + r] = sc[i];
        }
    }
}

// ---- top-k per token (grid T, 32 threads) -> sel[t][16] = ids (as float) + weights
struct TopkBArgs { uint logits_off, bias_off, sel_off, T; float wscale; uint pad0, pad1, pad2; };
kernel void k_topk_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant TopkBArgs & a [[buffer(5)]],
                     uint t [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    device const float * lg = A + a.logits_off + t * NEXP;
    device const float * bias = (device const float *) (PB + a.bias_off);
    float pr[4], sel[4];
    for (int i = 0; i < 4; i++) { const int e = lane * 4 + i; pr[i] = sigm(lg[e]); sel[i] = pr[i] + bias[e]; }
    float wsum = 0, myw = 0; int myid = 0;
    for (int k = 0; k < TOPK; k++) {
        float bv = sel[0]; int bi = 0;
        for (int i = 1; i < 4; i++) if (sel[i] > bv) { bv = sel[i]; bi = i; }
        const float mv = simd_max(bv);
        const int win = simd_min((bv == mv) ? (int) (lane * 4 + bi) : 1 << 20);
        const float pw = simd_shuffle(pr[win % 4], (ushort) (win / 4));
        if ((int) (lane * 4 + bi) == win) sel[bi] = -INFINITY;
        if (lane == (ushort) k) { myid = win; myw = pw; }
        wsum += pw;
    }
    if (lane < TOPK) {
        A[a.sel_off + t * 16 + lane] = (float) myid;
        A[a.sel_off + t * 16 + TOPK + lane] = myw / (wsum + 1e-20f) * a.wscale;
    }
}

// ---- routing: counting sort of (token, slot) assignments by expert + 32-row tile list. 1 TG x 1024 threads.
struct RouteArgs { uint sel_off, T, perm_off, inv_off, tiles_off, ntiles_off, tile, pad1; };
kernel void k_moe_route(device float * A [[buffer(2)]], constant RouteArgs & a [[buffer(5)]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint cnt[NEXP];
    threadgroup uint off[NEXP], toff[NEXP];
    threadgroup atomic_uint fill[NEXP];
    if (tid < NEXP) { atomic_store_explicit(&cnt[tid], 0, memory_order_relaxed); atomic_store_explicit(&fill[tid], 0, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = a.T * TOPK;
    for (uint i = tid; i < n; i += 1024) {
        const uint e = (uint) A[a.sel_off + (i / TOPK) * 16 + (i % TOPK)];
        atomic_fetch_add_explicit(&cnt[e], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint s = 0, ts = 0;
        for (int e = 0; e < NEXP; e++) {
            const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
            off[e] = s; toff[e] = ts; s += c; ts += (c + a.tile - 1) / a.tile;
        }
        A[a.ntiles_off] = as_type<float>(ts);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint * perm = (device uint *) (A + a.perm_off);
    device uint * inv = (device uint *) (A + a.inv_off);
    device uint * tiles = (device uint *) (A + a.tiles_off);
    for (uint i = tid; i < n; i += 1024) {
        const uint t = i / TOPK, k = i % TOPK;
        const uint e = (uint) A[a.sel_off + t * 16 + k];
        const uint slot = off[e] + atomic_fetch_add_explicit(&fill[e], 1, memory_order_relaxed);
        perm[slot] = t;
        inv[i] = slot;
    }
    if (tid < NEXP) {
        const uint c = atomic_load_explicit(&cnt[tid], memory_order_relaxed);
        for (uint j = 0; j < (c + a.tile - 1) / a.tile; j++) {
            device uint * tt = tiles + (toff[tid] + j) * 4;
            tt[0] = tid; tt[1] = off[tid] + j * a.tile; tt[2] = min(a.tile, c - j * a.tile); tt[3] = 0;
        }
    }
}

struct MoeCombArgs { uint yd_off, inv_off, sel_off, out_off; };
// ---- routing for the fp16 path: counting sort of (token, slot) assignments by expert, per-assignment routing weights,
// and tile lists in two classes: class 0 = 8-simdgroup tiles (65..128 tokens), class 1 = 4-simdgroup tiles (1..64 tokens).
// An expert with c tokens gets c/128 full tiles and one tile for the remainder (simdgroups without tokens idle in the kernel).
// Also writes the indirect dispatch sizes (tiles, column blocks, 1) for gate/up and down per class. 1 TG x 1024 threads.
#define NTC 2
struct Route3Args { uint sel_off, T, perm_off, inv_off, aw_off, ind_off, gu_y, dn_y; uint tiles_off[NTC]; uint pad0, pad1; };
kernel void k_moe_route3(device float * A [[buffer(2)]], constant Route3Args & a [[buffer(5)]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint cnt[NEXP];
    threadgroup uint off[NEXP], toff[NTC][NEXP];
    threadgroup atomic_uint fill[NEXP];
    if (tid < NEXP) { atomic_store_explicit(&cnt[tid], 0, memory_order_relaxed); atomic_store_explicit(&fill[tid], 0, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = a.T * TOPK;
    for (uint i = tid; i < n; i += 1024) {
        const uint e = (uint) A[a.sel_off + (i / TOPK) * 16 + (i % TOPK)];
        atomic_fetch_add_explicit(&cnt[e], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint s = 0, ts[NTC] = {0, 0};
        for (int e = 0; e < NEXP; e++) {
            const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed), r = c % 128;
            off[e] = s; s += c;
            toff[0][e] = ts[0]; toff[1][e] = ts[1];
            ts[0] += c / 128 + (r > 64 ? 1 : 0);
            ts[1] += (r > 0 && r <= 64) ? 1 : 0;
        }
        device uint * ind = (device uint *) (A + a.ind_off);
        for (int c = 0; c < NTC; c++) {
            ind[c * 4 + 0] = ts[c]; ind[c * 4 + 1] = a.gu_y; ind[c * 4 + 2] = 1; ind[c * 4 + 3] = ts[c];
            ind[4 * NTC + c * 4 + 0] = ts[c]; ind[4 * NTC + c * 4 + 1] = a.dn_y; ind[4 * NTC + c * 4 + 2] = 1; ind[4 * NTC + c * 4 + 3] = ts[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint * perm = (device uint *) (A + a.perm_off);
    device uint * inv = (device uint *) (A + a.inv_off);
    for (uint i = tid; i < n; i += 1024) {
        const uint t = i / TOPK, k = i % TOPK;
        const uint e = (uint) A[a.sel_off + t * 16 + k];
        const uint slot = off[e] + atomic_fetch_add_explicit(&fill[e], 1, memory_order_relaxed);
        perm[slot] = t;
        inv[i] = slot;
        A[a.aw_off + slot] = A[a.sel_off + t * 16 + TOPK + k];
    }
    if (tid < NEXP) {
        const uint c = atomic_load_explicit(&cnt[tid], memory_order_relaxed), n128 = c / 128, r = c % 128;
        device uint * t8 = (device uint *) (A + a.tiles_off[0]) + toff[0][tid] * 4;
        for (uint j = 0; j < n128; j++) { t8[j * 4] = tid; t8[j * 4 + 1] = off[tid] + j * 128; t8[j * 4 + 2] = 128; t8[j * 4 + 3] = 0; }
        if (r > 0) {
            device uint * tt = r > 64 ? t8 + n128 * 4 : (device uint *) (A + a.tiles_off[1]) + toff[1][tid] * 4;
            tt[0] = tid; tt[1] = off[tid] + n128 * 128; tt[2] = r; tt[3] = 0;
        }
    }
}

// ---- expert sum per token for the fp16 path (routing weights are already folded into the activations):
// y[t][c] = sum_k ydh[inv[t*8+k]][c], ydh half. grid (T, ceil(E/1024)), 256 threads x 4 columns
kernel void k_moe_comb_h(device float * A [[buffer(2)]], constant MoeCombArgs & a [[buffer(5)]],
                         uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint t = tgp.x, c = (tgp.y * 256 + tid) * 4;
    if (c >= E) return;
    device const uint * inv = (device const uint *) (A + a.inv_off) + t * TOPK;
    device const half * yd = (device const half *) (A + a.yd_off);
    float4 y = 0;
    for (int k = 0; k < TOPK; k++) y += float4(*(device const half4 *) (yd + (ulong) inv[k] * E + c));
    *(device float4 *) (A + a.out_off + t * E + c) = y;
}

// ---- weighted expert sum per token: y[t][c] = sum_k w[t][k] * yd[inv[t*8+k]][c]. grid (E/256, T), 256 threads
kernel void k_moe_comb_b(device float * A [[buffer(2)]], constant MoeCombArgs & a [[buffer(5)]],
                         uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint t = tgp.y, c = tgp.x * 256 + tid;
    device const uint * inv = (device const uint *) (A + a.inv_off);
    float y = 0;
    for (int k = 0; k < TOPK; k++) y += A[a.sel_off + t * 16 + TOPK + k] * A[a.yd_off + inv[t * TOPK + k] * E + c];
    A[a.out_off + t * E + c] = y;
}

// ---- causal short conv over a token sequence with state history, + optional residual. grid (nch/256, T)
// out[t][c] = res[t][c] + (y_t + w0*y_{t-3} + w1*y_{t-2} + w2*y_{t-1} + w3*y_t) / sqrt(5)
struct ConvBArgs { uint y_off, ldy, res_off, out_off, cs_off, convw_off, has_res, pad; };
kernel void k_postconv_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device float * CS [[buffer(4)]],
                         constant ConvBArgs & a [[buffer(5)]], uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint t = tgp.y, c = tgp.x * 256 + tid;
    device const float * h = CS + a.cs_off + c * 3;
    device const float * w = (device const float *) (PB + a.convw_off) + c * 4;
    float x[4];
    for (int j = 0; j < 4; j++) {   // x[j] = y_{t-3+j}
        const int tt = int(t) - 3 + j;
        x[j] = tt >= 0 ? A[a.y_off + tt * a.ldy + c] : h[3 + tt];
    }
    const float o = (x[3] + w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3]) * 0.4472135954999579f;
    A[a.out_off + t * E + c] = (a.has_res ? A[a.res_off + t * E + c] : 0.0f) + o;
}

// ---- conv-state update after a chunk: state <- last 3 raw inputs. grid (nch/256), 256 threads
// In a verify pass (hist_off set) it also records [old state (3 rows), the T inputs] per channel at hist_off, rows NCH
// apart, so a rollback can rebuild the state after any accepted prefix.
#define DNCH (2 * NKV * HD + 2 * E)
struct StateArgs { uint x_off, ldx, cs_off, nch, T, hist_off, pad1, pad2; };
kernel void k_state_upd(device float * A [[buffer(2)]], device float * CS [[buffer(4)]], constant StateArgs & a [[buffer(5)]],
                        uint tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint c = tg * 256 + tid;
    if (c >= a.nch) return;
    device float * h = CS + a.cs_off + c * 3;
    if (a.hist_off != 0xFFFFFFFFu) {
        device float * H = A + a.hist_off + c;
        for (int j = 0; j < 3; j++) H[j * DNCH] = h[j];
        for (uint j = 0; j < a.T; j++) H[(3 + j) * DNCH] = A[a.x_off + j * a.ldx + c];
    }
    float v[3];
    for (int j = 0; j < 3; j++) {   // new h[j] = seq[T + j] where seq = [h0 h1 h2 x0 .. x_{T-1}]
        const int s = int(a.T) + j;
        v[j] = s >= 3 ? A[a.x_off + (s - 3) * a.ldx + c] : h[s];
    }
    h[0] = v[0]; h[1] = v[1]; h[2] = v[2];
}

// ---- K/V row copies between a sliding-window ring (KV, cap keys per group) and the linear prefill scratch (in the arena A).
// The persistent cache of a sliding-window layer holds exactly its window; a prefill batch attends through a scratch copy
// [window history | batch keys] and afterwards leaves the last window of keys in the ring. grid (n, NKV), 32 threads.
struct KvCopyArgs { uint n, pos0, ring_k, ring_v, cap, sk_off, sv_off, scap; int kpos0; uint to_ring, pad0, pad1; };
kernel void k_kvcopy(device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]], constant KvCopyArgs & a [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint pos = a.pos0 + tgp.x, g = tgp.y;
    const uint ri = (g * a.cap + pos % a.cap) * HD + lane * 4, si = (g * a.scap + (uint) ((int) pos - a.kpos0)) * HD + lane * 4;
    device kvw_t * rk = (device kvw_t *) (KV + a.ring_k + ri), * rv = (device kvw_t *) (KV + a.ring_v + ri);
    device kvw_t * sk = (device kvw_t *) ((device kv_t *) (A + a.sk_off) + si), * sv = (device kvw_t *) ((device kv_t *) (A + a.sv_off) + si);
    if (a.to_ring) { *rk = *sk; *rv = *sv; } else { *sk = *rk; *sv = *rv; }
}

// dev: copy 32-bit words from the arena to a side buffer
struct DevCopyArgs { uint src, dst, n, pad; };
kernel void k_devcopy(device const uint * A [[buffer(2)]], device uint * D [[buffer(3)]], constant DevCopyArgs & a [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < a.n) D[a.dst + gid] = A[a.src + gid];
}

// ---- per-token K/V conv + clip/norm + cache write, Q and R preparation. grid (T, NKV), 128 threads.
struct KvPrepArgs {
    uint T, p0, ring, cap, k_base, v_base, cs_off, q_off;
    uint k_off, v_off, r_off; int kdyn_off, vdyn_off; uint qp_off, rp_off, qhalf;
    uint kconv_w, vconv_w, qclip_w, relclip_w; float log_alpha, log_floor; uint is_full; int kpos0;   // kpos0: position of K/V index 0
};
kernel void k_kvprep_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                       device float * CS [[buffer(4)]], constant KvPrepArgs & a [[buffer(5)]],
                       uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4][8];
    const uint t = tgp.x, g = tgp.y, pos = a.p0 + t;
    const float tau = a.is_full ? 1.0f + a.log_alpha * log(max(1.0f, float(pos + 1) / a.log_floor)) : 1.0f;
    device const float * qclip = (device const float *) (PB + a.qclip_w);
    float q[3];
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3 + h, qi = t * (NH * HD) + H * HD + tid;
        q[h] = ((a.qhalf & 2) ? float(((device const half *) (A + a.q_off))[qi]) : A[a.q_off + qi]) * qclip[H * HD + tid];
    }
    const uint c = g * HD + tid;
    float kv[2];
    const float scl = a.kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
    for (int which = 0; which < 2; which++) {
        const uint xo = which == 0 ? a.k_off : a.v_off;
        device const float * hh = CS + a.cs_off + (which * NKV * HD + c) * 3;
        device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_w : a.vconv_w)) + c * 4;
        float x[4];
        for (int j = 0; j < 4; j++) { const int tt = int(t) - 3 + j; x[j] = tt >= 0 ? A[xo + tt * (NKV * HD) + c] : hh[3 + tt]; }
        float acc = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3];
        const int dyn = which == 0 ? a.kdyn_off : a.vdyn_off;
        if (dyn >= 0) { device const float * tp = A + dyn + t * 32 + g * 4; acc += tp[0] * x[3] + tp[1] * x[2] + tp[2] * x[1] + tp[3] * x[0]; }
        kv[which] = (x[3] + acc) * scl;
    }
    float r = 0;
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
        r = A[a.r_off + t * (NH * NREL) + H * NREL + i] * ((device const float *) (PB + a.relclip_w))[H * NREL + i];
    }
    float rn = r * r;
    rn += simd_shuffle_xor(rn, 1); rn += simd_shuffle_xor(rn, 2); rn += simd_shuffle_xor(rn, 4); rn += simd_shuffle_xor(rn, 8);
    {
        const float s0 = simd_sum(q[0] * q[0]), s1 = simd_sum(q[1] * q[1]), s2 = simd_sum(q[2] * q[2]);
        const float s3 = simd_sum(kv[0] * kv[0]), s4 = simd_sum(kv[1] * kv[1]);
        const float s5 = simd_max(fabs(kv[0])), s6 = simd_max(fabs(kv[1]));
        if (lane == 0) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; red[sg][3] = s3; red[sg][4] = s4; red[sg][5] = s5; red[sg][6] = s6; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[5];
    for (int i = 0; i < 5; i++) tot[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];
    for (int h = 0; h < 3; h++) {
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * tau * 0.08838834764831845f;
        if (a.qhalf & 1) ((device half *) (A + a.qp_off))[t * (NH * HD) + (g * 3 + h) * HD + tid] = half(q[h] * f);
        else A[a.qp_off + t * (NH * HD) + (g * 3 + h) * HD + tid] = q[h] * f;
    }
    if (tid < 48) A[a.rp_off + t * (NH * NREL) + (g * 3 + tid / 16) * NREL + tid % 16] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * tau;
    const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f), vf = rsqrt(tot[4] / HD + 1e-6f);
    const float k = kv[0] * kf, v = kv[1] * vf;
    const uint idx = (uint) ((int) pos - a.kpos0);
#ifdef KV8
    const uint ck = kv_code(max(max(red[0][5], red[1][5]), max(red[2][5], red[3][5])) * kf);
    const uint cv = kv_code(max(max(red[0][6], red[1][6]), max(red[2][6], red[3][6])) * vf);
    KV[a.k_base + (g * a.cap + idx) * HD + tid] = (kv_t) (int) kv_quant(k, kv_scale(ck), ck, tid);
    KV[a.v_base + (g * a.cap + idx) * HD + tid] = (kv_t) (int) kv_quant(v, kv_scale(cv), cv, tid);
#else
    KV[a.k_base + (g * a.cap + idx) * HD + tid] = half(k);
    KV[a.v_base + (g * a.cap + idx) * HD + tid] = half(v);
#endif
}

// ---- batched causal attention: TG = (8-token query tile, kv group g) -> 24 query rows. 128 threads.
// keys are read from the KV cache (chunk keys already written). sink = zero logit (init m=0, l=1).
#define AQT 8
struct AttnBArgs {
    uint T, p0, ring, cap, k_base, v_base, window, qp_off;
    uint rp_off, sink_off, gate_off, out_off, relbias_w, ghalf; int kpos0; uint pad2;   // ghalf: half gate array; kpos0: position of K/V index 0
};
kernel void k_attn_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                     constant AttnBArgs & a [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half qs[AQT * 3][HD];
    threadgroup half ks[32][HD];
    threadgroup half vs[32][HD];
    threadgroup float ss[AQT * 3][33];
    threadgroup float rs[AQT * 3][NREL];
    threadgroup float sk[AQT * 3];
    threadgroup float alpha[AQT * 3];
    const uint g = tgp.y, t0 = tgp.x * AQT;
    const uint nq = min((uint) AQT, a.T - t0);
    for (uint i = tid; i < AQT * 3 * HD; i += 128) {
        const uint row = i / HD, d = i % HD, tt = row / 3, h = row % 3;
        qs[row][d] = tt < nq ? half(A[a.qp_off + (t0 + tt) * (NH * HD) + (g * 3 + h) * HD + d]) : half(0);
    }
    if (tid < AQT * 3 * 4) {
        const uint row = tid / 4, part = tid % 4, tt = row / 3, h = row % 3;
        for (int i = 0; i < 4; i++) rs[row][part * 4 + i] = tt < nq ? A[a.rp_off + (t0 + tt) * (NH * NREL) + (g * 3 + h) * NREL + part * 4 + i] : 0.0f;
    }
    if (tid < AQT * 3) { const uint tt = tid / 3, h = tid % 3; sk[tid] = tt < nq ? A[a.sink_off + (t0 + tt) * NH + g * 3 + h] : 0.0f; }
    device const float * relb = (device const float *) (PB + a.relbias_w);
    const uint plast = a.p0 + t0 + nq - 1;
    const uint pfirst = a.p0 + t0;
    const uint kstart = (a.window > 0 && pfirst + 1 > a.window) ? pfirst + 1 - a.window : 0;
    float o[AQT * 3];
    for (int r = 0; r < AQT * 3; r++) o[r] = 0;
    // running stats in registers of every thread (identical copies)
    float m[AQT * 3], l[AQT * 3];
    for (int r = 0; r < AQT * 3; r++) { m[r] = 0.0f; l[r] = 1.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kc = kstart; kc <= plast; kc += 32) {
        const uint nk = min(32u, plast + 1 - kc);
        for (uint i = tid; i < 32 * HD / 4; i += 128) {
            const uint j = i / (HD / 4), d4 = i % (HD / 4);
            const uint p = kc + min(j, nk - 1);
            const uint idx = (uint) ((int) p - a.kpos0);
#ifdef KV8
            device const kv_t * krow = KV + a.k_base + (g * a.cap + idx) * HD, * vrow = KV + a.v_base + (g * a.cap + idx) * HD;
            ((threadgroup half4 *) ks[j])[d4] = half4(float4(char4(*(device const packed_char4 *) (krow + d4 * 4))) * kv_vscale(krow));
            ((threadgroup half4 *) vs[j])[d4] = half4(float4(char4(*(device const packed_char4 *) (vrow + d4 * 4))) * kv_vscale(vrow));
#else
            ((threadgroup half4 *) ks[j])[d4] = *(device const half4 *) (KV + a.k_base + (g * a.cap + idx) * HD + d4 * 4);
            ((threadgroup half4 *) vs[j])[d4] = *(device const half4 *) (KV + a.v_base + (g * a.cap + idx) * HD + d4 * 4);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // scores: 24 rows x 32 keys = 768 -> 6 per thread
        for (uint i = tid; i < AQT * 3 * 32; i += 128) {
            const uint row = i / 32, j = i % 32, tt = row / 3;
            const uint qp = a.p0 + t0 + tt, kp = kc + j;
            float s = -INFINITY;
            if (j < nk && tt < nq && kp <= qp && (a.window == 0 || qp - kp < a.window)) {
                float acc = 0;
                for (int d4 = 0; d4 < HD / 4; d4++)
                    acc += dot(float4(((threadgroup const half4 *) qs[row])[d4]), float4(((threadgroup const half4 *) ks[j])[d4]));
                const uint dist = qp - kp;
                if (dist < EXT) for (int r = 0; r < NREL; r++) acc += rs[row][r] * relb[dist * NREL + r];
                s = acc - sk[row];
            }
            ss[row][j] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax: every thread updates its copy of the row stats; thread-local exp of row/key pairs
        for (int r = 0; r < AQT * 3; r++) {
            float mx = m[r];
            for (uint j = 0; j < nk; j++) mx = max(mx, ss[r][j]);
            const float f = exp(m[r] - mx);
            l[r] *= f; o[r] *= f; m[r] = mx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < AQT * 3 * 32; i += 128) { const uint row = i / 32, j = i % 32; ss[row][j] = j < nk ? exp(ss[row][j] - m[row]) : 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < AQT * 3; r++) {
            float acc = 0, ls = 0;
            for (uint j = 0; j < nk; j++) { const float p = ss[r][j]; ls += p; acc += p * float(vs[j][tid]); }
            l[r] += ls; o[r] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < AQT * 3; r++) {
        const uint tt = r / 3, h = r % 3;
        if (tt < nq) {
            const uint i = (t0 + tt) * (NH * HD) + (g * 3 + h) * HD + tid;
            A[a.out_off + i] = o[r] / l[r] * sigm(A[a.gate_off + i]);
        }
    }
}

// ======================= tensor-core (M5 neural accelerator) GEMM variants via MetalPerformancePrimitives =======================
typedef mpp::tensor_ops::matmul2d<
    mpp::tensor_ops::matmul2d_descriptor(GM_BN, GM_BM, GM_BK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
    execution_simdgroups<4>> GemmOp;

kernel void k_gemm_t(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                     constant GemmArgs & args [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sa[GM_BM * GM_BK];
    threadgroup half sb[GM_BN * GM_BK];
    threadgroup float sc[GM_BN * GM_BM];
    uint si = 0;
    while (si + 1 < args.nseg && tgp.x >= args.seg[si + 1].tile_start) si++;
    constant GSeg & S = args.seg[si];
    const uint r0 = (tgp.x - S.tile_start) * GM_BM, t0 = tgp.y * GM_BN;
    const uint T = args.T, K = S.K;
    device const uchar * wb = (S.wsrc ? PB : W) + (ulong(S.w_hi) << 32 | S.w_lo);
    const uint wr = tid / 2, wh = tid % 2;
    const uint xt = tid / 4, xk = (tid % 4) * 8;
    const bool wvalid = r0 + wr < S.nrows, xvalid = t0 + xt < T;
    device const uchar * wrow = wb + (ulong) min(r0 + wr, S.nrows - 1) * S.row_bytes;
    const uint xr = min(t0 + xt, T - 1);
    const uint KX = S.in2 ? K / 2 : K;
    device const float * xrow = A + S.in_off + (ulong) xr * KX;
    device const float * xrow2 = A + S.in2 + (ulong) xr * KX - KX;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(GM_BK, GM_BM));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(GM_BK, GM_BN));
    GemmOp mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint k0 = 0; k0 < K; k0 += GM_BK) {
        half wv[16];
        deq16(wrow, S.wtype, k0 + 16 * wh, wv);
        device const float * xs = (S.in2 && k0 >= KX) ? xrow2 : xrow;
        const float4 x0 = *(device const float4 *) (xs + k0 + xk), x1 = *(device const float4 *) (xs + k0 + xk + 4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < 16; i++) sa[wr * GM_BK + 16 * wh + i] = wvalid ? wv[i] : half(0);
        for (int i = 0; i < 4; i++) { sb[xt * GM_BK + xk + i] = xvalid ? half(x0[i]) : half(0); sb[xt * GM_BK + xk + 4 + i] = xvalid ? half(x1[i]) : half(0); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm.run(tB, tA, cT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(GM_BM, GM_BN));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < GM_BN * GM_BM; i += 128) {
        const uint t = i / GM_BM, r = i % GM_BM;
        if (t0 + t < T && r0 + r < S.nrows) A[S.out_off + (t0 + t) * S.ldo + r0 + r] = sc[i];
    }
}

kernel void k_moe_gemm_t(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant MoeGemmArgs & a [[buffer(5)]],
                         uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sa[GM_BM * GM_BK];
    threadgroup half sb[GM_BN * GM_BK];
    threadgroup float sc[GM_BN * GM_BM];
    const uint ntiles = as_type<uint>(A[a.ntiles_off]);
    if (tgp.y >= ntiles) return;
    device const uint * tl = (device const uint *) (A + a.tiles_off) + tgp.y * 4;
    const uint e = tl[0], as0 = tl[1], cnt = tl[2];
    device const uint * perm = (device const uint *) (A + a.perm_off);
    const uint wr = tid / 2, wh = tid % 2;
    uint row;
    if (a.situ) row = (wr < 32) ? tgp.x * 32 + wr : FF + tgp.x * 32 + (wr - 32);
    else row = tgp.x * GM_BM + wr;
    device const uchar * wrow = W + (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) e * a.expert_bytes + (ulong) row * a.row_bytes;
    const uint xt = tid / 4, xk = (tid % 4) * 8;
    const bool xvalid = xt < cnt;
    const uint arow = as0 + min(xt, cnt - 1);
    const uint xsrc = a.gather ? perm[arow] : arow;
    device const float * xrow = A + a.x_off + (ulong) xsrc * a.K;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(GM_BK, GM_BM));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(GM_BK, GM_BN));
    GemmOp mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint k0 = 0; k0 < a.K; k0 += GM_BK) {
        half wv[16];
        deq16(wrow, a.wtype, k0 + 16 * wh, wv);
        const float4 x0 = *(device const float4 *) (xrow + k0 + xk), x1 = *(device const float4 *) (xrow + k0 + xk + 4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < 16; i++) sa[wr * GM_BK + 16 * wh + i] = wv[i];
        for (int i = 0; i < 4; i++) { sb[xt * GM_BK + xk + i] = xvalid ? half(x0[i]) : half(0); sb[xt * GM_BK + xk + 4 + i] = xvalid ? half(x1[i]) : half(0); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm.run(tB, tA, cT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(GM_BM, GM_BN));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a.wtype == 200) {
        const float g = gscale_at(W, a.gs_lo, a.gs_hi, e);
        for (uint i = tid; i < GM_BN * GM_BM; i += 128) sc[i] *= g;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (a.situ) {
        for (uint i = tid; i < GM_BN * 32; i += 128) {
            const uint t = i / 32, r = i % 32;
            if (t < cnt) {
                const float gt = sc[t * GM_BM + r], up = sc[t * GM_BM + 32 + r];
                const float act = a.beta * tanh(gt / a.beta) * sigm(gt) * (a.lbeta > 0 ? a.lbeta * tanh(up / a.lbeta) : up);
                A[a.out_off + (as0 + t) * a.ldo + tgp.x * 32 + r] = act;
            }
        }
    } else {
        for (uint i = tid; i < GM_BN * GM_BM; i += 128) {
            const uint t = i / GM_BM, r = i % GM_BM;
            if (t < cnt) A[a.out_off + (as0 + t) * a.ldo + tgp.x * GM_BM + r] = sc[i];
        }
    }
}

// ---- tensor-core flash attention for prefill: TG = (32-query tile, q head), 8 simdgroups, 64-key steps.
// logit = q.k (q pre-scaled) + r.B[dist] - sink ; sink is a zero logit (M=0, L=1 initially). Output gated by sigmoid.
#define FQ 32
#define FCK 64
#define FNSG 8
kernel void k_attn_bt(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                      constant AttnBArgs & a [[buffer(5)]],
                      uint2 tgp [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float region0[FQ * HD];            // 16 KB: [sq half FQ*HD | ss float FQ*FCK]; reused for O at the end
    threadgroup half sp[FQ * FCK];
    threadgroup float sr[FQ];
    threadgroup float rsh[FQ][NREL];
    threadgroup float sk[FQ];
    threadgroup half * sq = (threadgroup half *) region0;
    threadgroup float * ss = region0 + FQ * HD / 2;
    const ushort tid = sgitg * 32 + tiisg;
    const uint h = tgp.y, g = h / 3, t0 = tgp.x * FQ;
    const uint nq = min((uint) FQ, a.T - t0);
    for (uint i = tid; i < FQ * HD / 4; i += FNSG * 32) {
        const uint row = i / (HD / 4), d4 = i % (HD / 4);
        float4 q = 0;
        if (row < nq) q = *(device const float4 *) (A + a.qp_off + (t0 + row) * (NH * HD) + h * HD + d4 * 4);
        ((threadgroup half4 *) sq)[i] = half4(q);
    }
    for (uint i = tid; i < FQ * NREL; i += FNSG * 32) {
        const uint row = i / NREL, r = i % NREL;
        rsh[row][r] = row < nq ? A[a.rp_off + (t0 + row) * (NH * NREL) + h * NREL + r] : 0.0f;
    }
    if (tid < FQ) sk[tid] = tid < nq ? A[a.sink_off + (t0 + tid) * NH + h] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tq = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sq, dextents<int32_t, 2>(HD, FQ));
    auto ts = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(ss, dextents<int32_t, 2>(FCK, FQ));
    auto tp = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sp, dextents<int32_t, 2>(FCK, FQ));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(FQ, FCK, HD, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
                              execution_simdgroups<FNSG>> mm_qk;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(FQ, HD, FCK, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<FNSG>> mm_pv;
    device kv_t * kbase = KV + a.k_base + g * a.cap * HD;
    device kv_t * vbase = KV + a.v_base + g * a.cap * HD;
    auto tv0 = tensor(vbase, dextents<int32_t, 2>(HD, FCK), array<int, 2>({1, HD}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    for (short i = 0; i < co.get_capacity(); ++i) if (co.is_valid_element(i)) co[i] = 0.0f;

    constexpr short NQR = FQ / FNSG;   // rows per simdgroup
    float M[NQR], S[NQR];
    // P is stored in half scaled by e^8 (cancels in O/S) so small probabilities stay out of fp16 subnormals
    for (short jj = 0; jj < NQR; jj++) { M[jj] = 0.0f; S[jj] = 2980.9579870417283f; }   // the sink: logit 0 -> e^8
    device const float * relb = (device const float *) (PB + a.relbias_w);
    const uint pfirst = a.p0 + t0, plast = a.p0 + t0 + nq - 1;
    uint kstart = (a.window > 0 && pfirst + 1 > a.window) ? pfirst + 1 - a.window : 0;
    kstart = kstart / FCK * FCK;
    for (uint kc = kstart; kc <= plast; kc += FCK) {
        const uint idx = (uint) ((int) kc - a.kpos0);
        auto tk = tensor(kbase + idx * HD, dextents<int32_t, 2>(HD, FCK), array<int, 2>({1, HD}));
        mm_qk.run(tq, tk, ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef KV8
        float ksc[FCK / 32], vsc[FCK / 32];      // scales of this lane's keys
        for (short ii = 0; ii < FCK / 32; ii++) { ksc[ii] = kv_vscale(kbase + (idx + ii * 32 + tiisg) * HD); vsc[ii] = kv_vscale(vbase + (idx + ii * 32 + tiisg) * HD); }
#endif
        for (short jj = 0; jj < NQR; jj++) {
            const uint j = jj * FNSG + sgitg;
            const uint qp = a.p0 + t0 + j;
            float s[FCK / 32];
            float m = M[jj];
            for (short ii = 0; ii < FCK / 32; ii++) {
                const uint col = ii * 32 + tiisg, kp = kc + col;
                const bool valid = j < nq && kp <= qp && (a.window == 0 || qp - kp < a.window);
                float v = -INFINITY;
                if (valid) {
#ifdef KV8
                    v = ss[j * FCK + col] * ksc[ii] - sk[j];
#else
                    v = ss[j * FCK + col] - sk[j];
#endif
                    const uint dist = qp - kp;
                    if (dist < EXT) {
                        device const float4 * b4 = (device const float4 *) (relb + dist * NREL);
                        threadgroup const float4 * r4 = (threadgroup const float4 *) rsh[j];
                        v += dot(r4[0], b4[0]) + dot(r4[1], b4[1]) + dot(r4[2], b4[2]) + dot(r4[3], b4[3]);
                    }
                }
                s[ii] = v;
                m = max(m, v);
            }
            m = simd_max(m);
            const float ms = exp(M[jj] - m);
            M[jj] = m;
            float sum = 0;
            for (short ii = 0; ii < FCK / 32; ii++) {
                const half p = half(exp(s[ii] - m + 8.0f));
#ifdef KV8
                sp[j * FCK + ii * 32 + tiisg] = half(float(p) * vsc[ii]);
#else
                sp[j * FCK + ii * 32 + tiisg] = p;
#endif
                sum += float(p);
            }
            S[jj] = S[jj] * ms + simd_sum(sum);
            if (tiisg == 0) sr[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < co.get_capacity(); ++i) if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
        auto tv = tensor(vbase + idx * HD, dextents<int32_t, 2>(HD, FCK), array<int, 2>({1, HD}));
        mm_pv.run(tp, tv, co);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (short jj = 0; jj < NQR; jj++) { const uint j = jj * FNSG + sgitg; if (tiisg == 0) sr[j] = 1.0f / S[jj]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short i = 0; i < co.get_capacity(); ++i) if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
    auto to = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(region0, dextents<int32_t, 2>(HD, FQ));
    co.store(to);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < FQ * HD; i += FNSG * 32) {
        const uint row = i / HD, d = i % HD;
        if (row < nq) {
            const uint o = (t0 + row) * (NH * HD) + h * HD + d;
            A[a.out_off + o] = region0[i] * sigm(A[a.gate_off + o]);
        }
    }
}


// ---- tensor-core flash attention, low-ALU version. TG = (32-query tile, head), 8 simdgroups; grid (T/32, NH).
// Q (half), R and the sinks are read straight from device memory. Per 64-key chunk:
//   S = Q.K^T (tensor), G = R.B[dlo..dlo+95]^T (tensor: relative bias for every distance the tile can see; pair (j, col) reads
//   G[j][dist - dlo], the usual skew), online softmax in ALU (P in half, pre-scaled by e^8), O = O*ms + P.V (tensor, accumulators
//   in registers). Epilogue in registers: O/S * sigmoid(gate) -> half, the input of the output projection GEMM.
#define AQ 32
#define AK 64
#define AG 96
kernel void k_attn_h(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                     constant AttnBArgs & a [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ss[AQ * AK];
    threadgroup float G[AQ * AG];
    threadgroup half sp[AQ * AK];
    threadgroup float sr[AQ];
    const uint h = tgp.y, g = h / 3, t0 = tgp.x * AQ;
    const uint nq = min((uint) AQ, a.T - t0);
    auto tq = tensor((device half *) (A + a.qp_off) + (ulong) t0 * (NH * HD) + h * HD, dextents<int32_t, 2>(HD, (int) nq), array<int, 2>({1, NH * HD}));
    auto tr = tensor(A + a.rp_off + t0 * (NH * NREL) + h * NREL, dextents<int32_t, 2>(NREL, (int) nq), array<int, 2>({1, NH * NREL}));
    auto ts = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(ss, dextents<int32_t, 2>(AK, AQ));
    auto tg = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(G, dextents<int32_t, 2>(AG, AQ));
    auto tp = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sp, dextents<int32_t, 2>(AK, AQ));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AK, HD, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
                              execution_simdgroups<8>> mm_qk;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AG, NREL, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
                              execution_simdgroups<8>> mm_g;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, HD, AK, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<8>> mm_pv;
    device kv_t * kbase = KV + a.k_base + g * a.cap * HD;
    device kv_t * vbase = KV + a.v_base + g * a.cap * HD;
    auto tv0 = tensor(vbase, dextents<int32_t, 2>(HD, AK), array<int, 2>({1, HD}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    for (short i = 0; i < 16; ++i) co[i] = 0.0f;
    // this thread's accumulators: 4 quads of (4 consecutive dims, one query row)
    ushort cd[4], cr[4];
    for (short q = 0; q < 4; ++q) { const auto ix = co.get_multidimensional_index(4 * q); cd[q] = ix[0]; cr[q] = ix[1]; }
    // this simdgroup's query rows: j = jj * 8 + sg
    float M[4], S[4];
    for (short jj = 0; jj < 4; jj++) {
        const uint j = jj * 8 + sg;
        // the sink is one more key with logit sink[j] (running max starts there, its probability e^8 in the scaled sum)
        M[jj] = j < nq ? A[a.sink_off + (t0 + j) * NH + h] : 0.0f; S[jj] = 2980.9579870417283f;
    }
    device const float * relb = (device const float *) (PB + a.relbias_w);
    const uint pfirst = a.p0 + t0, plast = a.p0 + t0 + nq - 1;
    uint kstart = (a.window > 0 && pfirst + 1 > a.window) ? pfirst + 1 - a.window : 0;
    kstart = kstart / AK * AK;
    for (uint kc = kstart; kc <= plast; kc += AK) {
        const uint idx = (uint) ((int) kc - a.kpos0);
        auto tk = tensor(kbase + idx * HD, dextents<int32_t, 2>(HD, AK), array<int, 2>({1, HD}));
        // (a threadgroup-memory destination is far slower than register accumulators + store)
        { auto cs = mm_qk.template get_destination_cooperative_tensor<decltype(tq), decltype(tk), float>(); mm_qk.run(tq, tk, cs); cs.store(ts); }
        const int D0 = int(pfirst) - int(kc);            // dist(j, col) = D0 + j - col
        const int dlo = max(D0 - (AK - 1), 0);
        const bool bias = dlo < EXT;
        if (bias) {
            auto tb = tensor((device float *) relb + dlo * NREL, dextents<int32_t, 2>(NREL, AG), array<int, 2>({1, NREL}));
            auto cg = mm_g.template get_destination_cooperative_tensor<decltype(tr), decltype(tb), float>(); mm_g.run(tr, tb, cg); cg.store(tg);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // interior chunk: every key is visible to every query of the tile and inside the bias table (or entirely beyond it)
        const bool fast = nq == AQ && D0 >= AK - 1 && (a.window == 0 || D0 + AQ - 1 < (int) a.window) && (!bias || D0 + AQ - 1 < EXT);
        const int gofs = D0 - dlo - int(lane);
#ifdef KV8
        // scales of this lane's two keys: K into the scores, V into the probabilities that weight the stored integers
        const float ks0 = kv_vscale(kbase + (idx + lane) * HD), ks1 = kv_vscale(kbase + (idx + 32 + lane) * HD);
        const float vs0 = kv_vscale(vbase + (idx + lane) * HD), vs1 = kv_vscale(vbase + (idx + 32 + lane) * HD);
#endif
        for (short jj = 0; jj < 4; jj++) {
            const uint j = jj * 8 + sg;
#ifdef KV8
            float s0 = ss[j * AK + lane] * ks0, s1 = ss[j * AK + 32 + lane] * ks1;
#else
            float s0 = ss[j * AK + lane], s1 = ss[j * AK + 32 + lane];
#endif
            if (fast) {
                if (bias) { threadgroup const float * gp = G + j * AG + j + gofs; s0 += gp[0]; s1 += gp[-32]; }
            } else {
                const int d0 = D0 + int(j) - int(lane), d1 = d0 - 32;
                if (bias && d0 < EXT) s0 += G[j * AG + clamp(d0 - dlo, 0, AG - 1)];
                if (bias && d1 < EXT) s1 += G[j * AG + clamp(d1 - dlo, 0, AG - 1)];
                if (!(j < nq && d0 >= 0 && (a.window == 0 || d0 < (int) a.window))) s0 = -INFINITY;
                if (!(j < nq && d1 >= 0 && (a.window == 0 || d1 < (int) a.window))) s1 = -INFINITY;
            }
            const float m = simd_max(max(M[jj], max(s0, s1)));
            const float ms = exp(M[jj] - m);
            M[jj] = m;
            const half p0 = half(exp(s0 - m + 8.0f)), p1 = half(exp(s1 - m + 8.0f));
#ifdef KV8
            sp[j * AK + lane] = half(float(p0) * vs0);
            sp[j * AK + 32 + lane] = half(float(p1) * vs1);
#else
            sp[j * AK + lane] = p0;
            sp[j * AK + 32 + lane] = p1;
#endif
            S[jj] = S[jj] * ms + simd_sum(float(p0) + float(p1));
            if (lane == 0) sr[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const float r0 = sr[cr[0]], r1 = sr[cr[1]], r2 = sr[cr[2]], r3 = sr[cr[3]];
            co[0] *= r0; co[1] *= r0; co[2] *= r0; co[3] *= r0;
            co[4] *= r1; co[5] *= r1; co[6] *= r1; co[7] *= r1;
            co[8] *= r2; co[9] *= r2; co[10] *= r2; co[11] *= r2;
            co[12] *= r3; co[13] *= r3; co[14] *= r3; co[15] *= r3;
        }
        auto tv = tensor(vbase + idx * HD, dextents<int32_t, 2>(HD, AK), array<int, 2>({1, HD}));
        mm_pv.run(tp, tv, co);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short jj = 0; jj < 4; jj++) { if (lane == 0) sr[jj * 8 + sg] = 1.0f / S[jj]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half * out = (device half *) (A + a.out_off);
    for (short q = 0; q < 4; ++q) {
        const uint row = cr[q];
        if (row < nq) {
            const uint o = (t0 + row) * (NH * HD) + h * HD + cd[q];
            const float4 gt = a.ghalf ? float4(*(device const half4 *) ((device const half *) (A + a.gate_off) + o)) : *(device const float4 *) (A + a.gate_off + o);
            const float4 v = float4(co[4 * q], co[4 * q + 1], co[4 * q + 2], co[4 * q + 3]) * sr[row] / (1.0f + exp(-gt));
            *(device half4 *) (out + o) = half4(v);
        }
    }
}
// ---- small-T attention for the batched path (inkserve, DSPARK.md): k_attn_h split over the keys, as the decode path
// splits. At n <= 32 queries k_attn_h runs one threadgroup per head over every key; k_attn_hs runs one per (key range,
// head), each range a multiple of 64 keys, and writes partials to part[(s * n + t) * NH + h][HD + 2] = (O, M, S).
// k_attn_hs_comb (grid (n, NH), HD threads) merges them and applies the gate, as k_attn_h's epilogue.
struct AttnHsArgs { AttnBArgs b; uint kps, nsplit, part_off, pad; };
kernel void k_attn_hs(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                     constant AttnHsArgs & ta [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ss[AQ * AK];
    threadgroup float G[AQ * AG];
    threadgroup half sp[AQ * AK];
    threadgroup float sr[AQ];
    constant AttnBArgs & a = ta.b;
    const uint h = tgp.y, g = h / 3, t0 = 0, spl = tgp.x;
    const uint nq = min((uint) AQ, a.T - t0);
    auto tq = tensor((device half *) (A + a.qp_off) + (ulong) t0 * (NH * HD) + h * HD, dextents<int32_t, 2>(HD, (int) nq), array<int, 2>({1, NH * HD}));
    auto tr = tensor(A + a.rp_off + t0 * (NH * NREL) + h * NREL, dextents<int32_t, 2>(NREL, (int) nq), array<int, 2>({1, NH * NREL}));
    auto ts = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(ss, dextents<int32_t, 2>(AK, AQ));
    auto tg = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(G, dextents<int32_t, 2>(AG, AQ));
    auto tp = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sp, dextents<int32_t, 2>(AK, AQ));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AK, HD, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
                              execution_simdgroups<8>> mm_qk;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AG, NREL, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
                              execution_simdgroups<8>> mm_g;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, HD, AK, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<8>> mm_pv;
    device kv_t * kbase = KV + a.k_base + g * a.cap * HD;
    device kv_t * vbase = KV + a.v_base + g * a.cap * HD;
    auto tv0 = tensor(vbase, dextents<int32_t, 2>(HD, AK), array<int, 2>({1, HD}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    for (short i = 0; i < 16; ++i) co[i] = 0.0f;
    // this thread's accumulators: 4 quads of (4 consecutive dims, one query row)
    ushort cd[4], cr[4];
    for (short q = 0; q < 4; ++q) { const auto ix = co.get_multidimensional_index(4 * q); cd[q] = ix[0]; cr[q] = ix[1]; }
    // this simdgroup's query rows: j = jj * 8 + sg
    float M[4], S[4];
    for (short jj = 0; jj < 4; jj++) {
        const uint j = jj * 8 + sg;
        // the sink is one more key with logit sink[j] (running max starts there, its probability e^8 in the scaled sum)
        // split 0 carries the sink; the others start empty (a finite floor keeps fully masked splits finite)
        if (spl == 0) { M[jj] = j < nq ? A[a.sink_off + (t0 + j) * NH + h] : 0.0f; S[jj] = 2980.9579870417283f; }
        else { M[jj] = -1e30f; S[jj] = 0.0f; }
    }
    device const float * relb = (device const float *) (PB + a.relbias_w);
    const uint pfirst = a.p0 + t0, plast = a.p0 + t0 + nq - 1;
    uint kstart = (a.window > 0 && pfirst + 1 > a.window) ? pfirst + 1 - a.window : 0;
    kstart = kstart / AK * AK;
    const uint kbeg = kstart + spl * ta.kps, kend = min(kbeg + ta.kps, plast + 1);
    for (uint kc = kbeg; kc < kend; kc += AK) {
        const uint idx = (uint) ((int) kc - a.kpos0);
        auto tk = tensor(kbase + idx * HD, dextents<int32_t, 2>(HD, AK), array<int, 2>({1, HD}));
        // (a threadgroup-memory destination is far slower than register accumulators + store)
        { auto cs = mm_qk.template get_destination_cooperative_tensor<decltype(tq), decltype(tk), float>(); mm_qk.run(tq, tk, cs); cs.store(ts); }
        const int D0 = int(pfirst) - int(kc);            // dist(j, col) = D0 + j - col
        const int dlo = max(D0 - (AK - 1), 0);
        const bool bias = dlo < EXT;
        if (bias) {
            auto tb = tensor((device float *) relb + dlo * NREL, dextents<int32_t, 2>(NREL, AG), array<int, 2>({1, NREL}));
            auto cg = mm_g.template get_destination_cooperative_tensor<decltype(tr), decltype(tb), float>(); mm_g.run(tr, tb, cg); cg.store(tg);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // interior chunk: every key is visible to every query of the tile and inside the bias table (or entirely beyond it)
        const bool fast = nq == AQ && D0 >= AK - 1 && (a.window == 0 || D0 + AQ - 1 < (int) a.window) && (!bias || D0 + AQ - 1 < EXT);
        const int gofs = D0 - dlo - int(lane);
#ifdef KV8
        // scales of this lane's two keys: K into the scores, V into the probabilities that weight the stored integers
        const float ks0 = kv_vscale(kbase + (idx + lane) * HD), ks1 = kv_vscale(kbase + (idx + 32 + lane) * HD);
        const float vs0 = kv_vscale(vbase + (idx + lane) * HD), vs1 = kv_vscale(vbase + (idx + 32 + lane) * HD);
#endif
        for (short jj = 0; jj < 4; jj++) {
            const uint j = jj * 8 + sg;
#ifdef KV8
            float s0 = ss[j * AK + lane] * ks0, s1 = ss[j * AK + 32 + lane] * ks1;
#else
            float s0 = ss[j * AK + lane], s1 = ss[j * AK + 32 + lane];
#endif
            if (fast) {
                if (bias) { threadgroup const float * gp = G + j * AG + j + gofs; s0 += gp[0]; s1 += gp[-32]; }
            } else {
                const int d0 = D0 + int(j) - int(lane), d1 = d0 - 32;
                if (bias && d0 < EXT) s0 += G[j * AG + clamp(d0 - dlo, 0, AG - 1)];
                if (bias && d1 < EXT) s1 += G[j * AG + clamp(d1 - dlo, 0, AG - 1)];
                if (!(j < nq && d0 >= 0 && (a.window == 0 || d0 < (int) a.window))) s0 = -INFINITY;
                if (!(j < nq && d1 >= 0 && (a.window == 0 || d1 < (int) a.window))) s1 = -INFINITY;
            }
            const float m = simd_max(max(M[jj], max(s0, s1)));
            const float ms = exp(M[jj] - m);
            M[jj] = m;
            const half p0 = half(exp(s0 - m + 8.0f)), p1 = half(exp(s1 - m + 8.0f));
#ifdef KV8
            sp[j * AK + lane] = half(float(p0) * vs0);
            sp[j * AK + 32 + lane] = half(float(p1) * vs1);
#else
            sp[j * AK + lane] = p0;
            sp[j * AK + 32 + lane] = p1;
#endif
            S[jj] = S[jj] * ms + simd_sum(float(p0) + float(p1));
            if (lane == 0) sr[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const float r0 = sr[cr[0]], r1 = sr[cr[1]], r2 = sr[cr[2]], r3 = sr[cr[3]];
            co[0] *= r0; co[1] *= r0; co[2] *= r0; co[3] *= r0;
            co[4] *= r1; co[5] *= r1; co[6] *= r1; co[7] *= r1;
            co[8] *= r2; co[9] *= r2; co[10] *= r2; co[11] *= r2;
            co[12] *= r3; co[13] *= r3; co[14] *= r3; co[15] *= r3;
        }
        auto tv = tensor(vbase + idx * HD, dextents<int32_t, 2>(HD, AK), array<int, 2>({1, HD}));
        mm_pv.run(tp, tv, co);
    }
    // this split's partials: O (scaled by e^8, relative to M), M and S per (query, head)
    device float * P = A + ta.part_off;
    for (short jj = 0; jj < 4; jj++) {
        const uint j = jj * 8 + sg;
        if (lane == 0 && j < nq) { device float * base = P + ((ulong) (spl * nq + j) * NH + h) * (HD + 2); base[HD] = M[jj]; base[HD + 1] = S[jj]; }
    }
    for (short q = 0; q < 4; ++q) {
        const uint row = cr[q];
        if (row < nq) {
            device float * base = P + ((ulong) (spl * nq + row) * NH + h) * (HD + 2) + cd[q];
            base[0] = co[4 * q]; base[1] = co[4 * q + 1]; base[2] = co[4 * q + 2]; base[3] = co[4 * q + 3];
        }
    }
}
kernel void k_attn_hs_comb(device float * A [[buffer(2)]], constant AttnHsArgs & ta [[buffer(5)]],
                           uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constant AttnBArgs & a = ta.b;
    const uint t = tgp.x, h = tgp.y, nq = a.T;
    device const float * P = A + ta.part_off;
    float M = -1e30f;
    for (uint s = 0; s < ta.nsplit; s++) M = max(M, P[((ulong) (s * nq + t) * NH + h) * (HD + 2) + HD]);
    float S = 0.0f, O = 0.0f;
    for (uint s = 0; s < ta.nsplit; s++) {
        device const float * base = P + ((ulong) (s * nq + t) * NH + h) * (HD + 2);
        const float w = exp(base[HD] - M);
        S += base[HD + 1] * w; O += base[tid] * w;
    }
    const ulong i = (ulong) t * (NH * HD) + h * HD + tid;
    const float gate = a.ghalf ? float(((device const half *) (A + a.gate_off))[i]) : A[a.gate_off + i];
    ((device half *) (A + a.out_off))[i] = half(O / S / (1.0f + exp(-gate)));
}
// accumulator layout probe for k_attn_h's P.V op (same format as k_layout)
kernel void k_layout_att(device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]], constant uint & out_off [[buffer(5)]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sp[AQ * AK];
    auto tp = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sp, dextents<int32_t, 2>(AK, AQ));
    auto tv0 = tensor(KV, dextents<int32_t, 2>(HD, AK), array<int, 2>({1, HD}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, HD, AK, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<8>> mm_pv;
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    device int * o = (device int *) (A + out_off) + tid * 64;
    for (short i = 0; i < co.get_capacity() && i < 63; ++i) {
        if (co.is_valid_element(i)) { const auto ix = co.get_multidimensional_index(i); o[i] = (ix[0] << 16) | ix[1]; } else o[i] = -1;
    }
    o[63] = co.get_capacity();
}

// ---- batched loop-controller apply: out = u*(1 - tanh(f)) + tanh(g)*m, then RMSNorm(weight). grid (T, nC), 256 threads
struct CtrlBJob { uint u_off, m_off, g_off, f_off, out_off, norm_w; int h_off; uint nof32; };   // h_off >= 0: also (nof32: only) half output
struct CtrlBArgs { uint n; uint pad0, pad1, pad2; CtrlBJob job[4]; };
kernel void k_ctrl_apply_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant CtrlBArgs & a [[buffer(5)]],
                           uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    constant CtrlBJob & J = a.job[tgp.y];
    const uint t = tgp.x, off = t * E;
    float o[9], ss = 0;
    for (int k = 0; k < 9; k++) {
        const uint i = tid + k * 256;
        const float u = A[J.u_off + off + i];
        if (J.m_off == 0xFFFFFFFFu) o[k] = u;      // already applied by k_ctrl_gf
        else { const float m = A[J.m_off + off + i]; o[k] = u - tanh(A[J.f_off + off + i]) * u + tanh(A[J.g_off + off + i]) * m; }
        ss += o[k] * o[k];
    }
    ss = tg_sum(ss, sh, sg, lane, 8);
    const float r = rsqrt(ss / E + 1e-6f);
    device const float * w = (device const float *) (PB + J.norm_w);
    for (int k = 0; k < 9; k++) {
        const uint i = tid + k * 256;
        const float v = o[k] * r * w[i];
        if (J.h_off >= 0) ((device half *) (A + J.h_off))[off + i] = half(v);
        if (!J.nof32) A[J.out_off + off + i] = v;
    }
}

// ---- batched depth mix (same math as k_mix): one simdgroup per (job, token), no barriers.
// pass 1 accumulates per-state dot products with the queries AND the Gram matrix of the states, so the output
// RMS norm is known analytically (|sum a_i s_i|^2 = sum a_i a_j <s_i,s_j>); pass 2 writes normalized outputs directly.
// Specialized on (number of states, number of queries) so every register array is statically indexed.
template <int NS, int NQ, bool IDENT>
kernel void k_mix_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant MixJob & J [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint tok = tg * 4 + sg;
    if (tok >= J.pad_t) return;
    const uint toff = tok * J.tstride;
    device const float4 * S[NS];
    #pragma unroll
    for (int i = 0; i < NS; i++) S[i] = (device const float4 *) (A + J.s_off[i] + toff);
    float G[NS][NS], D[NS][NQ];
    #pragma unroll
    for (int i = 0; i < NS; i++) {
        #pragma unroll
        for (int j = 0; j < NS; j++) G[i][j] = 0;
        #pragma unroll
        for (int q = 0; q < NQ; q++) D[i][q] = 0;
    }
    device const float4 * Q = (device const float4 *) (PB + max(J.q_off, 0));
    for (uint k = lane; k < E / 4; k += 32) {
        float4 v[NS];
        #pragma unroll
        for (int i = 0; i < NS; i++) v[i] = S[i][k];
        float4 qv[NQ];
        #pragma unroll
        for (int q = 0; q < NQ; q++) qv[q] = IDENT ? float4(0) : Q[q * (E / 4) + k];
        #pragma unroll
        for (int i = 0; i < NS; i++) {
            #pragma unroll
            for (int j = i; j < NS; j++) G[i][j] += dot(v[i], v[j]);
            #pragma unroll
            for (int q = 0; q < NQ; q++) D[i][q] += dot(v[i], qv[q]);
        }
    }
    #pragma unroll
    for (int i = 0; i < NS; i++) {
        #pragma unroll
        for (int j = i; j < NS; j++) { G[i][j] = simd_sum(G[i][j]); G[j][i] = G[i][j]; }
        #pragma unroll
        for (int q = 0; q < NQ; q++) D[i][q] = simd_sum(D[i][q]);
    }
    float al[NQ][NS], rn[NQ], rw[NQ];
    #pragma unroll
    for (int q = 0; q < NQ; q++) {
        if (IDENT) {
            #pragma unroll
            for (int i = 0; i < NS; i++) al[q][i] = i == 0 ? 1.0f : 0.0f;
        } else {
            float sc[NS], mx = -INFINITY;
            #pragma unroll
            for (int i = 0; i < NS; i++) { sc[i] = rsqrt(G[i][i] / E + 1e-6f) * D[i][q] * (1.0f / 48.0f); mx = max(mx, sc[i]); }
            float den = 0;
            #pragma unroll
            for (int i = 0; i < NS; i++) { sc[i] = exp(sc[i] - mx); den += sc[i]; }
            #pragma unroll
            for (int i = 0; i < NS; i++) al[q][i] = sc[i] / den * rsqrt(G[i][i] / E + 1e-6f);
        }
        float nn = 0;
        #pragma unroll
        for (int i = 0; i < NS; i++) {
            #pragma unroll
            for (int j = 0; j < NS; j++) nn += al[q][i] * al[q][j] * G[i][j];
        }
        rn[q] = rsqrt(nn / E + 1e-6f);
        rw[q] = rsqrt(nn / E + J.eps);
    }
    for (uint k = lane; k < E / 4; k += 32) {
        float4 v[NS];
        #pragma unroll
        for (int i = 0; i < NS; i++) v[i] = S[i][k];
        #pragma unroll
        for (int q = 0; q < NQ; q++) {
            float4 o = 0;
            #pragma unroll
            for (int i = 0; i < NS; i++) o += al[q][i] * v[i];
            #pragma unroll
            for (int c2 = 0; c2 < 3; c2++) {
                if (J.ncopy[q][c2] >= 0) ((device float4 *) (A + J.ncopy[q][c2] + toff))[k] = o * rn[q];
                if (J.hcopy[q][c2] >= 0) ((device half4 *) ((device half *) (A + J.hcopy[q][c2]) + (ulong) tok * J.hstride))[k] = half4(o * rn[q]);
            }
            if (J.norm_off[q] >= 0) o *= rw[q] * ((device const float4 *) (PB + J.norm_off[q]))[k];
            if (J.hout[q] >= 0) ((device half4 *) ((device half *) (A + J.hout[q]) + (ulong) tok * E))[k] = half4(o);
            if (!((J.nof32 >> q) & 1)) ((device float4 *) (A + J.out_off[q] + toff))[k] = o;
        }
    }
}
#define MIXB(NS, NQ) template [[host_name("k_mix_b_" #NS "_" #NQ)]] kernel void k_mix_b<NS, NQ, false>(device const uchar *, device float *, constant MixJob &, uint, ushort, ushort);
MIXB(1, 1) MIXB(2, 1) MIXB(3, 1) MIXB(4, 1) MIXB(5, 1) MIXB(6, 1)
MIXB(1, 3) MIXB(2, 3) MIXB(3, 3) MIXB(4, 3) MIXB(5, 3) MIXB(6, 3)
MIXB(1, 4) MIXB(2, 4) MIXB(3, 4) MIXB(4, 4) MIXB(5, 4) MIXB(6, 4)
template [[host_name("k_mix_b_id")]] kernel void k_mix_b<1, 1, true>(device const uchar *, device float *, constant MixJob &, uint, ushort, ushort);

// The same mix with one threadgroup of 8 simdgroups per token (verify passes: at 2..=8 tokens k_mix_b runs one
// threadgroup in all, a simdgroup per token walking every channel). grid (T), 256 threads.
template <int NS, int NQ, bool IDENT>
kernel void k_mix_bw(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant MixJob & J [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint tok = tg, tid = sg * 32 + lane;
    threadgroup float red[8][64];
    if (tok >= J.pad_t) return;
    const uint toff = tok * J.tstride;
    device const float4 * S[NS];
    #pragma unroll
    for (int i = 0; i < NS; i++) S[i] = (device const float4 *) (A + J.s_off[i] + toff);
    float G[NS][NS], D[NS][NQ];
    #pragma unroll
    for (int i = 0; i < NS; i++) {
        #pragma unroll
        for (int j = 0; j < NS; j++) G[i][j] = 0;
        #pragma unroll
        for (int q = 0; q < NQ; q++) D[i][q] = 0;
    }
    device const float4 * Q = (device const float4 *) (PB + max(J.q_off, 0));
    for (uint k = tid; k < E / 4; k += 256) {
        float4 v[NS];
        #pragma unroll
        for (int i = 0; i < NS; i++) v[i] = S[i][k];
        float4 qv[NQ];
        #pragma unroll
        for (int q = 0; q < NQ; q++) qv[q] = IDENT ? float4(0) : Q[q * (E / 4) + k];
        #pragma unroll
        for (int i = 0; i < NS; i++) {
            #pragma unroll
            for (int j = i; j < NS; j++) G[i][j] += dot(v[i], v[j]);
            #pragma unroll
            for (int q = 0; q < NQ; q++) D[i][q] += dot(v[i], qv[q]);
        }
    }
    #pragma unroll
    for (int i = 0; i < NS; i++) {
        #pragma unroll
        for (int j = i; j < NS; j++) { G[i][j] = simd_sum(G[i][j]); G[j][i] = G[i][j]; }
        #pragma unroll
        for (int q = 0; q < NQ; q++) D[i][q] = simd_sum(D[i][q]);
    }
    // across the 8 simdgroups (each summed a stripe of the channels)
    if (lane == 0) {
        int n = 0;
        #pragma unroll
        for (int i = 0; i < NS; i++) {
            #pragma unroll
            for (int j = 0; j < NS; j++) red[sg][n++] = G[i][j];
            #pragma unroll
            for (int q = 0; q < NQ; q++) red[sg][n++] = D[i][q];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        int n = 0;
        #pragma unroll
        for (int i = 0; i < NS; i++) {
            #pragma unroll
            for (int j = 0; j < NS; j++) { float a = 0; for (int g = 0; g < 8; g++) a += red[g][n]; G[i][j] = a; n++; }
            #pragma unroll
            for (int q = 0; q < NQ; q++) { float a = 0; for (int g = 0; g < 8; g++) a += red[g][n]; D[i][q] = a; n++; }
        }
    }
    float al[NQ][NS], rn[NQ], rw[NQ];
    #pragma unroll
    for (int q = 0; q < NQ; q++) {
        if (IDENT) {
            #pragma unroll
            for (int i = 0; i < NS; i++) al[q][i] = i == 0 ? 1.0f : 0.0f;
        } else {
            float sc[NS], mx = -INFINITY;
            #pragma unroll
            for (int i = 0; i < NS; i++) { sc[i] = rsqrt(G[i][i] / E + 1e-6f) * D[i][q] * (1.0f / 48.0f); mx = max(mx, sc[i]); }
            float den = 0;
            #pragma unroll
            for (int i = 0; i < NS; i++) { sc[i] = exp(sc[i] - mx); den += sc[i]; }
            #pragma unroll
            for (int i = 0; i < NS; i++) al[q][i] = sc[i] / den * rsqrt(G[i][i] / E + 1e-6f);
        }
        float nn = 0;
        #pragma unroll
        for (int i = 0; i < NS; i++) {
            #pragma unroll
            for (int j = 0; j < NS; j++) nn += al[q][i] * al[q][j] * G[i][j];
        }
        rn[q] = rsqrt(nn / E + 1e-6f);
        rw[q] = rsqrt(nn / E + J.eps);
    }
    for (uint k = tid; k < E / 4; k += 256) {
        float4 v[NS];
        #pragma unroll
        for (int i = 0; i < NS; i++) v[i] = S[i][k];
        #pragma unroll
        for (int q = 0; q < NQ; q++) {
            float4 o = 0;
            #pragma unroll
            for (int i = 0; i < NS; i++) o += al[q][i] * v[i];
            #pragma unroll
            for (int c2 = 0; c2 < 3; c2++) {
                if (J.ncopy[q][c2] >= 0) ((device float4 *) (A + J.ncopy[q][c2] + toff))[k] = o * rn[q];
                if (J.hcopy[q][c2] >= 0) ((device half4 *) ((device half *) (A + J.hcopy[q][c2]) + (ulong) tok * J.hstride))[k] = half4(o * rn[q]);
            }
            if (J.norm_off[q] >= 0) o *= rw[q] * ((device const float4 *) (PB + J.norm_off[q]))[k];
            if (J.hout[q] >= 0) ((device half4 *) ((device half *) (A + J.hout[q]) + (ulong) tok * E))[k] = half4(o);
            if (!((J.nof32 >> q) & 1)) ((device float4 *) (A + J.out_off[q] + toff))[k] = o;
        }
    }
}
#define MIXBW(NS, NQ) template [[host_name("k_mix_bw_" #NS "_" #NQ)]] kernel void k_mix_bw<NS, NQ, false>(device const uchar *, device float *, constant MixJob &, uint, ushort, ushort);
MIXBW(1, 1) MIXBW(2, 1) MIXBW(3, 1) MIXBW(4, 1) MIXBW(5, 1) MIXBW(6, 1)
MIXBW(1, 3) MIXBW(2, 3) MIXBW(3, 3) MIXBW(4, 3) MIXBW(5, 3) MIXBW(6, 3)
MIXBW(1, 4) MIXBW(2, 4) MIXBW(3, 4) MIXBW(4, 4) MIXBW(5, 4) MIXBW(6, 4)
template [[host_name("k_mix_bw_id")]] kernel void k_mix_bw<1, 1, true>(device const uchar *, device float *, constant MixJob &, uint, ushort, ushort);

// ======================= fp16-activation tensor GEMMs (activations streamed from device memory as half) =======================
#define HB_M 64
#define HB_N 64
#define HB_K 32

// f32 [rows][K] -> half; optionally gathered rows (perm) ; grid (K/256, rows), 256 threads... generic elementwise
struct ToHalfArgs { uint src_off, dst_off, K, rows, perm_off, gather, src_stride, dst_stride; };
kernel void k_tohalf(device float * A [[buffer(2)]], constant ToHalfArgs & a [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint r = tgp.y, c = (tgp.x * 256 + tid) * 4;
    if (c >= a.K) return;
    const uint sr = (a.gather & 1) ? ((device const uint *) (A + a.perm_off))[r] : r;
    device half4 * dst = (device half4 *) ((device half *) (A + a.dst_off) + (ulong) r * a.dst_stride + c);
    if (a.gather & 2) *dst = *(device const half4 *) ((device const half *) (A + a.src_off) + (ulong) sr * a.src_stride + c);   // half source
    else *dst = half4(*(device const float4 *) (A + a.src_off + (ulong) sr * a.src_stride + c));
}

// ---- weight tile staging (64 rows x 32 k, half) for the tensor GEMMs below.
// Q4_K fast path: the block header (d, dmin, 12 scale bytes) is loaded once per super-block and kept in registers, so a k-step
// is one 16-byte load + unpack with no dependent loads in front of it: the staging latency stalls every simdgroup of the tile.
struct WHead { float d, m; uint s0, s1, s2; };
inline void wstage(device const uchar * wr, uint wtype, uint k0, uint hp, thread WHead & h, thread half4 * w) {
    if (wtype != 12) { deq16(wr, wtype, k0 + hp * 16, (thread half *) w); return; }
    device const block_q4_K * b = (device const block_q4_K *) wr + k0 / 256;
    const uint j = (k0 % 256) / 32;
    if (j == 0) { h.d = b->d; h.m = b->dmin; const uint3 sc3 = *(device const packed_uint3 *) b->scales; h.s0 = sc3.x; h.s1 = sc3.y; h.s2 = sc3.z; }
    uint sc, mn;    // 6-bit scale / min of sub-block j (same decoding as scale_min_k4)
    if (j < 4) { sc = (h.s0 >> (8 * j)) & 63; mn = (h.s1 >> (8 * j)) & 63; }
    else {
        const uint hi = (h.s2 >> (8 * (j - 4))) & 0xFF;
        sc = (hi & 0xF) | (((h.s0 >> (8 * (j - 4) + 6)) & 3) << 4);
        mn = (hi >> 4) | (((h.s1 >> (8 * (j - 4) + 6)) & 3) << 4);
    }
    // fp32 scale/offset, one rounding to half (bit-identical to the scalar reference path)
    const float df = h.d * float(sc), mf = h.m * float(mn);
    const uint4 v = (*(device const uint4 *) (b->qs + (j / 2) * 32 + hp * 16) >> ((j & 1) * 4)) & 0x0F0F0F0Fu;
    w[0] = half4(fma(float4(as_type<uchar4>(v.x)), df, -mf)); w[1] = half4(fma(float4(as_type<uchar4>(v.y)), df, -mf));
    w[2] = half4(fma(float4(as_type<uchar4>(v.z)), df, -mf)); w[3] = half4(fma(float4(as_type<uchar4>(v.w)), df, -mf));
}

// All fp16 GEMMs share one tile shape: a threadgroup of NSG simdgroups stages a 64-row x 32-k weight tile per k-step (first
// 128 threads, double-buffered), and every simdgroup runs its OWN tensor op (execution_simdgroup) on its own 16 tokens with the
// accumulators (64 x 16) in registers. Compared with one op over the whole threadgroup this is a little faster, and simdgroups
// whose tokens do not exist simply skip the matmul, so ragged token counts cost 16-token granularity instead of a whole tile.
// Register epilogues rely on the op's accumulator layout: 8 quads of 4 consecutive rows per thread, quads q and q+4 holding rows
// r and r+32 of the same token. Engine::load verifies this with k_layout_sg before enabling the path.
#define SGOP(mm) mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, HB_M, HB_K, false, true, false, \
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> mm

// segmented dense GEMM: Y[t][r] = sum_k Xh[t][k] W[r][k]; Xh half [T][K]. grid (token tiles of 16*NSG, row tiles of 64):
// Metal sweeps y fastest, so the row tiles of one token tile run together and its X stays cached.
struct HSeg { uint w_lo, w_hi, row_bytes, wtype, nrows, K, tile_start, xh_off, out_off, ldo, wsrc, hout; };   // hout: half output [T][ldo]
struct GemmHArgs { uint nseg, T, pad0, pad1; HSeg seg[10]; };
template <int NSG>
kernel void k_gemm_sg(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                      constant GemmHArgs & args [[buffer(5)]], uint2 tgq [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[2][HB_M * HB_K];
    uint si = 0;
    while (si + 1 < args.nseg && tgq.y >= args.seg[si + 1].tile_start) si++;
    constant HSeg & S = args.seg[si];
    const uint r0 = (tgq.y - S.tile_start) * HB_M, t0 = (tgq.x * NSG + sg) * 16, K = S.K, T = args.T;
    const bool on = t0 < T;
    const int nt = on ? min(16, (int) (T - t0)) : 0;
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(HB_K, HB_M));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(HB_K, HB_M));
    auto tX = tensor((device half *) (A + S.xh_off) + (ulong) (on ? t0 : 0) * K, dextents<int32_t, 2>((int) K, nt), array<int, 2>({1, (int) K}));
    SGOP(mm);
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    const bool wact = tid < 128;
    const uint widx = tid * 16, hp = (widx % HB_K) / 16;
    device const uchar * wr = (S.wsrc ? PB : W) + (ulong(S.w_hi) << 32 | S.w_lo) + (ulong) min(r0 + widx / HB_K, S.nrows - 1) * S.row_bytes;
    WHead wh; half4 wv[4];
    if (wact) { wstage(wr, S.wtype, 0, hp, wh, wv); for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[0] + widx))[i] = wv[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += HB_K) {
        const bool more = k0 + HB_K < K;
        if (more && wact) wstage(wr, S.wtype, k0 + HB_K, hp, wh, wv);
        if (on) { auto sx = tX.slice(k0, 0); if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT); }
        if (more && wact) for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[buf ^ 1] + widx))[i] = wv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    if (!on) return;
    if (S.hout) {
        device half * yo = (device half *) (A + S.out_off);
        for (short q = 0; q < 8; ++q) {
            const auto ix = cT.get_multidimensional_index(4 * q);
            const uint r = r0 + ix[0], t = t0 + ix[1];
            if (r < S.nrows && t < T) *(device half4 *) (yo + (ulong) t * S.ldo + r) = half4(float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]));
        }
        return;
    }
    auto tC = tensor(A + S.out_off + (ulong) t0 * S.ldo + r0, dextents<int32_t, 2>(min(HB_M, (int) (S.nrows - r0)), nt), array<int, 2>({1, (int) S.ldo}));
    cT.store(tC);
}
template [[host_name("k_gemm_sg8")]] kernel void k_gemm_sg<8>(device const uchar *, device const uchar *, device float *, constant GemmHArgs &, uint2, ushort, ushort);
template [[host_name("k_gemm_sg4")]] kernel void k_gemm_sg<4>(device const uchar *, device const uchar *, device float *, constant GemmHArgs &, uint2, ushort, ushort);

// ---- small-T matvec for the batched path (inkserve, DSPARK.md): T <= 8 tokens, the decode matvec's layout (8 rows per
// threadgroup, 2 per simdgroup), each weight block read once and applied to every token. Same arguments and output as
// k_gemm_sg (half input [T][K], f32 or half output [T][ldo]), but tile_start counts 8-row tiles. At small T the
// 64-row GEMM tiles leave too few threadgroups to keep memory busy. Types: Q8_0, Q6_K and BF16 (the engine checks).
template <int NT>
kernel void k_mvT(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                  constant GemmHArgs & args [[buffer(5)]], uint tg [[threadgroup_position_in_grid]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    uint si = 0;
    while (si + 1 < args.nseg && tg >= args.seg[si + 1].tile_start) si++;
    constant HSeg & S = args.seg[si];
    const uint r0 = (tg - S.tile_start) * 8 + sg * 2;
    if (r0 >= S.nrows) return;
    const uint K = S.K;
    device const uchar * wb = (S.wsrc ? PB : W) + (ulong(S.w_hi) << 32 | S.w_lo);
    device const uchar * wr[2] = { wb + (ulong) r0 * S.row_bytes, wb + (ulong) min(r0 + 1, S.nrows - 1) * S.row_bytes };
    device const half * xh = (device const half *) (A + S.xh_off);
    float acc[2][NT];
    for (int r = 0; r < 2; r++) for (int t = 0; t < NT; t++) acc[r][t] = 0.0f;
    if (S.wtype == 8) {
        // Q8_0: 4 lanes per block (8 values each), 8 blocks per pass
        const ushort ix = lane / 4, il = lane % 4;
        for (uint ib = ix; ib < K / 32; ib += 8) {
            float4 wa[2], wc[2]; float d[2];
            for (int r = 0; r < 2; r++) {
                device const ushort * p = (device const ushort *) (wr[r] + ib * 34);
                d[r] = float(as_type<half>(p[0]));
                const char2 q0 = as_type<char2>(p[1 + 4 * il]), q1 = as_type<char2>(p[2 + 4 * il]);
                const char2 q2 = as_type<char2>(p[3 + 4 * il]), q3 = as_type<char2>(p[4 + 4 * il]);
                wa[r] = float4(q0.x, q0.y, q1.x, q1.y); wc[r] = float4(q2.x, q2.y, q3.x, q3.y);
            }
            for (int t = 0; t < NT; t++) {
                device const half4 * x4 = (device const half4 *) (xh + (ulong) t * K + ib * 32 + il * 8);
                const float4 xa = float4(x4[0]), xb = float4(x4[1]);
                for (int r = 0; r < 2; r++) acc[r][t] += d[r] * (dot(xa, wa[r]) + dot(xb, wc[r]));
            }
        }
    } else if (S.wtype == 14) {
        // Q6_K: dot_q6k's lane layout (2 lanes per 16-value group, 2 superblocks per pass); weights decoded once per block
        const short tq = lane / 2, ix = lane % 2, ip = tq / 8, il = tq % 8, l0 = 4 * il, is = 8 * ip + l0 / 16;
        const short y_off = 128 * ip + l0, ql_off = 64 * ip + l0, qh_off = 32 * ip + l0;
        for (uint i = ix; i < K / 256; i += 2) {
            float w[2][16], sc[2][4], d[2];
            for (int r = 0; r < 2; r++) {
                device const block_q6_K * b = (device const block_q6_K *) wr[r] + i;
                device const uchar * q1 = b->ql + ql_off;
                device const uchar * q2 = q1 + 32;
                device const uchar * qh = b->qh + qh_off;
                device const char * s8 = b->scales + is;
                for (int l = 0; l < 4; l++) {
                    w[r][4 * l + 0] = float((int8_t) ((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
                    w[r][4 * l + 1] = float((int8_t) ((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
                    w[r][4 * l + 2] = float((int8_t) ((q1[l] >> 4) | ((qh[l] & 0x30) << 0)) - 32);
                    w[r][4 * l + 3] = float((int8_t) ((q2[l] >> 4) | ((qh[l] & 0xC0) >> 2)) - 32);
                }
                sc[r][0] = s8[0]; sc[r][1] = s8[2]; sc[r][2] = s8[4]; sc[r][3] = s8[6];
                d[r] = float(b->d);
            }
            for (int t = 0; t < NT; t++) {
                device const half * y = xh + (ulong) t * K + i * 256 + y_off;
                float yl[16];
                for (int l = 0; l < 4; l++) { yl[4 * l + 0] = y[l]; yl[4 * l + 1] = y[l + 32]; yl[4 * l + 2] = y[l + 64]; yl[4 * l + 3] = y[l + 96]; }
                for (int r = 0; r < 2; r++) {
                    float s4[4] = {0, 0, 0, 0};
                    for (int l = 0; l < 4; l++) for (int c = 0; c < 4; c++) s4[c] += yl[4 * l + c] * w[r][4 * l + c];
                    acc[r][t] += d[r] * (s4[0] * sc[r][0] + s4[1] * sc[r][1] + s4[2] * sc[r][2] + s4[3] * sc[r][3]);
                }
            }
        }
    } else {
        // BF16: 8 values per lane, 256 per simdgroup pass
        for (uint i = lane * 8; i < K; i += 256) {
            float4 wa[2], wc[2];
            for (int r = 0; r < 2; r++) {
                const ushort4 a = *(device const ushort4 *) (wr[r] + 2 * i), b = *(device const ushort4 *) (wr[r] + 2 * i + 8);
                wa[r] = float4(bf16f(a.x), bf16f(a.y), bf16f(a.z), bf16f(a.w)); wc[r] = float4(bf16f(b.x), bf16f(b.y), bf16f(b.z), bf16f(b.w));
            }
            for (int t = 0; t < NT; t++) {
                device const half4 * x4 = (device const half4 *) (xh + (ulong) t * K + i);
                const float4 xa = float4(x4[0]), xb = float4(x4[1]);
                for (int r = 0; r < 2; r++) acc[r][t] += dot(xa, wa[r]) + dot(xb, wc[r]);
            }
        }
    }
    for (int t = 0; t < NT; t++) for (int r = 0; r < 2; r++) {
        const float y = simd_sum(acc[r][t]);
        if (lane == 0 && r0 + r < S.nrows) {
            if (S.hout) ((device half *) (A + S.out_off))[(ulong) t * S.ldo + r0 + r] = half(y);
            else A[S.out_off + (ulong) t * S.ldo + r0 + r] = y;
        }
    }
}
#define MVT_INST(n) template [[host_name("k_mvT" #n)]] kernel void k_mvT<n>(device const uchar *, device const uchar *, device float *, constant GemmHArgs &, uint, ushort, ushort);
MVT_INST(2) MVT_INST(3) MVT_INST(4) MVT_INST(5) MVT_INST(6) MVT_INST(7) MVT_INST(8)

// ---- small-T MoE for the batched path (inkserve, DSPARK.md): n <= 8 tokens. Each expert in the union of the
// tokens' choices is read once, in the decode kernels' layout (4 lanes per block), and applied to every token that
// chose it. k_moe_routeT builds the list at list_off: [0] = U, [1 + u] = expert u, [65 + 8u + j] = its assignments
// t * TOPK + k (NONE past the end). k_moe_guT writes SiTU(gate, up) as half to acth[t * TOPK + k][FF]; k_moe_downT
// writes each token's weighted expert sum to out[t][E] (what k_moe_comb_h would), so the gather and combine go away.
#define MOET_NONE 0xFFFFFFFFu
struct MoeTArgs { uint sel_off, n, list_off, xh_off, acth_off, out_off;
                  uint gu_lo, gu_hi, gu_eb, gu_rb, gu_type, gu_gs_lo, gu_gs_hi;
                  uint dn_lo, dn_hi, dn_eb, dn_rb, dn_type, dn_gs_lo, dn_gs_hi; float beta, lbeta; uint pad0, pad1; };
// list layout (u32 at list_off): [0] U, [1 + u] expert, [65 + 8u + j] assignment t * TOPK + k, [577 + u] count.
// 64 threads; the dedupe runs on one thread in threadgroup memory (n * TOPK <= 64 assignments), then all threads write.
kernel void k_moe_routeT(device float * A [[buffer(2)]], constant MoeTArgs & a [[buffer(5)]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint ids[64], ue[64], uc[64], ua[64 * 8];
    threadgroup uint nU;
    const uint N = a.n * TOPK;
    if (tid < N) ids[tid] = (uint) A[a.sel_off + (tid / TOPK) * 16 + (tid % TOPK)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint U = 0;
        for (uint i = 0; i < N; i++) {
            const uint e = ids[i];
            uint u = 0;
            while (u < U && ue[u] != e) u++;
            if (u == U) { ue[U] = e; uc[U] = 0; U++; }
            ua[u * 8 + uc[u]] = i;                      // a token picks an expert at most once, so uc[u] < n <= 8
            uc[u]++;
        }
        nU = U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint * P = (device uint *) (A + a.list_off);
    const uint U = nU;
    if (tid == 0) P[0] = U;
    if (tid < U) { P[1 + tid] = ue[tid]; P[577 + tid] = uc[tid]; }
    for (uint i = tid; i < U * 8; i += 64) P[65 + i] = (i % 8) < uc[i / 8] ? ua[i] : MOET_NONE;
}
// grid (FF / 4, n * TOPK): threadgroup y = union slot u, simdgroup = gate row i and up row i
template <int NT>
kernel void k_moe_guT(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant MoeTArgs & a [[buffer(5)]],
                      uint2 tgp [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    device const uint * P = (device const uint *) (A + a.list_off);
    const uint u = tgp.y;
    if (u >= P[0]) return;
    const uint e = P[1 + u], i = tgp.x * 4 + sg;
    const ulong base = (ulong(a.gu_hi) << 32 | a.gu_lo) + (ulong) e * a.gu_eb;
    device const uchar * wg = W + base + (ulong) i * a.gu_rb;
    device const uchar * wu = W + base + (ulong) (FF + i) * a.gu_rb;
    device const half * xh = (device const half *) (A + a.xh_off);
    uint as[NT]; float g[NT], up[NT];
    for (int j = 0; j < NT; j++) { as[j] = P[65 + u * 8 + j]; g[j] = 0.0f; up[j] = 0.0f; }
    const ushort ix = lane / 4, il = lane % 4;
    if (a.gu_type == 200) {
        for (uint ib = ix; ib < E / 32; ib += 8) {
            device const ushort * pg = (device const ushort *) (wg + ib * 18);
            device const ushort * pu = (device const ushort *) (wu + ib * 18);
            const float2 s1 = e4m3f2(pg[0]), s2 = e4m3f2(pu[0]);
            float4 glo, ghi, ulo, uhi;
            fp4x8(uint(pg[1 + 2 * il]) | (uint(pg[2 + 2 * il]) << 16), glo, ghi);
            fp4x8(uint(pu[1 + 2 * il]) | (uint(pu[2 + 2 * il]) << 16), ulo, uhi);
            for (int j = 0; j < NT; j++) {
                if (as[j] == MOET_NONE) continue;
                device const half * xr = xh + (ulong) (as[j] / TOPK) * E + ib * 32 + il * 4;
                const float4 xl = float4(*(device const half4 *) xr), xv = float4(*(device const half4 *) (xr + 16));
                g[j] += s1.x * dot(xl, glo) + s1.y * dot(xv, ghi);
                up[j] += s2.x * dot(xl, ulo) + s2.y * dot(xv, uhi);
            }
        }
    } else {
        for (uint ib = ix; ib < E / 32; ib += 8) {
            device const ushort * pg = (device const ushort *) (wg + ib * 34);
            device const ushort * pu = (device const ushort *) (wu + ib * 34);
            const float d1 = float(as_type<half>(pg[0])) * MX8_K, d2 = float(as_type<half>(pu[0])) * MX8_K;
            float4 ga, gb, ua, ub;
            e4m3x8(pg + 1 + 4 * il, ga, gb);
            e4m3x8(pu + 1 + 4 * il, ua, ub);
            for (int j = 0; j < NT; j++) {
                if (as[j] == MOET_NONE) continue;
                device const half * xr = xh + (ulong) (as[j] / TOPK) * E + ib * 32 + il * 8;
                const float4 xa = float4(*(device const half4 *) xr), xb = float4(*(device const half4 *) (xr + 4));
                g[j] += d1 * (dot(xa, ga) + dot(xb, gb));
                up[j] += d2 * (dot(xa, ua) + dot(xb, ub));
            }
        }
    }
    const float G = a.gu_type == 200 ? gscale_at(W, a.gu_gs_lo, a.gu_gs_hi, (int) e) * NV4_K : 1.0f;
    device half * act = (device half *) (A + a.acth_off);
    for (int j = 0; j < NT; j++) {
        const float gt = simd_sum(g[j]) * G, uv = simd_sum(up[j]) * G;
        if (lane == 0 && as[j] != MOET_NONE)
            act[(ulong) as[j] * FF + i] = half(a.beta * tanh(gt / a.beta) * sigm(gt) * (a.lbeta > 0 ? a.lbeta * tanh(uv / a.lbeta) : uv));
    }
}
// grid E / (2 * DNT_R), 64 * DNT_KS threads: 2 row groups of DNT_R output rows (more rows per simdgroup reuse each
// activation read) x DNT_KS simdgroups splitting the (union expert, block) items, 8 per simdgroup pass; the splits
// are summed through threadgroup memory
#define DNT_R 4
#define DNT_KS 4
template <int NT>
kernel void k_moe_downT(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant MoeTArgs & a [[buffer(5)]],
                        uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    device const uint * P = (device const uint *) (A + a.list_off);
    // the union list, routing weights and global scales, once per threadgroup
    threadgroup uint te[64], tc[64], ta[64 * 8];
    threadgroup float tgs[64], tw[64 * 8];
    const uint U = P[0], tid = sg * 32 + lane, rg = sg % 2, ks = sg / 2;
    threadgroup float red[DNT_KS][2][NT * DNT_R];
    for (uint i = tid; i < U; i += 64 * DNT_KS) {
        te[i] = P[1 + i]; tc[i] = P[577 + i];
        tgs[i] = a.dn_type == 200 ? gscale_at(W, a.dn_gs_lo, a.dn_gs_hi, (int) te[i]) * NV4_K : MX8_K;
    }
    for (uint i = tid; i < U * 8; i += 64 * DNT_KS) {
        const uint as = P[65 + i];
        ta[i] = as;
        tw[i] = as == MOET_NONE ? 0.0f : A[a.sel_off + (as / TOPK) * 16 + TOPK + (as % TOPK)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (tg * 2 + rg) * DNT_R;
    const ulong base = ulong(a.dn_hi) << 32 | a.dn_lo;
    device const half * act = (device const half *) (A + a.acth_off);
    float acc[NT][DNT_R];
    for (int t = 0; t < NT; t++) for (int r = 0; r < DNT_R; r++) acc[t][r] = 0.0f;
    const ushort ix = lane / 4, il = lane % 4;
    const uint nb = FF / 32;
    for (uint f = ix + 8 * ks; f < U * nb; f += 8 * DNT_KS) {
        const uint u = f / nb, b = f % nb, e = te[u];
        device const uchar * eb = W + base + (ulong) e * a.dn_eb;
        float4 wa[DNT_R], wc[DNT_R]; float2 sc[DNT_R];
        if (a.dn_type == 200) {
            for (int r = 0; r < DNT_R; r++) {
                device const ushort * p = (device const ushort *) (eb + (ulong) (row0 + r) * a.dn_rb + b * 18);
                sc[r] = e4m3f2(p[0]);
                fp4x8(uint(p[1 + 2 * il]) | (uint(p[2 + 2 * il]) << 16), wa[r], wc[r]);
            }
        } else {
            for (int r = 0; r < DNT_R; r++) {
                device const ushort * p = (device const ushort *) (eb + (ulong) (row0 + r) * a.dn_rb + b * 34);
                const float d = float(as_type<half>(p[0]));
                sc[r] = float2(d, d);
                e4m3x8(p + 1 + 4 * il, wa[r], wc[r]);
            }
        }
        const float G = tgs[u];
        const uint cnt = tc[u];
        for (uint j = 0; j < cnt; j++) {
            const uint as = ta[u * 8 + j];
            const uint t = as / TOPK;
            const float w = tw[u * 8 + j] * G;
            device const half * ar = act + (ulong) as * FF + b * 32;
            float v[DNT_R];
            if (a.dn_type == 200) {
                const float4 xl = float4(*(device const half4 *) (ar + il * 4)), xv = float4(*(device const half4 *) (ar + 16 + il * 4));
                for (int r = 0; r < DNT_R; r++) v[r] = sc[r].x * dot(xl, wa[r]) + sc[r].y * dot(xv, wc[r]);
            } else {
                const float4 xa = float4(*(device const half4 *) (ar + il * 8)), xb = float4(*(device const half4 *) (ar + il * 8 + 4));
                for (int r = 0; r < DNT_R; r++) v[r] = sc[r].x * (dot(xa, wa[r]) + dot(xb, wc[r]));
            }
            for (int tt = 0; tt < NT; tt++) if ((uint) tt == t) for (int r = 0; r < DNT_R; r++) acc[tt][r] += w * v[r];
        }
    }
    for (int t = 0; t < NT; t++) for (int r = 0; r < DNT_R; r++) {
        const float y = simd_sum(acc[t][r]);
        if (lane == 0) red[ks][rg][t * DNT_R + r] = y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0 && lane < NT * DNT_R) {
        float y = 0.0f;
        for (int k = 0; k < DNT_KS; k++) y += red[k][rg][lane];
        A[a.out_off + (ulong) (lane / DNT_R) * E + row0 + lane % DNT_R] = y;
    }
}
#define MOET_INST(n) \
    template [[host_name("k_moe_guT" #n)]] kernel void k_moe_guT<n>(device const uchar *, device float *, constant MoeTArgs &, uint2, ushort, ushort); \
    template [[host_name("k_moe_downT" #n)]] kernel void k_moe_downT<n>(device const uchar *, device float *, constant MoeTArgs &, uint, ushort, ushort);
MOET_INST(2) MOET_INST(3) MOET_INST(4) MOET_INST(5) MOET_INST(6) MOET_INST(7) MOET_INST(8)

// grouped MoE GEMM (contiguous half rows per tile: gathered gate_up input / act rows). grid (tiles, column blocks).
// situ=1: rows = 32 gate + 32 up of column block y; SiTU * routing weight in registers -> half act [A][FF]. situ=0: half out [A][ldo].
struct MoeGemmHArgs {
    uint w_lo, w_hi, expert_bytes, row_bytes, wtype, nrows, K, xh_off;
    uint tiles_off, ntiles_off, out_off, ldo, situ, aw_off; float beta, lbeta;
    uint gs_lo, gs_hi, pad0, pad1;
};
template <int NSG>
kernel void k_moe_sg(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant MoeGemmHArgs & a [[buffer(5)]],
                     uint2 tgq [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[2][HB_M * HB_K];
    const uint cb = tgq.y;
    if (tgq.x >= ((device const uint *) (A + a.ntiles_off))[0]) return;
    device const uint * tl = (device const uint *) (A + a.tiles_off) + tgq.x * 4;
    const uint e = tl[0], as0 = tl[1], cnt = tl[2], ts0 = sg * 16;
    const bool on = ts0 < cnt;
    const int nt = on ? min(16, (int) (cnt - ts0)) : 0;
    const uint K = a.K;
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(HB_K, HB_M));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(HB_K, HB_M));
    auto tX = tensor((device half *) (A + a.xh_off) + (ulong) (as0 + (on ? ts0 : 0)) * K, dextents<int32_t, 2>((int) K, nt), array<int, 2>({1, (int) K}));
    SGOP(mm);
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    const bool wact = tid < 128;
    const uint widx = tid * 16, rr = widx / HB_K, hp = (widx % HB_K) / 16;
    const uint wrow = a.situ ? ((rr < 32) ? cb * 32 + rr : FF + cb * 32 + (rr - 32)) : cb * HB_M + rr;
    device const uchar * wr = W + (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) e * a.expert_bytes + (ulong) wrow * a.row_bytes;
    WHead wh; half4 wv[4];
    if (wact) { wstage(wr, a.wtype, 0, hp, wh, wv); for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[0] + widx))[i] = wv[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += HB_K) {
        const bool more = k0 + HB_K < K;
        if (more && wact) wstage(wr, a.wtype, k0 + HB_K, hp, wh, wv);
        if (on) { auto sx = tX.slice(k0, 0); if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT); }
        if (more && wact) for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[buf ^ 1] + widx))[i] = wv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    if (!on) return;
    if (a.wtype == 200) { const float g = gscale_at(W, a.gs_lo, a.gs_hi, e); for (short i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] *= g; }
    if (!a.situ) {
        device half * yd = (device half *) (A + a.out_off);
        for (short q = 0; q < 8; ++q) {
            const auto ix = cT.get_multidimensional_index(4 * q);
            const uint r = ix[0], t = ts0 + ix[1];
            if (t < cnt) *(device half4 *) (yd + (ulong) (as0 + t) * a.ldo + cb * HB_M + r) = half4(float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]));
        }
        return;
    }
    device half * act = (device half *) (A + a.out_off);
    for (short q = 0; q < 4; ++q) {
        const auto ix = cT.get_multidimensional_index(4 * q);          // gate rows r..r+3; quad q+4 holds the matching up rows
        const uint r = ix[0], t = ts0 + ix[1];
        if (t < cnt) {
            const float4 gt = float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]);
            const float4 up = float4(cT[4 * q + 16], cT[4 * q + 17], cT[4 * q + 18], cT[4 * q + 19]);
            const float4 su = a.lbeta > 0 ? a.lbeta * tanh(up / a.lbeta) : up;
            const float4 o = (A[a.aw_off + as0 + t] * a.beta) * tanh(gt / a.beta) / (1.0f + exp(-gt)) * su;
            *(device half4 *) (act + (ulong) (as0 + t) * FF + cb * 32 + r) = half4(o);
        }
    }
}
template [[host_name("k_moe_sg8")]] kernel void k_moe_sg<8>(device const uchar *, device float *, constant MoeGemmHArgs &, uint2, ushort, ushort);
template [[host_name("k_moe_sg4")]] kernel void k_moe_sg<4>(device const uchar *, device float *, constant MoeGemmHArgs &, uint2, ushort, ushort);

// ---- loop controllers, gain/forget GEMM with the apply fused into the epilogue (fp16 path).
// TG = (128-token tile, job, 32-channel block): rows = 32 gain + 32 forget rows of the block, K = 64 (the controller state g, half).
// Every thread owns matching gain and forget accumulators, so raw = u - tanh(F.g)*u + tanh(G.g)*m is formed in registers and
// written once; the gain / forget pre-activations never touch memory. grid (T/128, njobs * E/32), 256 threads.
struct CtrlGfJob { uint gain_w, forget_w, gh_off, u_off, m_off, raw_off, pad0, pad1; };
struct CtrlGfArgs { uint T, njobs, pad0, pad1; CtrlGfJob job[4]; };
kernel void k_ctrl_gf(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant CtrlGfArgs & a [[buffer(5)]],
                      uint2 tgq [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[2][HB_M * HB_K];
    constant CtrlGfJob & J = a.job[tgq.y / (E / 32)];
    const uint cb = tgq.y % (E / 32), t0 = (tgq.x * 8 + sg) * 16;
    const bool on = t0 < a.T;
    const int nt = on ? min(16, (int) (a.T - t0)) : 0;
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(HB_K, HB_M));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(HB_K, HB_M));
    auto tX = tensor((device half *) (A + J.gh_off) + (ulong) (on ? t0 : 0) * 64, dextents<int32_t, 2>(64, nt), array<int, 2>({1, 64}));
    SGOP(mm);
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    const uint widx = tid * 16, rr = widx / HB_K, wkk = widx % HB_K;
    device const uchar * wrow = PB + (rr < 32 ? J.gain_w : J.forget_w) + (cb * 32 + (rr & 31)) * 128;
    if (tid < 128) {
        half wv[16];
        deq16(wrow, 30, wkk, wv);
        for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[0] + widx))[i] = ((thread half4 *) wv)[i];
        deq16(wrow, 30, 32 + wkk, wv);
        for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[1] + widx))[i] = ((thread half4 *) wv)[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!on) return;
    { auto sx = tX.slice(0, 0); mm.run(sx, tA0, cT); }
    { auto sx = tX.slice(32, 0); mm.run(sx, tA1, cT); }
    for (short q = 0; q < 4; ++q) {
        const auto ix = cT.get_multidimensional_index(4 * q);
        const uint r = ix[0], t = t0 + ix[1];
        if (t < a.T) {
            const uint o = t * E + cb * 32 + r;
            const float4 gn = float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]);
            const float4 fg = float4(cT[4 * q + 16], cT[4 * q + 17], cT[4 * q + 18], cT[4 * q + 19]);
            const float4 u = *(device const float4 *) (A + J.u_off + o), m = *(device const float4 *) (A + J.m_off + o);
            *(device float4 *) (A + J.raw_off + o) = u - tanh(fg) * u + tanh(gn) * m;
        }
    }
}
// accumulator layout probe for the kernels above: out[tid * 64 + i] = (row << 16 | token) of element i, out[tid * 64 + 63] = capacity
kernel void k_layout_sg(device float * A [[buffer(2)]], constant uint & out_off [[buffer(5)]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sa[HB_M * HB_K];
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(HB_K, HB_M));
    auto tX = tensor((device half *) A, dextents<int32_t, 2>(HB_K, 16), array<int, 2>({1, HB_K}));
    SGOP(mm);
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    device int * o = (device int *) (A + out_off) + tid * 64;
    for (short i = 0; i < cT.get_capacity() && i < 63; ++i) {
        if (cT.is_valid_element(i)) { const auto ix = cT.get_multidimensional_index(i); o[i] = (ix[0] << 16) | ix[1]; } else o[i] = -1;
    }
    o[63] = cT.get_capacity();
}

// ---- DSpark taps (inkserve, DSPARK.md): the target features the drafter reads, per position, into a ring of 1024
// positions x 4 features x E floats at ring_off (row = position % 1024). Rows below `from` are skipped. Source: n rows
// of E values from src_off (f32, or half if src_half). grid (E / 256, n), 256 threads.
struct TapArgs { uint src_off, src_half, n, pos0, from, feat, ring_off, pad; };
kernel void k_tapcopy(device float * A [[buffer(2)]], constant TapArgs & a [[buffer(5)]],
                      uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint r = tgp.y, pos = a.pos0 + r, c = tgp.x * 256 + tid;
    if (pos < a.from || c >= E) return;
    const float v = a.src_half ? float(((device const half *) (A + a.src_off))[(ulong) r * E + c]) : A[a.src_off + (ulong) r * E + c];
    A[a.ring_off + ((ulong) (pos % 1024) * 4 + a.feat) * E + c] = v;
}

// ======================= DSpark drafter (inkserve, DSPARK.md → Drafter spec) =======================
// Weights come from the drafter GGUF, bound at buffer 0 (W) while these kernels run; scratch lives in the activation
// arena (buffer 2). Context rings: per draft layer K and V, half [1024 slots][8 heads][128], slot = position % 1024,
// with the position each slot holds in a shared u32 array (DNONE = empty).
#define DNONE 0xFFFFFFFFu
#define DQ_HEADS 24
#define DKV_HEADS 8
#define DROPE_LOG2 19.931568569324174f     // log2(1e6), the RoPE base

// tap mix for n <= 8 new positions: rows of draft layer l (l < 3) as half to mix[l][r][E]; l = 3 (the Markov head)
// only for the newest position, as f32 to rows_off. grid (n, 4), 256 threads.
struct DCtxArgs { uint taps_off, alpha_off, n, pos0, mix_off, rows_off, norm0, norm1, norm2, norm3; float eps; uint pad; };
kernel void k_dctx_mix(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant DCtxArgs & a [[buffer(5)]],
                       uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    const uint r = tgp.x, l = tgp.y, pos = a.pos0 + r;
    if (l == 3 && r != a.n - 1) return;
    device const float * al = (device const float *) (W + a.alpha_off) + l * 4;
    device const float * tp = A + a.taps_off + (ulong) (pos % 1024) * 4 * E;
    float m[E / 256]; float ss = 0;
    for (int i = 0; i < E / 256; i++) {
        const uint c = tid + 256 * i;
        const float v = al[0] * tp[c] + al[1] * tp[E + c] + al[2] * tp[2 * E + c] + al[3] * tp[3 * E + c];
        m[i] = v; ss += v * v;
    }
    ss = tg_sum(ss, sh, sg, lane, 8);
    const float inv = rsqrt(ss / E + a.eps);
    const uint noff = l == 0 ? a.norm0 : l == 1 ? a.norm1 : l == 2 ? a.norm2 : a.norm3;
    device const float * nw = (device const float *) (W + noff);
    for (int i = 0; i < E / 256; i++) {
        const uint c = tid + 256 * i;
        const float v = m[i] * inv * nw[c];
        if (l < 3) ((device half *) (A + a.mix_off))[((ulong) l * 8 + r) * E + c] = half(v);
        else A[a.rows_off + c] = v;
    }
}

// RMSNorm over 128 values held one per thread (128 threads); returns this thread's normalized value
inline float drms128(float v, threadgroup float * sh, ushort sg, ushort lane, float eps) {
    const float ss = tg_sum(v * v, sh, sg, lane, 4);
    return v * rsqrt(ss / 128.0f + eps);
}
// RoPE (rotate-half pairing d <-> d +- 64) of one 128-value head held one per thread, at position pos
inline float drope(float v, threadgroup float * hv, ushort d, float pos) {
    hv[d] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = d % 64;
    const float th = pos * exp2(-DROPE_LOG2 * float(2 * i) / 128.0f);
    const float c = cos(th), s = sin(th);
    const float o = d < 64 ? hv[d] * c - hv[d + 64] * s : hv[d] * c + hv[d - 64] * s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return o;
}

// context keys/values of n positions into one layer's rings: k = RoPE(RMSNorm_head(kc) * k_norm_ctx), v = vc.
// grid (n, 8), 128 threads.
struct DFinArgs { uint kc_off, vc_off, n, pos0, knorm_off, ring_k, ring_v, pos_off; float eps; uint pad0, pad1, pad2; };
kernel void k_dctx_fin(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant DFinArgs & a [[buffer(5)]],
                       uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32]; threadgroup float hv[128];
    const uint r = tgp.x, h = tgp.y, pos = a.pos0 + r, slot = pos % 1024;
    float k = A[a.kc_off + r * (DKV_HEADS * 128) + h * 128 + tid];
    k = drms128(k, sh, sg, lane, a.eps) * ((device const float *) (W + a.knorm_off))[tid];
    k = drope(k, hv, tid, float(pos));
    ((device half *) (A + a.ring_k))[((ulong) slot * DKV_HEADS + h) * 128 + tid] = half(k);
    ((device half *) (A + a.ring_v))[((ulong) slot * DKV_HEADS + h) * 128 + tid] = half(A[a.vc_off + r * (DKV_HEADS * 128) + h * 128 + tid]);
    if (h == 0 && tid == 0) ((device uint *) (A + a.pos_off))[slot] = pos;
}

// rows 1..7 of the block: the mask embedding. grid (7, E / 256), 256 threads.
struct DFillArgs { uint h_off, mask_off, pad0, pad1; };
kernel void k_dfill(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant DFillArgs & a [[buffer(5)]],
                    uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint c = tgp.y * 256 + tid;
    if (c < E) A[a.h_off + (tgp.x + 1) * E + c] = ((device const float *) (W + a.mask_off))[c];
}

// RMSNorm of rows r0.. of an [8][E] f32 array: half rows to h_out (from row 0), f32 rows to f_out if it is not DNONE.
// grid (n), 256 threads.
struct DRmsArgs { uint in_off, r0, w_off, h_out, f_out; float eps; uint pad0, pad1; };
kernel void k_drms(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant DRmsArgs & a [[buffer(5)]],
                   uint r [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    device const float * x = A + a.in_off + (ulong) (a.r0 + r) * E;
    float v[E / 256]; float ss = 0;
    for (int i = 0; i < E / 256; i++) { v[i] = x[tid + 256 * i]; ss += v[i] * v[i]; }
    ss = tg_sum(ss, sh, sg, lane, 8);
    const float inv = rsqrt(ss / E + a.eps);
    device const float * w = (device const float *) (W + a.w_off);
    for (int i = 0; i < E / 256; i++) {
        const uint c = tid + 256 * i;
        const float y = v[i] * inv * w[c];
        ((device half *) (A + a.h_out))[(ulong) r * E + c] = half(y);
        if (a.f_out != DNONE) A[a.f_out + (ulong) r * E + c] = y;
    }
}

// per-head RMSNorm and RoPE of the block's queries (heads 0..23) and keys (24..31), in place, at positions n + row.
// grid (8, 32), 128 threads.
struct DQkArgs { uint q_off, k_off, qn_off, kn_off, n; float eps; uint pad0, pad1; };
kernel void k_dqk(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant DQkArgs & a [[buffer(5)]],
                  uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32]; threadgroup float hv[128];
    const uint r = tgp.x, h = tgp.y;
    const bool isq = h < DQ_HEADS;
    device float * p = isq ? A + a.q_off + r * (DQ_HEADS * 128) + h * 128 : A + a.k_off + r * (DKV_HEADS * 128) + (h - DQ_HEADS) * 128;
    float v = drms128(p[tid], sh, sg, lane, a.eps) * ((device const float *) (W + (isq ? a.qn_off : a.kn_off)))[tid];
    v = drope(v, hv, tid, float(a.n + r));
    p[tid] = v;
}

// block attention: query (row r, head h) over the context ring (positions in [n - 1023, n)) and the 8 block keys, both
// directions, scale 1/sqrt(128), no sink. Output half [8][24 * 128]. grid (24, 8), 128 threads.
struct DAttnArgs { uint q_off, kb_off, vb_off, ring_k, ring_v, pos_off, n, out_off; };
kernel void k_dattn(device float * A [[buffer(2)]], constant DAttnArgs & a [[buffer(5)]],
                    uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float qs[128]; threadgroup float sc[1024 + 8]; threadgroup float sh[32];
    const uint h = tgp.x, r = tgp.y, g = h / 3;
    qs[tid] = A[a.q_off + r * (DQ_HEADS * 128) + h * 128 + tid] * 0.08838834764831845f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uint * P = (device const uint *) (A + a.pos_off);
    device const half * RK = (device const half *) (A + a.ring_k);
    device const half * RV = (device const half *) (A + a.ring_v);
    float mx = -INFINITY;
    for (uint j = tid; j < 1024 + 8; j += 128) {
        float s = -INFINITY;
        if (j < 1024) {
            const uint p = P[j];
            if (p != DNONE && p < a.n && p + 1023 >= a.n) {
                device const half * k = RK + ((ulong) j * DKV_HEADS + g) * 128;
                float acc = 0;
                for (int d = 0; d < 128; d += 4) acc += dot(float4(qs[d], qs[d + 1], qs[d + 2], qs[d + 3]), float4(*(device const half4 *) (k + d)));
                s = acc;
            }
        } else {
            device const float * k = A + a.kb_off + (j - 1024) * (DKV_HEADS * 128) + g * 128;
            float acc = 0;
            for (int d = 0; d < 128; d++) acc += qs[d] * k[d];
            s = acc;
        }
        sc[j] = s; mx = max(mx, s);
    }
    mx = tg_max(mx, sh, sg, lane, 4);
    float ls = 0;
    for (uint j = tid; j < 1024 + 8; j += 128) { const float e = sc[j] == -INFINITY ? 0.0f : exp(sc[j] - mx); sc[j] = e; ls += e; }
    ls = tg_sum(ls, sh, sg, lane, 4);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = 0;
    for (uint j = 0; j < 1024; j++) { const float pj = sc[j]; if (pj != 0.0f) o += pj * float(RV[((ulong) j * DKV_HEADS + g) * 128 + tid]); }
    for (uint j = 0; j < 8; j++) o += sc[1024 + j] * A[a.vb_off + j * (DKV_HEADS * 128) + g * 128 + tid];
    ((device half *) (A + a.out_off))[r * (DQ_HEADS * 128) + h * 128 + tid] = half(o / ls);
}

// y += x over n floats; and act = SiLU(g) * u as half over n values. grid (ceil(n / 256)), 256 threads.
struct DEwArgs { uint a_off, b_off, c_off, n; };
kernel void k_dadd(device float * A [[buffer(2)]], constant DEwArgs & a [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) A[a.a_off + i] += A[a.b_off + i];
}
kernel void k_dsilu(device float * A [[buffer(2)]], constant DEwArgs & a [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) { const float g = A[a.a_off + i]; ((device half *) (A + a.c_off))[i] = half(g / (1.0f + exp(-g)) * A[a.b_off + i]); }
}

// Markov head, step s: in = [state, w1[prev], hidden_s, rows_f] (5120). prev = toks[prev_index]. 1 threadgroup, 256.
struct DMInArgs { uint in_off, state_off, w1_off, hid_off, rows_off, prev_index, pad0, pad1; };
kernel void k_dmin(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], device int * toks [[buffer(6)]],
                   constant DMInArgs & a [[buffer(5)]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint prev = (uint) toks[a.prev_index];
    device const ushort * w1 = (device const ushort *) (W + a.w1_off) + (ulong) prev * 256;
    A[a.in_off + tid] = A[a.state_off + tid];
    A[a.in_off + 256 + tid] = bf16f(w1[tid]);
    for (uint c = tid; c < E; c += 256) { A[a.in_off + 512 + c] = A[a.hid_off + c]; A[a.in_off + 512 + E + c] = A[a.rows_off + c]; }
}
// single-vector BF16 matvec + optional f32 bias: out[row] = W[row] . x (+ b[row]). grid (rows / 4), 128 threads.
struct DMvArgs { uint w_off, b_off, x_off, out_off, rows, k, pad0, pad1; };
kernel void k_dmv_bf16(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant DMvArgs & a [[buffer(5)]],
                       uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 4 + sg;
    if (row >= a.rows) return;
    device const ushort * w = (device const ushort *) (W + a.w_off) + (ulong) row * a.k;
    float s = 0;
    for (uint i = lane; i < a.k; i += 32) s += bf16f(w[i]) * A[a.x_off + i];
    s = simd_sum(s);
    if (lane == 0) A[a.out_off + row] = s + (a.b_off != DNONE ? ((device const float *) (W + a.b_off))[row] : 0.0f);
}
// the RNN update: gate = sigmoid(p0), state = gate * state + (1 - gate) * tanh(p1), latent = tanh(p2). 256 threads.
struct DMGateArgs { uint p_off, state_off, lat_off, pad; };
kernel void k_dmgate(device float * A [[buffer(2)]], constant DMGateArgs & a [[buffer(5)]], ushort tid [[thread_index_in_threadgroup]]) {
    const float g = 1.0f / (1.0f + exp(-A[a.p_off + tid]));
    A[a.state_off + tid] = g * A[a.state_off + tid] + (1.0f - g) * tanh(A[a.p_off + 256 + tid]);
    A[a.lat_off + tid] = tanh(A[a.p_off + 512 + tid]);
}
// logits_s += Markov bias + folded bias, over rows values. grid (ceil(rows / 256)), 256 threads.
struct DMFinArgs { uint logit_off, bias_off, fb_off, rows; };
kernel void k_dmfin(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant DMFinArgs & a [[buffer(5)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < a.rows) A[a.logit_off + i] += A[a.bias_off + i] + ((device const float *) (W + a.fb_off))[i];
}
// the Markov state starts at zero for each block; the context ring's slots start empty
kernel void k_dzero(device float * A [[buffer(2)]], constant DEwArgs & a [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) ((device uint *) (A + a.a_off))[i] = a.b_off;
}

// ---- standard speculative sampling for a verify pass (DSPARK.md): for draft x_i ~ q_i at position q0 + i, accept with
// probability min(1, p_i(x_i) / q_i(x_i)); on rejection, sample the residual max(0, p_i - q_i). p and q are softmax of
// softcap(logits) / T (the think token banned when the budget ended thinking, as k_gumbel does). Slot i
// (toks[slot0 + i]) ends up holding the target's token for position q0 + 1 + i: x_i if accepted, else the residual
// draw. Exact in distribution. Noise: the residual uses a second Gumbel stream, the acceptance test a uniform stream.
struct SpecArgs { SampArgs s; uint p_off, q_off, res_off, lse_off, depth, q0, slot0, rows; };
inline float spec_z(float x, constant SampArgs & s) { return (s.softcap > 0.0f ? s.softcap * precise::tanh(x / s.softcap) : x) / s.temp; }
inline float spec_u(uint seed_lo, uint seed_hi, uint pos, uint salt) {
    const uint h = mix32(mix32(mix32(mix32(salt ^ 0x9e3779b9u) ^ pos) ^ seed_lo ^ 0xa5a5a5a5u) ^ seed_hi);
    return (float(h >> 9) + 0.5f) * (1.0f / 8388608.0f);
}
// log-sum-exp of z over the vocabulary: row (i, which) with which 0 = p (verify logits), 1 = q (draft logits).
// grid (depth + 1, 2), 1024 threads.
kernel void k_slse(device float * A [[buffer(2)]], device int * toks [[buffer(6)]], constant SpecArgs & a [[buffer(5)]],
                   uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    const uint i = tgp.x, which = tgp.y;
    if (which == 1 && i >= a.depth) return;
    device const float * z = A + (which ? a.q_off : a.p_off) + (ulong) i * a.rows;
    const bool ban = toks[a.s.state + 2] != 0;
    float m = -INFINITY;
    for (uint v = tid; v < a.s.n; v += 1024) if (!(ban && v == a.s.think)) m = max(m, spec_z(z[v], a.s));
    m = tg_max(m, sh, sg, lane, 32);
    float sum = 0;
    for (uint v = tid; v < a.s.n; v += 1024) if (!(ban && v == a.s.think)) sum += exp(spec_z(z[v], a.s) - m);
    sum = tg_sum(sum, sh, sg, lane, 32);
    if (tid == 0) A[a.lse_off + i * 2 + which] = m + log(sum);
}
// residual weights + Gumbel noise (stream 2) for rows i < depth, ready for argmax. grid (ceil(n / 256), depth).
kernel void k_sresid(device float * A [[buffer(2)]], device int * toks [[buffer(6)]], constant SpecArgs & a [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint v = tgp.x * 256 + tid, i = tgp.y;
    if (v >= a.s.n) return;
    const bool ban = toks[a.s.state + 2] != 0 && v == a.s.think;
    const float lp = spec_z(A[a.p_off + (ulong) i * a.rows + v], a.s) - A[a.lse_off + i * 2];
    const float lq = spec_z(A[a.q_off + (ulong) i * a.rows + v], a.s) - A[a.lse_off + i * 2 + 1];
    const float r = exp(lp) - exp(lq);
    float w = -INFINITY;
    if (!ban && r > 0.0f) {
        const uint h = mix32(mix32(mix32(mix32(v ^ 0x85ebca6bu) ^ (a.q0 + i)) ^ a.s.seed_lo ^ 0x5a5a5a5au) ^ a.s.seed_hi);
        const float u = (float(h >> 9) + 0.5f) * (1.0f / 8388608.0f);
        w = log(r) - precise::log(-precise::log(u));
    }
    A[a.res_off + (ulong) i * a.rows + v] = w;
}
// the acceptance test for each draft: if u < p(x) / q(x), the slot gets x (otherwise it keeps the residual draw).
// grid (depth), 32 threads.
kernel void k_sdecide(device float * A [[buffer(2)]], device int * toks [[buffer(6)]], constant SpecArgs & a [[buffer(5)]],
                      uint i [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    const uint x = (uint) toks[a.q0 + 1 + i];
    const float lp = spec_z(A[a.p_off + (ulong) i * a.rows + x], a.s) - A[a.lse_off + i * 2];
    const float lq = spec_z(A[a.q_off + (ulong) i * a.rows + x], a.s) - A[a.lse_off + i * 2 + 1];
    const float u = spec_u(a.s.seed_lo, a.s.seed_hi, a.q0 + i, 0x7fffffffu);
    if (u < exp(min(0.0f, lp - lq))) toks[a.slot0 + i] = (int) x;
}
// f32 row copy (the drafter keeps its logits before the Gumbel noise). grid (ceil(n / 256)), 256 threads.
kernel void k_dcopyf(device float * A [[buffer(2)]], constant DEwArgs & a [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) A[a.b_off + i] = A[a.a_off + i];
}
