//! DSpark drafter (inkserve, DSPARK.md → Drafter spec): the bundled block drafter, converted by `inkconvert mtp`.
//! Its weights are a second mapped GGUF, bound at buffer 0 while its kernels run; its scratch and context rings live
//! in the activation arena (Arena::dsp). A cycle is: commit the newly accepted positions' taps into the context rings
//! (`encode_draft_commit`), then draft `depth` proposals into toks[n + 1 ..= n + depth] (`encode_draft`), which a verify
//! pass (`encode_verify`) checks.

use super::*;

const DL_N: usize = 3;
const DQ: usize = 24 * 128;
const DKV: usize = 8 * 128;
const DFF: usize = 9216;
const DRANK: usize = 256;
const DNONE: u32 = u32::MAX;
const DEPS: f32 = 1e-6;

struct DLayer {
    attn_norm: u32, ffn_norm: u32, ctx_norm: u32, q_norm: u32, k_norm: u32, k_norm_ctx: u32,
    wq: QW, wk: QW, wv: QW, wkc: QW, wvc: QW, wo: QW, gate: QW, up: QW, down: QW,
}

/// The drafter's weights: the mapped GGUF and byte offsets / matrices in it.
pub struct Drafter {
    pub g: Gguf,
    w: Buf,
    layers: Vec<DLayer>,
    mask: u32, norm: u32, alpha: u32,
    m_norm: u32, m_rnn: u32, m_rnn_b: u32, m_w1: u32, m_fold: QW, m_fold_b: u32,
}

impl Drafter {
    pub fn load(gpu: &Gpu, path: &str) -> Drafter {
        let g = Gguf::open(path).unwrap_or_else(|e| panic!("drafter {path}: {e}"));
        let w = gpu.buffer_nocopy(g.map, g.map_len);
        let off = |n: &str| g.tensor(n).file_off as u32;
        let layers = (0..DL_N).map(|l| {
            let n = |s: &str| format!("mtp.{l}.{s}");
            DLayer {
                attn_norm: off(&n("attn_norm")), ffn_norm: off(&n("ffn_norm")), ctx_norm: off(&n("ctx_norm")),
                q_norm: off(&n("q_norm")), k_norm: off(&n("k_norm")), k_norm_ctx: off(&n("k_norm_ctx")),
                wq: qw(&g, &n("wq")), wk: qw(&g, &n("wk")), wv: qw(&g, &n("wv")), wkc: qw(&g, &n("wk_ctx")), wvc: qw(&g, &n("wv_ctx")),
                wo: qw(&g, &n("wo")), gate: qw(&g, &n("gate")), up: qw(&g, &n("up")), down: qw(&g, &n("down")),
            }
        }).collect();
        assert!(g.file_len < u32::MAX as usize, "drafter offsets are 32-bit");
        let d = Drafter {
            layers, mask: off("mtp.mask_embed"), norm: off("mtp.norm"), alpha: off("mtp.alpha"),
            m_norm: off("mtp.markov.ctx_norm"), m_rnn: off("mtp.markov.rnn"), m_rnn_b: off("mtp.markov.rnn_bias"),
            m_w1: off("mtp.markov.w1"), m_fold: qw(&g, "mtp.markov.fold_w"), m_fold_b: off("mtp.markov.fold_b"),
            w, g,
        };
        eprintln!("[inkling] drafter {} ({:.0} MiB)", path.rsplit('/').next().unwrap_or(path), d.g.file_len as f64 / 1048576.0);
        d
    }
}

/// Offsets (f32 units) of the drafter's scratch inside the arena block at `base`.
#[derive(Clone, Copy)]
pub(super) struct DScratch {
    ring_k: [u32; DL_N], ring_v: [u32; DL_N], pos: u32,
    h: u32, x: u32, q: u32, kb: u32, vb: u32, att: u32, y: u32, g: u32, u: u32, act: u32,
    mix: u32, kc: u32, vc: u32, hid: u32, hidh: u32,
    m_state: u32, m_in: u32, m_p: u32, m_lat: u32, m_rows: u32, m_bias: u32,
    /// speculative sampling: the drafter's logits per proposal (before noise), residual rows, log-sum-exps
    qlog: u32, res: u32, lse: u32,
    pub total: u32,
}
impl DScratch {
    pub(super) fn new(base: u32, vocab_pad: usize) -> DScratch {
        let mut o = base;
        let mut take = |n: usize| { let r = o; o += ((n + 63) / 64 * 64) as u32; r };
        let ring = 1024 * DKV / 2;          // half
        let ring_k = [take(ring), take(ring), take(ring)];
        let ring_v = [take(ring), take(ring), take(ring)];
        let pos = take(1024);
        let h = take(8 * E); let x = take(8 * E / 2);
        let q = take(8 * DQ); let kb = take(8 * DKV); let vb = take(8 * DKV); let att = take(8 * DQ / 2); let y = take(8 * E);
        let g = take(8 * DFF); let u = take(8 * DFF); let act = take(8 * DFF / 2);
        let mix = take(DL_N * 8 * E / 2); let kc = take(8 * DKV); let vc = take(8 * DKV); let hid = take(8 * E); let hidh = take(8 * E / 2);
        let m_state = take(DRANK); let m_in = take(2 * DRANK + 2 * E); let m_p = take(3 * DRANK); let m_lat = take(DRANK); let m_rows = take(E);
        let m_bias = take(vocab_pad);
        let qlog = take(7 * vocab_pad); let res = take(7 * vocab_pad); let lse = take(64);
        DScratch { ring_k, ring_v, pos, h, x, q, kb, vb, att, y, g, u, act, mix, kc, vc, hid, hidh, m_state, m_in, m_p, m_lat, m_rows, m_bias, qlog, res, lse, total: o - base }
    }
}

#[repr(C)] #[derive(Clone, Copy, Default)] struct DCtxArgs { taps_off: u32, alpha_off: u32, n: u32, pos0: u32, mix_off: u32, rows_off: u32, norm: [u32; 4], eps: f32, pad: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)] struct DFinArgs { kc_off: u32, vc_off: u32, n: u32, pos0: u32, knorm_off: u32, ring_k: u32, ring_v: u32, pos_off: u32, eps: f32, pad: [u32; 3] }
#[repr(C)] #[derive(Clone, Copy, Default)] struct DFillArgs { h_off: u32, mask_off: u32, pad: [u32; 2] }
#[repr(C)] #[derive(Clone, Copy, Default)] struct DRmsArgs { in_off: u32, r0: u32, w_off: u32, h_out: u32, f_out: u32, eps: f32, pad: [u32; 2] }
#[repr(C)] #[derive(Clone, Copy, Default)] struct DQkArgs { q_off: u32, k_off: u32, qn_off: u32, kn_off: u32, n: u32, eps: f32, pad: [u32; 2] }
#[repr(C)] #[derive(Clone, Copy, Default)] struct DAttnArgs { q_off: u32, kb_off: u32, vb_off: u32, ring_k: u32, ring_v: u32, pos_off: u32, n: u32, out_off: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)] struct DEwArgs { a_off: u32, b_off: u32, c_off: u32, n: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)] struct DMInArgs { in_off: u32, state_off: u32, w1_off: u32, hid_off: u32, rows_off: u32, prev_index: u32, pad: [u32; 2] }
#[repr(C)] #[derive(Clone, Copy, Default)] struct DMvArgs { w_off: u32, b_off: u32, x_off: u32, out_off: u32, rows: u32, k: u32, pad: [u32; 2] }
#[repr(C)] #[derive(Clone, Copy, Default)] struct DMGateArgs { p_off: u32, state_off: u32, lat_off: u32, pad: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)] struct DMFinArgs { logit_off: u32, bias_off: u32, fb_off: u32, rows: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)] struct SpecArgs { s: SampArgs, p_off: u32, q_off: u32, res_off: u32, lse_off: u32, depth: u32, q0: u32, slot0: u32, rows: u32 }

impl Engine {
    /// Standard speculative sampling over the last verify pass at q (depth drafts + the bonus position): slot i of
    /// verify_token gets the target's token for position q + 1 + i (the draft if accepted, else a residual draw; the
    /// bonus a plain draw). Exact in distribution. Needs the drafter's logits from encode_draft with the same `samp`.
    pub fn encode_verify_spec(&self, cb: &CmdBuf, q: usize, depth: usize, samp: SampArgs) {
        assert!(depth + 1 <= 5);
        let s = self.ds();
        let rows = self.output.rows as usize;
        let e = Enc::new(cb);
        self.bind(&e);
        let a = SpecArgs { s: samp, p_off: self.ar.vlog, q_off: s.qlog, res_off: s.res, lse_off: s.lse, depth: depth as u32, q0: q as u32,
                           slot0: (self.max_ctx + 3) as u32, rows: rows as u32 };
        self.go(&e, C_ARGMAX, "k_slse", &a, (depth + 1, 2, 1), 1024);
        self.go(&e, C_ARGMAX, "k_sresid", &a, (self.n_vocab.div_ceil(256), depth, 1), 256);
        for i in 0..depth {
            let am = ArgmaxArgs { logits_off: s.res + (i * rows) as u32, n: self.n_vocab as u32, part_off: self.ar.amax, nparts: 128, out_index: (self.max_ctx + 3 + i) as u32, pad: [0; 3] };
            self.go(&e, C_ARGMAX, "k_argmax1", &am, (128, 1, 1), 1024);
            self.go(&e, C_ARGMAX, "k_argmax2", &am, (1, 1, 1), 32);
        }
        self.go(&e, C_ARGMAX, "k_sdecide", &a, (depth, 1, 1), 32);
        // the bonus position: a plain draw from p
        let lo = self.ar.vlog + (depth * rows) as u32;
        self.go(&e, C_ARGMAX, "k_gumbel", &SampArgs { logits_off: lo, pos: (q + depth) as u32, ..samp }, (self.n_vocab.div_ceil(256), 1, 1), 256);
        let am = ArgmaxArgs { logits_off: lo, n: self.n_vocab as u32, part_off: self.ar.amax, nparts: 128, out_index: (self.max_ctx + 3 + depth) as u32, pad: [0; 3] };
        self.go(&e, C_ARGMAX, "k_argmax1", &am, (128, 1, 1), 1024);
        self.go(&e, C_ARGMAX, "k_argmax2", &am, (1, 1, 1), 32);
        e.end();
    }

    fn ds(&self) -> DScratch { DScratch::new(self.ar.dsp, self.output.rows as usize) }

    /// Empty the drafter's context rings (a new conversation, or a restored snapshot the rings do not match).
    pub fn draft_reset(&self) {
        let s = self.ds();
        let cb = self.gpu.cmd();
        let e = Enc::new(&cb);
        self.bind(&e);
        self.go(&e, C_ARGMAX, "k_dzero", &DEwArgs { a_off: s.pos, b_off: DNONE, c_off: 0, n: 1024 }, (4, 1, 1), 256);
        e.end();
        cb.commit();
        cb.waitUntilCompleted();
    }

    /// Project the taps of positions p0 .. p0 + n (in the tap ring) into the drafter's context rings, 8 at a time;
    /// also the Markov head's row of the newest position.
    pub fn encode_draft_commit(&self, d: &Drafter, cb: &CmdBuf, p0: usize, n: usize) {
        if n == 0 { return; }
        let s = self.ds();
        let e = Enc::new(cb);
        self.bind(&e);
        e.buf(&d.w, 0, 0);
        let verify = self.verify.get();
        self.verify.set(true);                                    // k_mvT for the small projections
        let mut q = p0;
        while q < p0 + n {
            let m = (p0 + n - q).min(8);
            let a = DCtxArgs { taps_off: self.ar.taps, alpha_off: d.alpha, n: m as u32, pos0: q as u32, mix_off: s.mix, rows_off: s.m_rows,
                               norm: [d.layers[0].ctx_norm, d.layers[1].ctx_norm, d.layers[2].ctx_norm, d.m_norm], eps: DEPS, pad: 0 };
            self.go(&e, C_ARGMAX, "k_dctx_mix", &a, (m, 4, 1), 256);
            let mt = m.max(2);                                    // k_mvT takes 2..=8 rows; a spare row is ignored
            for (l, dl) in d.layers.iter().enumerate() {
                let xh = s.mix + (l * 8 * E / 2) as u32;
                self.gemm_h(&e, C_ARGMAX, &[Self::hseg(&dl.wkc, xh, s.kc), Self::hseg(&dl.wvc, xh, s.vc)], mt);
                let f = DFinArgs { kc_off: s.kc, vc_off: s.vc, n: m as u32, pos0: q as u32, knorm_off: dl.k_norm_ctx,
                                   ring_k: s.ring_k[l], ring_v: s.ring_v[l], pos_off: s.pos, eps: DEPS, pad: [0; 3] };
                self.go(&e, C_ARGMAX, "k_dctx_fin", &f, (m, 8, 1), 128);
            }
            q += m;
        }
        self.verify.set(verify);
        e.buf(&self.w, 0, 0);
        e.end();
    }

    /// Draft `depth` (1..=7) proposals after the n committed positions: the anchor is toks[n], the proposals go to
    /// toks[n + 1 ..= n + depth]. With `samp` they are drawn with the target's Gumbel noise for those positions (so a
    /// proposal equals the target's draw whenever their argmaxes agree); without, greedy.
    pub fn encode_draft(&self, d: &Drafter, cb: &CmdBuf, n: usize, depth: usize, samp: Option<SampArgs>) {
        assert!((1..=7).contains(&depth));
        let s = self.ds();
        let rows = self.output.rows as usize;
        let e = Enc::new(cb);
        self.bind(&e);
        let verify = self.verify.get();
        self.verify.set(true);
        // block: row 0 = the anchor's target embedding (+ its norm), rows 1..7 = the mask embedding
        let ea = EmbedArgs { emb_lo: self.tok_embd.off as u32, emb_hi: (self.tok_embd.off >> 32) as u32, row_bytes: self.tok_embd.row_bytes,
                             norm_off: self.tok_norm, out0: s.h, out1: s.h, tok_index: n as u32, eps: self.eps };
        self.go(&e, C_ARGMAX, "k_embed", &ea, (1, 1, 1), 256);
        e.buf(&d.w, 0, 0);
        self.go(&e, C_ARGMAX, "k_dfill", &DFillArgs { h_off: s.h, mask_off: d.mask, pad: [0; 2] }, (7, E.div_ceil(256), 1), 256);
        let ew = |e: &Enc, k: &str, a: u32, b: u32, c: u32, cnt: usize| self.go(e, C_ARGMAX, k, &DEwArgs { a_off: a, b_off: b, c_off: c, n: cnt as u32 }, (cnt.div_ceil(256), 1, 1), 256);
        for (l, dl) in d.layers.iter().enumerate() {
            // attention
            self.go(&e, C_ARGMAX, "k_drms", &DRmsArgs { in_off: s.h, r0: 0, w_off: dl.attn_norm, h_out: s.x, f_out: DNONE, eps: DEPS, pad: [0; 2] }, (8, 1, 1), 256);
            self.gemm_h(&e, C_ARGMAX, &[Self::hseg(&dl.wq, s.x, s.q), Self::hseg(&dl.wk, s.x, s.kb), Self::hseg(&dl.wv, s.x, s.vb)], 8);
            self.go(&e, C_ARGMAX, "k_dqk", &DQkArgs { q_off: s.q, k_off: s.kb, qn_off: dl.q_norm, kn_off: dl.k_norm, n: n as u32, eps: DEPS, pad: [0; 2] }, (8, 32, 1), 128);
            let at = DAttnArgs { q_off: s.q, kb_off: s.kb, vb_off: s.vb, ring_k: s.ring_k[l], ring_v: s.ring_v[l], pos_off: s.pos, n: n as u32, out_off: s.att };
            self.go(&e, C_ARGMAX, "k_dattn", &at, (24, 8, 1), 128);
            self.gemm_h(&e, C_ARGMAX, &[Self::hseg(&dl.wo, s.att, s.y)], 8);
            ew(&e, "k_dadd", s.h, s.y, 0, 8 * E);
            // MLP
            self.go(&e, C_ARGMAX, "k_drms", &DRmsArgs { in_off: s.h, r0: 0, w_off: dl.ffn_norm, h_out: s.x, f_out: DNONE, eps: DEPS, pad: [0; 2] }, (8, 1, 1), 256);
            self.gemm_h(&e, C_ARGMAX, &[Self::hseg(&dl.gate, s.x, s.g), Self::hseg(&dl.up, s.x, s.u)], 8);
            ew(&e, "k_dsilu", s.g, s.u, s.act, 8 * DFF);
            self.gemm_h(&e, C_ARGMAX, &[Self::hseg(&dl.down, s.act, s.y)], 8);
            ew(&e, "k_dadd", s.h, s.y, 0, 8 * E);
        }
        // the proposal rows (block rows 1..=depth): final norm, then the target's output head
        let t = depth.max(2);
        self.go(&e, C_ARGMAX, "k_drms", &DRmsArgs { in_off: s.h, r0: 1, w_off: d.norm, h_out: s.hidh, f_out: s.hid, eps: DEPS, pad: [0; 2] }, (t, 1, 1), 256);
        e.buf(&self.w, 0, 0);
        let mut head = Self::hseg(&self.output, s.hidh, self.ar.vlog);
        head.ldo = self.output.rows;
        self.gemm_h(&e, C_ARGMAX, &[head], t);
        e.buf(&d.w, 0, 0);
        // Markov head: one step per proposal, each reading the one before
        self.go(&e, C_ARGMAX, "k_dzero", &DEwArgs { a_off: s.m_state, b_off: 0, c_off: 0, n: DRANK as u32 }, (1, 1, 1), 256);
        for st in 0..depth {
            let logit = self.ar.vlog + (st * rows) as u32;
            self.go(&e, C_ARGMAX, "k_dmin", &DMInArgs { in_off: s.m_in, state_off: s.m_state, w1_off: d.m_w1, hid_off: s.hid + (st * E) as u32,
                                                     rows_off: s.m_rows, prev_index: (n + st) as u32, pad: [0; 2] }, (1, 1, 1), 256);
            self.go(&e, C_ARGMAX, "k_dmv_bf16", &DMvArgs { w_off: d.m_rnn, b_off: d.m_rnn_b, x_off: s.m_in, out_off: s.m_p, rows: (3 * DRANK) as u32,
                                                        k: (2 * DRANK + 2 * E) as u32, pad: [0; 2] }, (3 * DRANK / 4, 1, 1), 128);
            self.go(&e, C_ARGMAX, "k_dmgate", &DMGateArgs { p_off: s.m_p, state_off: s.m_state, lat_off: s.m_lat, pad: 0 }, (1, 1, 1), 256);
            self.mv(&e, C_ARGMAX, &[Self::seg(&d.m_fold, s.m_lat, s.m_bias)]);
            self.go(&e, C_ARGMAX, "k_dmfin", &DMFinArgs { logit_off: logit, bias_off: s.m_bias, fb_off: d.m_fold_b, rows: rows as u32 }, (rows.div_ceil(256), 1, 1), 256);
            if let Some(sa) = samp {
                // keep q's logits for the acceptance test (speculative sampling), then draw the proposal
                self.go(&e, C_ARGMAX, "k_dcopyf", &DEwArgs { a_off: logit, b_off: s.qlog + (st * rows) as u32, c_off: 0, n: rows as u32 }, (rows.div_ceil(256), 1, 1), 256);
                let g = SampArgs { logits_off: logit, pos: (n + st) as u32, ..sa };
                self.go(&e, C_ARGMAX, "k_gumbel", &g, (self.n_vocab.div_ceil(256), 1, 1), 256);
            }
            let am = ArgmaxArgs { logits_off: logit, n: self.n_vocab as u32, part_off: self.ar.amax, nparts: 128, out_index: (n + st + 1) as u32, pad: [0; 3] };
            self.go(&e, C_ARGMAX, "k_argmax1", &am, (128, 1, 1), 1024);
            self.go(&e, C_ARGMAX, "k_argmax2", &am, (1, 1, 1), 32);
        }
        self.verify.set(verify);
        e.buf(&self.w, 0, 0);
        e.end();
    }
}
