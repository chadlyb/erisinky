//! `inkling serve`: a long-lived engine for a front end (inkserve) that talks to it over stdin and stdout.
//!
//! One request runs at a time. The protocol is line-based:
//!
//! front end -> engine
//!   `G key=val ... len=N` then N bytes of prompt text, with control tokens written literally (`<|message_user|>`)
//!   `X` while a generation runs: stop it
//! engine -> front end
//!   `R ctx=N vocab=N eos=ID` once, when the model is loaded
//!   `P prompt=N cached=N us=N` after the prefill (cached: prompt tokens reused from the previous request)
//!   `T ID s|n HEX` per generated token (s: control token; HEX: the token's bytes)
//!   `E REASON gen=N think=N us=N` at the end. REASON: stop | length | cancel | context
//!   `F MESSAGE` for a malformed request
//!
//! Request keys: max (new tokens), temp, top_p, top_k (0 = off), seed, budget (thinking tokens; 0 = off).
//!
//! Prefix cache. The full-attention K/V only grows, so any prefix of it stays valid. The sliding-window rings and the
//! convolution state are overwritten as the context advances, so going back needs a snapshot of them. The engine keeps
//! one, taken two tokens before the end of each prompt. A new prompt then reuses:
//!   - the whole live context, if the prompt extends it (the client sent back exactly what was generated), or
//!   - the snapshot, if the prompt extends that (the client re-rendered the reply differently, e.g. without reasoning),
//!   - nothing otherwise.
//!
//! Thinking budget. After `budget` thinking tokens the engine emits `<|end_message|>` itself, which closes the thinking
//! block, and bans `<|content_thinking|>` for the rest of the reply. The model then writes text or calls a tool.
//!
//! Sampling. Without top-k or top-p (the model's defaults), tokens are drawn on the GPU and the next step is committed
//! before the CPU reads the token (`gen_chained`), so decoding runs at the engine's GPU-chained speed. Otherwise, or with
//! INKSERVE_CPU_SAMPLE=1, the CPU samples between steps (`Pick`). The two draw from the same distribution, with different
//! random streams. INKSERVE_PROF=1 prints the CPU loop's per-token times to stderr.
//!
//! DSpark (DSPARK.md). Started with a drafter (`inkling serve --ref Inkling-2-Mini-dspark.gguf`), the GPU path decodes
//! speculatively instead (`gen_dspark`): the drafter proposes DEPTH tokens, one verify pass checks them, and standard
//! speculative sampling keeps the output distribution exact. INKSERVE_DSPARK_DEPTH sets the maximum depth (default 3).

use crate::engine::{Drafter, Engine, SampArgs, EXT, VERIFY_BATCH};
use crate::session::Session;
use crate::tokenizer::Tokenizer;
use objc2_metal::MTLCommandBuffer;
use std::io::{BufRead, Read, Write};
use std::time::Instant;

struct Snap { n: usize, data: (Vec<u8>, Vec<u8>) }

struct Req { max: usize, temp: f32, top_p: f32, top_k: usize, seed: u64, budget: usize, len: usize }

fn parse_req(line: &str) -> Result<Req, String> {
    let mut r = Req { max: 16384, temp: 1.0, top_p: 1.0, top_k: 0, seed: 0, budget: 0, len: usize::MAX };
    for kv in line.split_whitespace().skip(1) {
        let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad field {kv}"))?;
        let bad = || format!("bad value {kv}");
        match k {
            "max" => r.max = v.parse().map_err(|_| bad())?,
            "temp" => r.temp = v.parse().map_err(|_| bad())?,
            "top_p" => r.top_p = v.parse().map_err(|_| bad())?,
            "top_k" => r.top_k = v.parse().map_err(|_| bad())?,
            "seed" => r.seed = v.parse().map_err(|_| bad())?,
            "budget" => r.budget = v.parse().map_err(|_| bad())?,
            "len" => r.len = v.parse().map_err(|_| bad())?,
            _ => return Err(format!("unknown field {k}")),
        }
    }
    if r.len == usize::MAX { return Err("missing len".into()); }
    Ok(r)
}

/// Sampling over the whole vocabulary: softcap -> temperature -> top-k -> top-p -> draw. Temperature 0 is argmax.
struct Pick { state: u64 }

impl Pick {
    fn next_f32(&mut self) -> f32 {
        // splitmix64, as in sampler.rs
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    fn pick(&mut self, lg: &[f32], softcap: f32, r: &Req, ban: Option<u32>) -> u32 {
        let mut best = if ban == Some(0) { 1 } else { 0 };
        for (i, &x) in lg.iter().enumerate() {
            if Some(i as u32) != ban && x > lg[best] { best = i; }
        }
        if r.temp <= 0.0 { return best as u32; }
        let f = |x: f32| (if softcap > 0.0 { softcap * (x / softcap).tanh() } else { x }) / r.temp;
        let m = f(lg[best]);
        // tokens more than 30 nats below the best carry < e^-30 each; leaving them out changes nothing measurable.
        // The softcap is monotonic, so the cut maps back to a raw-logit threshold; the cheap test on the raw logit runs
        // first (with a margin), and only its survivors pay for tanh and the exact test, which picks the same set.
        let y = r.temp * (m - 30.0);
        let raw_cut = if softcap <= 0.0 { y } else if y <= -softcap { f32::NEG_INFINITY } else { softcap * (y / softcap).atanh() };
        let raw_cut = raw_cut - 1e-3 * raw_cut.abs() - 1e-3;
        let mut cand: Vec<(f32, u32)> = lg.iter().enumerate()
            .filter(|&(i, &x)| x > raw_cut && Some(i as u32) != ban)
            .filter_map(|(i, &x)| { let s = f(x) - m; if s > -30.0 { Some((s, i as u32)) } else { None } })
            .collect();
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if r.top_k > 0 { cand.truncate(r.top_k); }
        let p: Vec<f32> = cand.iter().map(|c| c.0.exp()).collect();
        let sum: f32 = p.iter().sum();
        let mut n = p.len();
        if r.top_p < 1.0 {
            let mut cum = 0.0;
            for (i, &v) in p.iter().enumerate() { cum += v / sum; if cum >= r.top_p { n = i + 1; break; } }
        }
        let total: f32 = p[..n].iter().sum();
        let mut x = self.next_f32() * total;
        for i in 0..n { x -= p[i]; if x <= 0.0 { return cand[i].1; } }
        cand[n - 1].1
    }
}

fn special_id(tk: &Tokenizer, n_vocab: usize, name: &str) -> u32 {
    (0..n_vocab as u32).find(|&i| tk.is_special(i) && tk.decode_bytes(i) == name.as_bytes())
        .unwrap_or_else(|| panic!("control token {name} not in the vocabulary"))
}

/// Run `toks` through the model after the session's context; the logits of the last one stay in the engine.
/// A single token goes through a decode step, which is cheaper than a one-token prefill. (At erisinky 3890bd5 a
/// one-token prefill also gave wrong logits; upstream pull request #2, merged here, fixes that.)
fn feed(eng: &Engine, sess: &mut Session, toks: &[u32]) {
    if toks.len() > 1 { sess.prefill(toks); return; }
    let pos = sess.n_past;
    eng.set_tokens(pos, toks);
    eng.gen_reserve(pos);
    eng.step(pos, true, false).waitUntilCompleted();
    sess.n_past += 1;
}

/// Generation with sampling on the GPU and one step in flight. While the GPU runs the step that consumes token q (and
/// draws token q + 1), the CPU streams token q and commits the next step, so the GPU never waits for the CPU. The draw is
/// exact (Gumbel-max over the whole vocabulary, softcap and temperature applied), and the thinking budget runs on the GPU
/// too (k_sampstate); the CPU mirrors it for the counts. The step for the stop token runs as well: the template writes
/// that token after every assistant message, so the state it leaves is the next prompt's, and `live` gets the token.
/// Returns the end reason; `gen` and `think_n` count as in the CPU loop.
#[allow(clippy::too_many_arguments)]
fn gen_chained(eng: &Engine, sess: &mut Session, live: &mut Vec<u32>, r: &Req, tk: &Tokenizer, think: u32, end: u32,
               input: &mut impl BufRead, out: &mut impl Write, gen: &mut usize, think_n: &mut usize) -> &'static str {
    let state = eng.max_ctx;                     // spare token slots past the context hold the sampler's state
    eng.set_tokens(state, &[0, 0, 0]);
    eng.samp.set(Some(SampArgs { n: eng.n_vocab as u32, seed_lo: r.seed as u32, seed_hi: (r.seed >> 32) as u32, think, end,
                                 budget: r.budget as u32, state: state as u32, temp: r.temp, softcap: eng.softcap, ..Default::default() }));
    let allowed = |gen: usize, q: usize| gen < r.max && q + 2 < eng.max_ctx;
    // commit the step that consumes toks[q]; growing the cache needs an idle GPU, so wait for `prev` first if it must grow
    let commit = |q: usize, prev: &crate::metal::CmdBuf| {
        if q >= eng.kv_capacity() { prev.waitUntilCompleted(); eng.gen_reserve(q); }
        eng.step(q, true, true)
    };
    let mut q = sess.n_past;                     // position of the next token to stream
    let first = eng.gpu.cmd();
    eng.encode_sample(&first, q - 1);
    first.commit();
    let mut ready = first;                       // completes once toks[q] is final
    let mut consuming = if allowed(0, q) { Some(commit(q, &ready)) } else { None };
    let mut thinking = false;
    let reason = loop {
        if !allowed(*gen, q) { break "length"; }
        ready.waitUntilCompleted();
        let tok = eng.token_at(q);
        let step = consuming.take().expect("the step for an allowed token is committed");
        sess.n_past = q + 1;
        live.push(tok);
        if tok == tk.eos { ready = step; break "stop"; }
        writeln!(out, "T {tok} {} {}", if tk.is_special(tok) { 's' } else { 'n' }, hex(&tk.decode_bytes(tok))).unwrap();
        out.flush().unwrap();
        *gen += 1;
        if tok == think { thinking = true; } else if tok == end { thinking = false; } else if thinking { *think_n += 1; }
        if cancelled(input) { ready = step; break "cancel"; }
        q += 1;
        if allowed(*gen, q) { consuming = Some(commit(q, &step)); }
        ready = step;
    };
    ready.waitUntilCompleted();
    eng.samp.set(None);
    reason
}

/// DSpark generation: speculative cycles on the GPU (DSPARK.md). Each cycle drafts `depth` tokens after the anchor (the
/// last token streamed, not yet fed through the model), runs one verify pass over anchor + drafts, and keeps the
/// accepted drafts plus the target's next token, which becomes the new anchor. The CPU walks the kept tokens in order and
/// applies what the GPU sampler state does in plain decoding: the thinking budget (forcing `end`, then banning `think`),
/// eos, the token limit. A token it changes or stops at cuts the cycle there, and `verify_rollback` drops the rest.
///
/// Depth and fall-back. A cycle pays off only if it keeps more tokens than plain steps would make in the same time.
/// The loop keeps a running acceptance of draft j given that drafts before it were accepted. Each cycle takes the depth
/// with the most expected tokens per unit of cost: 1 + the sum over j <= depth of the product of acceptances, divided by
/// the cycle's cost in plain steps (`COST`, from prof_verify at ~2.4K context). The costs are fixed, not timed, so the
/// choices depend only on the tokens, and a seed always gives the same reply. When after 8 cycles no depth beats a plain
/// step, it runs a stretch of plain steps (one in flight, as `gen_chained`): 32 tokens, doubling up to 128 while DSpark
/// keeps losing. Generation starts with 16 plain steps. INKSERVE_DSPARK_ALWAYS=1 runs every cycle at the full depth,
/// with no fall-back.
///
/// The last anchor goes through one decode step at the end, so the state matches plain decoding's (which also feeds the
/// stop token). The draws differ from `gen_chained`'s (another random stream), with the same distribution.
#[allow(clippy::too_many_arguments)]
fn gen_dspark(eng: &Engine, d: &Drafter, max_depth: usize, tap_hi: usize, sess: &mut Session, live: &mut Vec<u32>, r: &Req, tk: &Tokenizer,
              think: u32, end: u32, input: &mut impl BufRead, out: &mut impl Write, gen: &mut usize, think_n: &mut usize) -> &'static str {
    if r.max == 0 { return "length"; }
    let state = eng.max_ctx;                     // the sampler state slots: [thinking, thinking tokens, think banned]
    eng.set_tokens(state, &[0, 0, 0]);
    let sa = SampArgs { n: eng.n_vocab as u32, seed_lo: r.seed as u32, seed_hi: (r.seed >> 32) as u32, think, end,
                        budget: r.budget as u32, state: state as u32, temp: r.temp, softcap: eng.softcap, ..Default::default() };
    let samp = (r.temp > 0.0).then_some(sa);     // None: greedy drafts and verify
    let mut q = sess.n_past;                     // the anchor's position
    let (t0, prof) = (Instant::now(), std::env::var("INKSERVE_PROF").is_ok());
    let always = std::env::var("INKSERVE_DSPARK_ALWAYS").is_ok();
    // the first token, and the drafter's context: the taps of the last 1023 positions whose tap ring slots still hold
    // them (`tap_hi`: the highest position whose tap was written; an earlier reply may have run past this prompt's end)
    eng.samp.set(Some(sa));
    let cb = eng.gpu.cmd();
    eng.encode_sample(&cb, q - 1);
    cb.commit();
    eng.samp.set(None);
    eng.draft_reset();
    let c0 = (tap_hi + 1).max(q).saturating_sub(1023).min(q);
    let cb2 = eng.gpu.cmd();
    eng.encode_draft_commit(d, &cb2, c0, q - c0);
    cb2.commit();
    cb2.waitUntilCompleted();
    let (mut thinking, mut ban) = (false, false);
    let t_setup = t0.elapsed().as_secs_f64();
    // stream one token; Some(reason) if generation stops with it as the last
    let mut emit = |tok: u32, pos: usize, gen: &mut usize, think_n: &mut usize, thinking: &mut bool| -> Option<&'static str> {
        live.push(tok);
        if tok == tk.eos { return Some("stop"); }
        writeln!(out, "T {tok} {} {}", if tk.is_special(tok) { 's' } else { 'n' }, hex(&tk.decode_bytes(tok))).unwrap();
        out.flush().unwrap();
        *gen += 1;
        if tok == think { *thinking = true; } else if tok == end { *thinking = false; } else if *thinking { *think_n += 1; }
        if *gen >= r.max || pos + 3 >= eng.max_ctx { return Some("length"); }
        None
    };
    let mut reason = emit(eng.token_at(q), q, gen, think_n, &mut thinking);
    let mut pending = 0usize;                    // positions fed since the drafter last saw the taps
    let mut inflight: Option<crate::metal::CmdBuf> = None;     // a plain step that already fed the anchor
    // the fall-back's measurements: plain step ms, and averages of tokens and ms per cycle
    let (mut use_ds, mut stretch, mut n_ds) = (always, 16usize, 0usize);
    // cycle ms over plain step ms by depth (prof_verify at 2.4K context plus the draft), until measured
    // cycle cost in plain steps by depth: verify pass + draft over a decode step (prof_verify at ~2.4K context)
    const COST: [f64; 8] = [1.0, 1.64, 1.88, 2.22, 2.6, 3.0, 3.4, 3.8];
    let (mut step_ms, cost, mut acc_p) = (0f64, COST, [0.7f64, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5]);
    // expected tokens per plain step of a cycle at depth dd
    let rate = |dd: usize, acc_p: &[f64; 7], cost: &[f64; 8]| {
        let (mut e, mut pr) = (1.0, 1.0);
        for a in &acc_p[..dd] { pr *= a; e += pr; }
        e / cost[dd]
    };
    let mut by_depth = [0usize; 8];
    let (mut cycles, mut ds_toks, mut t_gpu, mut plain_toks) = (0usize, 0usize, 0f64, 0usize);
    while reason.is_none() {
        if !use_ds {
            // a stretch of plain steps, one in flight: the step for position i feeds token i and draws token i + 1
            eng.samp.set(Some(sa));
            let (tp, q_in) = (Instant::now(), q);
            if q >= eng.kv_capacity() { eng.gen_reserve(q); }
            let mut cur = eng.step(q, true, true);
            loop {
                let more = q + 1 - q_in < stretch && q + 4 < eng.max_ctx;
                let next = more.then(|| {
                    if q + 1 >= eng.kv_capacity() { cur.waitUntilCompleted(); eng.gen_reserve(q + 1); }
                    eng.step(q + 1, true, true)
                });
                cur.waitUntilCompleted();
                q += 1;
                reason = emit(eng.token_at(q), q, gen, think_n, &mut thinking);
                if reason.is_none() && cancelled(input) { reason = Some("cancel"); }
                match next {
                    Some(nx) if reason.is_some() => { inflight = Some(nx); break; }
                    Some(nx) => cur = nx,
                    None => break,
                }
            }
            eng.samp.set(None);
            // the GPU sampler state ran these tokens (its forced end and ban included)
            thinking = eng.token_at(state) != 0;
            ban = eng.token_at(state + 2) != 0;
            pending += q - q_in;
            plain_toks += q - q_in;
            step_ms = tp.elapsed().as_secs_f64() * 1e3 / (q - q_in) as f64;
            sess.n_past = q;
            (use_ds, n_ds) = (true, 0);
            continue;
        }
        let depth = if always { max_depth } else {
            (1..=max_depth).max_by(|&a, &b| rate(a, &acc_p, &cost).total_cmp(&rate(b, &acc_p, &cost))).unwrap()
        };
        if q + depth + 3 >= eng.max_ctx { reason = Some("length"); break; }
        if q + depth >= eng.kv_capacity() { eng.gen_reserve(q + depth); }
        let saved = eng.swa_save(q, depth + 1);
        let cb = eng.gpu.cmd();
        if pending > 0 { eng.encode_draft_commit(d, &cb, q - pending, pending); }
        eng.encode_draft(d, &cb, q, depth, samp);
        eng.encode_verify(&cb, q, depth + 1);
        match samp { Some(sa) => eng.encode_verify_spec(&cb, q, depth, sa), None => eng.encode_verify_sample(&cb, q, depth + 1, None) }
        cb.commit();
        cb.waitUntilCompleted();
        t_gpu += (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3;
        let tgt: Vec<u32> = (0..=depth).map(|i| eng.verify_token(i)).collect();
        let mut acc = 0;
        while acc < depth && eng.token_at(q + 1 + acc) == tgt[acc] { acc += 1; }
        // walk the kept tokens; k = how many positions past the anchor stay
        let mut k = 0;
        while k <= acc {
            let mut tok = tgt[k];
            let forced = thinking && r.budget > 0 && *think_n >= r.budget;
            if forced { tok = end; ban = true; }
            k += 1;
            reason = emit(tok, q + k, gen, think_n, &mut thinking);
            eng.set_tokens(q + k, &[tok]);
            if forced || reason.is_some() { break; }
        }
        eng.verify_rollback(&saved, q, depth + 1, k);
        pending = k;
        q += k;
        sess.n_past = q;
        eng.set_tokens(state, &[thinking as u32, *think_n as u32, ban as u32]);
        if reason.is_none() && cancelled(input) { reason = Some("cancel"); }
        // draft j was seen if all before it were accepted
        for (j, a) in acc_p.iter_mut().enumerate().take((acc + 1).min(depth)) { *a += 0.1 * ((j < acc) as u32 as f64 - *a); }
        by_depth[depth] += 1;
        (n_ds, cycles, ds_toks) = (n_ds + 1, cycles + 1, ds_toks + k);
        if !always && n_ds > 8 && (1..=max_depth).all(|dd| rate(dd, &acc_p, &cost) < 1.0) {
            stretch = if stretch < 32 { 32 } else { (stretch * 2).min(128) };
            use_ds = false;
        } else if n_ds == 33 {
            stretch = 16;                        // DSpark is winning: the next loss starts with a short stretch again
        }
    }
    // feed the last anchor (the stop token included, as gen_chained does), unless a plain step already did
    match inflight {
        Some(cb) => cb.waitUntilCompleted(),
        None => {
            if q >= eng.kv_capacity() { eng.gen_reserve(q); }
            eng.step(q, true, false).waitUntilCompleted();
        }
    }
    sess.n_past = q + 1;
    if prof {
        let c = cycles.max(1) as f64;
        eprintln!("[dspark prof] ctx={q} setup {:.1} ms; {cycles} cycles, {:.2} tokens per cycle, GPU {:.2} ms per cycle; {plain_toks} plain tokens \
                   ({step_ms:.2} ms each, last stretch); {:.1} ms in all; cycles by depth {:?}", t_setup * 1e3, ds_toks as f64 / c, t_gpu / c, t0.elapsed().as_secs_f64() * 1e3, &by_depth[1..=max_depth]);
    }
    reason.unwrap()
}

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }

/// True if the front end sent `X` (stop). Checks stdin without blocking.
fn cancelled(input: &mut impl BufRead) -> bool {
    let mut fds = libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 };
    if unsafe { libc::poll(&mut fds, 1, 0) } <= 0 { return false; }
    let mut line = String::new();
    input.read_line(&mut line).unwrap_or(0) == 0 || line.trim() == "X"
}

pub fn run(model: &str, ctx: usize, drafter: &str) {
    let eng = Engine::load(model, ctx);
    let dspark = (!drafter.is_empty()).then(|| Drafter::load(&eng.gpu, drafter));
    let depth: usize = std::env::var("INKSERVE_DSPARK_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    // taps for the drafter, from each prompt's last 1024 positions (set per request below)
    if dspark.is_some() { eng.taps_on.set(true); }
    let tk = Tokenizer::load(&eng.g);
    // The rings reach their full size (one window per slot) now, so a snapshot always has the live layout.
    eng.reserve(EXT.min(ctx - 1), false);
    let (think, end) = (special_id(&tk, eng.n_vocab, "<|content_thinking|>"), special_id(&tk, eng.n_vocab, "<|end_message|>"));
    let mut sess = Session::new(&eng, &tk);
    let mut live: Vec<u32> = Vec::new();     // tokens whose state the engine holds, in order
    let mut snap: Option<Snap> = None;
    let mut tap_hi = 0usize;                 // the highest position whose tap (DSpark) was written
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut out = std::io::stdout().lock();
    writeln!(out, "R ctx={} vocab={} eos={}", eng.max_ctx, eng.n_vocab, tk.eos).unwrap();
    out.flush().unwrap();

    loop {
        let mut line = String::new();
        if input.read_line(&mut line).unwrap_or(0) == 0 { return; }      // front end gone
        let line = line.trim_end();
        if line.is_empty() || line == "X" { continue; }                    // a stop that arrived after the reply ended
        let r = match parse_req(line) { Ok(r) => r, Err(e) => { writeln!(out, "F {e}").unwrap(); out.flush().unwrap(); continue; } };
        let mut body = vec![0u8; r.len];
        if input.read_exact(&mut body).is_err() { return; }
        let toks = tk.encode(&String::from_utf8_lossy(&body), true);
        if toks.is_empty() || toks.len() + 2 >= eng.max_ctx {
            writeln!(out, "P prompt={} cached=0 us=0\nE context gen=0 think=0 us=0", toks.len()).unwrap();
            out.flush().unwrap();
            continue;
        }

        // ---- prefix cache
        let t0 = Instant::now();
        let cached = if !live.is_empty() && toks.len() > live.len() && toks.starts_with(&live) {
            live.len()
        } else if let Some(s) = snap.as_ref().filter(|s| toks.len() > s.n && live.len() >= s.n && toks[..s.n] == live[..s.n]) {
            eng.restore(&s.data);
            live.truncate(s.n);
            sess.n_past = s.n;
            s.n
        } else {
            sess = Session::new(&eng, &tk);
            live.clear();
            snap = None;
            tap_hi = 0;
            0
        };
        let n = toks.len();
        eng.tap_from.set(n.saturating_sub(1024));
        if std::env::var("INKSERVE_DEBUG").as_deref() == Ok("nosplit") {
            // test hook: the whole uncached prompt as one prefill, no snapshot (the reference for proto_test.py)
            feed(&eng, &mut sess, &toks[cached..]);
        } else if n - cached >= 3 {
            // snapshot two tokens before the end, then prefill those two: prefill results don't depend on where the
            // prompt is split, so this matches a whole-prompt prefill bit for bit
            feed(&eng, &mut sess, &toks[cached..n - 2]);
            snap = Some(Snap { n: n - 2, data: eng.snapshot() });
            feed(&eng, &mut sess, &toks[n - 2..]);
        } else {
            feed(&eng, &mut sess, &toks[cached..]);
        }
        live.extend_from_slice(&toks[cached..]);
        tap_hi = tap_hi.max(n - 1);
        // h: a hash of the last position's logits, so tests can compare prefill paths bit for bit
        let h = eng.logits().iter().fold(0xcbf29ce484222325u64, |a, x| (a ^ x.to_bits() as u64).wrapping_mul(0x100000001b3));
        writeln!(out, "P prompt={n} cached={cached} us={} h={h:016x}", t0.elapsed().as_micros()).unwrap();
        out.flush().unwrap();

        // ---- generation, one step at a time (the logits come back to the CPU for sampling and the budget)
        let t1 = Instant::now();
        eng.release_batch();
        let mut pick = Pick { state: r.seed ^ 0x9E37_79B9_7F4A_7C15 };
        let (mut gen, mut think_n, mut thinking, mut budget_hit) = (0usize, 0usize, false, false);
        // the GPU path covers the default sampling (no top-k / top-p); INKSERVE_CPU_SAMPLE=1 forces the CPU loop
        let gpu = r.top_k == 0 && r.top_p >= 1.0 && !eng.tier_on() && std::env::var("INKSERVE_CPU_SAMPLE").is_err();
        let prof = std::env::var("INKSERVE_PROF").is_ok();
        let (mut t_pick, mut t_feed) = (0f64, 0f64);
        let reason = if let (true, Some(d)) = (gpu, dspark.as_ref()) {
            eng.ensure_batch(VERIFY_BATCH);      // the verify pass runs in the batch arena
            gen_dspark(&eng, d, depth, tap_hi, &mut sess, &mut live, &r, &tk, think, end, &mut input, &mut out, &mut gen, &mut think_n)
        } else if gpu {
            gen_chained(&eng, &mut sess, &mut live, &r, &tk, think, end, &mut input, &mut out, &mut gen, &mut think_n)
        } else { loop {
            if gen >= r.max || sess.n_past + 2 >= eng.max_ctx { break "length"; }
            let tok = if thinking && r.budget > 0 && think_n >= r.budget {
                budget_hit = true;
                end
            } else {
                let tp = Instant::now();
                let t = pick.pick(eng.logits(), eng.softcap, &r, if budget_hit { Some(think) } else { None });
                t_pick += tp.elapsed().as_secs_f64();
                t
            };
            if tok == tk.eos { break "stop"; }
            writeln!(out, "T {tok} {} {}", if tk.is_special(tok) { 's' } else { 'n' }, hex(&tk.decode_bytes(tok))).unwrap();
            out.flush().unwrap();
            gen += 1;
            if tok == think { thinking = true; } else if tok == end { thinking = false; } else if thinking { think_n += 1; }
            let tf = Instant::now();
            feed(&eng, &mut sess, &[tok]);
            t_feed += tf.elapsed().as_secs_f64();
            live.push(tok);
            if cancelled(&mut input) { break "cancel"; }
        } };
        tap_hi = tap_hi.max(sess.n_past.saturating_sub(1));
        if prof && gen > 0 {
            eprintln!("[serve prof] ctx={} gen={gen} pick={:.3}ms feed={:.3}ms per token, total {:.3}ms", sess.n_past, t_pick * 1e3 / gen as f64,
                      t_feed * 1e3 / gen as f64, t1.elapsed().as_secs_f64() * 1e3 / gen as f64);
        }
        writeln!(out, "E {reason} gen={gen} think={think_n} us={}", t1.elapsed().as_micros()).unwrap();
        out.flush().unwrap();
    }
}
