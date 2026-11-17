"""The native FlashSCA host on threads PyTorch has not bound a CUDA context to, and the
forward's scheduling-instrument policy.

1. TMA descriptors are encoded with a driver-API call (umma_host.h:EncodeInputMap), which needs
   a context current on the calling host thread. The runtime binds the primary context lazily,
   and PyTorch's device guard skips cudaSetDevice when the thread's device already matches, so
   the autograd engine's device thread has no current context until its first runtime call.
   When flash_sca's own node is the graph root, the native backward's encode is that thread's
   first CUDA work: before the encoder bound the context itself this raised "could not encode
   TMA source: CUDA driver error 201" (CUDA_ERROR_INVALID_CONTEXT) at both leaf buckets, and
   the explicit flash_sca_fwd/flash_sca_bwd pair on the main thread passed. Each case runs in a
   fresh interpreter, so no earlier test can have bound the autograd thread's context; a
   pre-hook on the node records whether a context was current when the node started.
2. XATTN_SCA_ORDER and XATTN_SCA_LAYOUT=hf instrument the generic forward body. Like the
   backward (umma_prepare.h, Refusal::kOrderArm), a live instrument keeps every native forward
   body out of the way. At level 0 the pipelined body (umma_fwd_pipe.cuh) is the default at
   both leaf buckets, (256,256) and (128,128); a live instrument resolves the generic body
   instead (host.cc ResolveForwardBody). XATTN_SCA_UMMA_FWD=1 is the only way to the older
   native leaf, so the instrument can only conflict with an explicit =1, and it refuses there
   rather than timing a native body under an instrument it ignores.
3. The backward's diagnostic record is process-wide and stamped: the dump read on the
   caller's thread after torch.autograd.grad describes the backward the engine's thread ran
   (record_seq, record_launch_counts), not an empty per-thread default.
"""

import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

import xattn


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="native FlashSCA runs on compute capability 10.3",
)

KNOBS = ("XATTN_SCA_BWD", "XATTN_SCA_BWD_ARM", "XATTN_SCA_ORDER", "XATTN_SCA_LAYOUT",
         "XATTN_SCA_UMMA_FWD", "XATTN_SCA_FWD_PIPE")

# The regression cell: dense, B=2, N=67, H=2, C=32, BF16, level 0, autograd entry.
_FRESH_THREAD = r"""
import ctypes, json, sys
import torch, torch.nn.functional as F
import xattn, xattn_cuda

dim = int(sys.argv[1])
libcuda = ctypes.CDLL("libcuda.so.1")
seen = {}

def context_current():
    ctx = ctypes.c_void_p()
    rc = libcuda.cuCtxGetCurrent(ctypes.byref(ctx))
    return rc == 0 and bool(ctx.value)

def tags():
    return {tuple(r[:4]): int(r[4]) for r in xattn_cuda.ops._flash_sca_native_phase_tags()}

torch.manual_seed(0)
B, N, H, C = 2, 67, 2, 32
q = F.normalize(torch.randn(B, N, H, dim, device="cuda"), dim=-1).to(torch.bfloat16)
k = F.normalize(torch.randn(B, N, H, dim, device="cuda"), dim=-1).to(torch.bfloat16)
v = (F.silu(torch.randn(B, N, H, dim, device="cuda")) + 0.1).to(torch.bfloat16)
dy = torch.randn(B, N, H, dim, device="cuda").to(torch.bfloat16)
q.requires_grad_(); k.requires_grad_(); v.requires_grad_()
y = xattn.flash_sca(q, k, v, C)


def record_context(grads):
    # A node pre-hook must return None (or replacement gradients); it only observes here.
    seen.setdefault("context_before_node", context_current())
    return None


y.grad_fn.register_prehook(record_context)
before = tags()
out = {"dim": dim, "error": None}
try:
    dq, dk, dv = torch.autograd.grad(y, (q, k, v), dy)
    torch.cuda.synchronize()
    # The backward ran on the autograd engine's thread; the record is read on this one.
    rec = xattn_cuda.ops._flash_sca_last_backward_preparation()
    out["record"] = {"seq": rec["record_seq"], "on_caller_thread": rec["recorded_on_this_thread"],
                     "launch_counts": list(rec["record_launch_counts"]),
                     "counts_now": list(xattn_cuda.ops._flash_sca_native_launch_counts()),
                     "launched": rec["launched"], "prepared": rec["prepared"]}
    out["finite"] = all(bool(torch.isfinite(g.float()).all()) for g in (dq, dk, dv))
    # FP64 reference of the same call: chunk visibility, no segments, T = N.
    qd, kd, vd, dyd = (t.detach().double().requires_grad_() for t in (q, k, v, dy))
    pos = torch.arange(N, device="cuda")
    lo = (pos // C) * C - C
    mask = (pos[None, :] >= lo[:, None]) & (pos[None, :] <= pos[:, None])
    s = torch.einsum("bqhd,bkhd->bhqk", qd, kd) / dim ** 0.5
    p = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
    yd = torch.einsum("bhqk,bkhd->bqhd", p, vd)
    rq, rk, rv = torch.autograd.grad(yd, (qd, kd, vd), dyd.detach())
    out["rel_to_max"] = {n: float((g.double() - r).abs().max() / r.abs().max())
                         for n, g, r in (("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv))}
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
after = tags()
out["tags"] = {":".join(map(str, key)): after[key] - before.get(key, 0) for key in after
               if after[key] != before.get(key, 0)}
out["context_before_node"] = seen.get("context_before_node")
print("RESULT " + json.dumps(out))
"""


def _fresh(dim):
    env = {key: value for key, value in os.environ.items() if key not in KNOBS}
    proc = subprocess.run([sys.executable, "-c", _FRESH_THREAD, str(dim)], env=env,
                          capture_output=True, text=True, timeout=240)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])


@pytest.mark.parametrize("dim", [128, 256])
def test_native_backward_as_the_autograd_threads_first_cuda_work(dim):
    out = _fresh(dim)
    assert out["error"] is None, out
    assert out["finite"], out
    for phase in ("dq", "dk", "dv"):
        assert out["tags"].get(f"{phase}:umma:umma:ran", 0) >= 1, out
    for name, err in out["rel_to_max"].items():
        assert err < 2e-2, (name, out)
    # The caller's dump describes the backward the engine thread ran, and says so by its
    # stamps: a recorded call (seq >= 1) that prepared and launched, published with the
    # launch counters the process holds now. A default record (seq 0, launched False)
    # would be the thread-local reader this replaced.
    rec = out["record"]
    assert rec["seq"] >= 1 and rec["launched"] is True and rec["prepared"] is True, out
    assert rec["launch_counts"] == rec["counts_now"], out
    print(f"dim={dim} record_seq={rec['seq']} on_caller_thread={rec['on_caller_thread']}")
    # Precondition of the regression, reported rather than assumed: on the builds this was
    # written against the node starts with no current context. If PyTorch ever binds one
    # first, the case still runs the native backward but no longer reproduces the defect.
    print(f"dim={dim} context_before_node={out['context_before_node']}")


def _row_divergence(a, b, label="rows that differ (unset, off)"):
    """Name the rows two tag snapshots disagree on, with both values, and never raise.

    An assertion's message is evaluated only on failure, which is exactly the path where a
    crash is worst: a diagnostic that raises turns a divergence report into a TypeError and
    the row that diverged is lost at the moment it is needed. `a ^ b` is the natural way to
    write this and it is only defined for MAPPINGS of keys-to-counts, so it raises on the two
    dicts it is meant to compare, and a caller that passes the bare key set raises one level
    over in `.get`. Total for any pair of dicts, including equal ones, empty ones, and key
    sets that do not overlap.

    Scope: MAPPINGS whose values are counts. `label` is the only thing that names which two
    arms are being compared, so a second call site comparing a different pair must pass its
    own rather than inherit this one's wording.
    """
    keys = set(a) | set(b)
    return label + ": " + repr(
        {key: (a.get(key), b.get(key)) for key in sorted(keys) if a.get(key) != b.get(key)})


def _forward_tags(monkeypatch, env):
    for key in KNOBS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import xattn_cuda
    # Every FORWARD assertion in this file is measured at the (128,128) bucket ONLY: these
    # shapes are exactly it (Hq=4, Hkv=2, D=Dv=128). The (256,256) bucket is inferred from
    # source here and is not measured by this file; do not read a green as covering it.
    torch.manual_seed(0)
    q = F.normalize(torch.randn(1, 256, 4, 128, device="cuda"), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, 256, 2, 128, device="cuda"), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, 256, 2, 128, device="cuda").to(torch.bfloat16)
    rows = lambda: {tuple(r[:4]): int(r[4]) for r in xattn_cuda.ops._flash_sca_native_phase_tags()}
    before = rows()
    pipe_before = int(xattn_cuda.ops._flash_sca_native_fwd_pipe_launches())
    leaf_before = int(xattn_cuda.ops._flash_sca_native_fwd_leaf_launches())
    y, _ = xattn.flash_sca_fwd(q, k, v, 128)
    torch.cuda.synchronize()
    after = rows()
    # Return the DELTAS, not the key set: a retained artifact is rows WITH their counts, so a
    # claim that two arms are indistinguishable in an artifact is a claim about the counts
    # too, and a key set would pass an arm that fired a row twice against one that fired it
    # once. The counters are cumulative in-process, so the delta is the row this call added;
    # raw counts would differ between the arms for that reason alone and could not be compared.
    # A mapping also matters on the failure path, which is the only path a diagnostic runs:
    # `_row_divergence` calls `.get`, and the plain set comprehension this replaces handed it
    # sets, so a divergence would have raised AttributeError exactly where the report was due.
    increased = {key: after[key] - before.get(key, 0) for key in after
                 if after[key] != before.get(key, 0)}
    # The pipelined body and the older leaf record the SAME umma row (dispatch_tag.cuh), so the
    # body that ran is named by the two host launch counts, read by difference around this call:
    # (pipe-launch delta, leaf-launch delta).
    bodies = (int(xattn_cuda.ops._flash_sca_native_fwd_pipe_launches()) - pipe_before,
              int(xattn_cuda.ops._flash_sca_native_fwd_leaf_launches()) - leaf_before)
    # `ran` projects the row down to its body-name field. The third return value is the whole
    # row with its count: a claim about "the same row" is about all four fields, and the
    # projection alone would pass two arms that differ in a field it drops.
    ran = {key[1] for key in increased if key[0] == "forward" and key[3] == "ran"}
    return y, ran, increased, bodies


PIPE, LEAF, NO_NATIVE = (1, 0), (0, 1), (0, 0)  # (pipe-launch delta, leaf-launch delta)


@pytest.mark.parametrize("instrument", [{"XATTN_SCA_ORDER": "sdesc"}, {"XATTN_SCA_LAYOUT": "hf"}])
def test_forward_instrument_keeps_the_native_body_out_of_the_way(monkeypatch, instrument):
    # At level 0 the pipelined body is this bucket's default: the unset arm reaches it.
    y_default, ran, _, bodies = _forward_tags(monkeypatch, {})
    assert ran == {"umma"} and bodies == PIPE, (ran, bodies)
    # A live instrument keeps every native body out of the way: the generic body runs.
    y_instr, ran, _, bodies = _forward_tags(monkeypatch, instrument)
    assert ran == {"mma_sync"} and bodies == NO_NATIVE, (ran, bodies)
    torch.testing.assert_close(y_instr.float(), y_default.float(), atol=2e-2, rtol=2e-2)
    # Also with the pipelined body asked for by name: the instrument wins, and nothing raises.
    _, ran, _, bodies = _forward_tags(monkeypatch, {**instrument, "XATTN_SCA_FWD_PIPE": "1"})
    assert ran == {"mma_sync"} and bodies == NO_NATIVE, (ran, bodies)
    with pytest.raises(RuntimeError, match="cannot be combined"):
        _forward_tags(monkeypatch, {**instrument, "XATTN_SCA_UMMA_FWD": "1"})
    # The opt-in arm is the only way to the older leaf, and it agrees with the default body.
    y_native, ran, _, bodies = _forward_tags(monkeypatch, {"XATTN_SCA_UMMA_FWD": "1"})
    assert ran == {"umma"} and bodies == LEAF, (ran, bodies)
    torch.testing.assert_close(y_native.float(), y_default.float(), atol=2e-2, rtol=2e-2)


def test_unset_and_zero_leave_the_same_forward_phase_row(monkeypatch):
    # dispatch_tag.cuh says the two arms that skip the native leaf record the SAME row, and
    # that this is why a retained artifact cannot tell "whatever main does" (unset) from
    # "forced off" ("0"). That sentence is the only place in the tree where the absence of a
    # distinction is claimed, so it is the one claim in the selection change that a later edit
    # to the predicate could falsify without any other test moving. Pin it here.
    #
    # It is pinned on the whole row, not on its body-name field: the claim is that a retained
    # artifact - which IS the set of tag rows - cannot separate the two arms, so any field
    # diverging breaks it, and `selected` is the field an edit would most naturally use to
    # separate them while both still launch mma_sync.
    # `selected=umma / launched=mma_sync` refusal row is unreachable on both of these arms
    # today, because both skip the leaf's selection entirely and it needs the predicate to
    # select native first. It becomes reachable exactly when the default flips to native, and
    # that is the same edit that turns the `{"mma_sync"}` assertion above red. So the two
    # events coincide, and a red here has TWO correct responses. Amend dispatch_tag.cuh's
    # sentence, which will by then be false; or record the resolved arm IN the row, which
    # also makes that sentence false and is the better fix, since a timing
    # row must be able to name its own body and `unset` names no body today. Relaxing this
    # assertion is neither of them.
    #
    # The default did flip at this bucket: the pipelined body serves level-0 (256,256) and
    # (128,128) calls when XATTN_SCA_FWD_PIPE and XATTN_SCA_UMMA_FWD are both unset, and
    # dispatch_tag.cuh's sentence was amended with it ("One selection precedes that decision";
    # XATTN_SCA_FWD_PIPE=0 => main's forward exactly). So the claim is pinned as amended, in
    # both halves. (1) Under XATTN_SCA_FWD_PIPE=0, which is main's forward selection, unset and
    # "0" still reach the same generic body and leave the SAME row: the assertions below, kept
    # whole. (2) With every knob unset, the unset arm runs the pipelined body, so its row is
    # umma and differs from "0"'s; the pipe-launch delta names that body, which `unset` alone
    # does not (test_the_default_arm_runs_the_pipelined_body_and_names_it below).
    y_unset, ran_unset, rows_unset, _ = _forward_tags(monkeypatch, {"XATTN_SCA_FWD_PIPE": "0"})
    y_off, ran_off, rows_off, _ = _forward_tags(monkeypatch, {"XATTN_SCA_FWD_PIPE": "0",
                                                             "XATTN_SCA_UMMA_FWD": "0"})
    # This assertion is LOAD-BEARING for the one two lines below it, not stylistic. A tag
    # store with no non-zero row serialises to no rows, so `rows_off == rows_unset` is
    # satisfied by two empty mappings; `ran` is projected out of that same mapping, so an
    # empty one cannot reach the equality while this line stands. Do not move it below the
    # row comparison or delete it as redundant - either edit makes the lower assertion
    # vacuously true.
    assert ran_unset == {"mma_sync"}, ran_unset
    assert ran_off == {"mma_sync"}, ran_off
    assert rows_off == rows_unset, _row_divergence(rows_unset, rows_off)
    # Both arms run the same generic body on identically seeded inputs, so this is a
    # comparison that should be exact; a tolerance here would hide one body failing to
    # reproduce its own bits, which would void the artifact indistinguishability above.
    # Labelled so this is not read as part of the row claim: it is a DETERMINISM property of
    # the generic forward (no cross-CTA accumulation). If that body ever gains a split-K or an
    # atomic reduction, this line goes red for a reason unrelated to row indistinguishability.
    assert torch.equal(y_off, y_unset), (y_off.float() - y_unset.float()).abs().max()


def test_the_default_arm_runs_the_pipelined_body_and_names_it(monkeypatch):
    # Half (2) of the amended sentence: at this level-0 (128,128) call the unset arm runs the
    # pipelined body. Its row is umma, so it is NOT "0"'s row, and the launch counts name it.
    _, ran_default, rows_default, bodies_default = _forward_tags(monkeypatch, {})
    _, ran_off, rows_off, bodies_off = _forward_tags(monkeypatch, {"XATTN_SCA_UMMA_FWD": "0"})
    assert ran_default == {"umma"} and bodies_default == PIPE, (ran_default, bodies_default)
    assert ran_off == {"mma_sync"} and bodies_off == NO_NATIVE, (ran_off, bodies_off)
    assert rows_default != rows_off, _row_divergence(rows_default, rows_off,
                                                     label="rows (unset, '0'), expected to differ")


def test_the_divergence_report_is_total_so_a_red_names_the_row():
    # The assert above is only ever asked for its message on the failure path, which is the
    # one path where raising instead of reporting costs the finding. `rows_a ^ rows_b` is how
    # that message was first written and it raises TypeError on two dicts, so a divergence
    # would have surfaced as a type error with the row lost. Pin the report, and pin the
    # reason: the replaced expression must still raise, or this test is guarding nothing.
    row = ("forward", "umma", "mma_sync", "ran")
    other = ("forward", "mma_sync", "mma_sync", "ran")
    with pytest.raises(TypeError):
        {row: 1} ^ {row: 2}
    assert _row_divergence({row: 1}, {row: 1}) == "rows that differ (unset, off): {}"
    assert _row_divergence({}, {}) == "rows that differ (unset, off): {}"
    # The label is the only thing naming which pair is compared, so it must come from the
    # caller: a second call site with a different pair that inherited this wording would
    # misname its own comparison, which is the class of defect the helper exists to prevent.
    assert _row_divergence({row: 1}, {row: 2}, label="dq").startswith("dq: ")
    # Same key, different count: the case a key-set symmetric difference would silently drop.
    counted = _row_divergence({row: 1}, {row: 2})
    assert repr(row) in counted and "1" in counted and "2" in counted, counted
    # Key present on one side only: named with the missing value shown as None.
    sided = _row_divergence({row: 1}, {other: 1})
    assert repr(row) in sided and repr(other) in sided and "None" in sided, sided
