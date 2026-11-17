#!/usr/bin/env python3
"""Two measurements of the backward seam in `csrc/flash_sca/native/host.cc`.

`probe` requires XATTN_SCA_BWD_ARM=generic and records two independent public
forward/backward calls on each fixed fixture, including per-call product-tag
deltas and raw-dtype sha256 hashes, for a two-build comparison. The comparator
uses checked generic source plus those tags to identify the actual dQ writer;
it never trusts the prep snapshot left stale by the base source swap.
It is **not** a correctness check:
there is no FP64 oracle here (those rows live in `sca-contract` and
`gekko-shapes`). Byte identity is meaningful for stable writers on matched
fallback paths, not a requirement for det-off atomic dQ. A default call at the
current head can select native products, so matching fixtures alone do not prove
that this is a fallback comparison; serving-body evidence is required too.

The discriminating fixture is **live-prev**: a previous K/V plane the current
chunk can actually see (one segment, no BOS at the call boundary), so
`dPrevK`/`dPrevV` are non-zero and a wrongly-initialised previous region would
show up as zeros, not as a small numerical difference.

`dead-prev` is its control: a BOS at the first current token closes the previous
plane as another segment, so `dPrev` must be exactly zero. **The first run of this
probe measured non-zero there** (16384 of 16384 elements) — because the fixture
built a `bos_mask` and passed it to *neither* call: without the mask the previous
plane stays visible, so the row was a live-prev fixture under another name and
non-zero `dPrev` was the right answer to the question actually asked. The mask
reaches both calls now, and every masked fixture records `mask_effect` — whether
its forward WITH the mask differs from its forward without it — because a mask
that is built and dropped looks exactly like a mask that hides nothing; a `same`
on a dead-prev fixture fails the run. `dead-prev-bos0` differs from `dead-prev` in
one thing on the same seed, `bos[0]` also true, which per the contract must shift
segment ids without changing visibility: the pair makes that measurable.

`order` checks the source order of the seam's steps: preparation, planning,
preparation-refusal branch and generic launch, then the phase driver's final
preflight before optional dPrev zeroing and its launch. This is a source-order
lint, not a control-flow or serving-body proof. Offsets are measured on
comment-stripped text, because the same identifiers appear in the prose above
the code.

Usage:

    cd kernel
    python tests_b300/sca/seam_inertness_probe.py order            # or: order --selftest
    python tests_b300/sca/seam_inertness_probe.py swap --to base --dry-run  # no write/build
    XATTN_SCA_BWD_ARM=generic python tests_b300/sca/seam_inertness_probe.py probe --label base \\
        --out "${OUT:-/tmp}/seam_inertness_base.json"

`probe` builds the extension in place if the import is unusable, with
`XATTN_TARGET_SM` unset (the way `sca-contract` builds), and prints which it did.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HOST_CC = os.path.join(ROOT, "csrc", "flash_sca", "native", "host.cc")

GUARD = "if (p.batch == 0) return {dq, dk, dv, dprev_k, dprev_v};"
END = "return {dq, dk, dv, dprev_k, dprev_v};"

# Source order in a53d5b0350: preparation may decline, but a failed plan raises;
# only !use_phase_driver enters the complete generic fallback. The final check
# belongs to the phase-driver path (the fallback returns before reaching it).
ANCHORS = [
    ("prepare", "PrepareBackwardUmma("),
    ("plan", "PlanBackwardPhases(p, umma.launch"),
    ("refusal-branch", "if (!use_phase_driver) {"),
    ("refusal-launch", "LaunchBackward(p, stream);"),
    ("final-check", "CheckBackwardPhases(p, umma.launch, phases"),
    ("dprev-zero", "dprev_k->zero_();"),
    ("native-launch", "LaunchBackwardPhases(p, umma.launch"),
]


# ---------------------------------------------------------------- order check


def seam_span(text):
    """The Backward body from its batch-0 guard to its final return, as one string.

    The end is the LAST return of that shape in the file, not the first: the seam
    has an early return inside its refusal branch, and the span a reader means is
    the whole body, both returns included.
    """
    start = text.rfind(GUARD)
    if start < 0:
        return None
    end = text.rfind(END)
    if end < start:
        return None
    return text[start:end + len(END)]


def strip_comments(span):
    """Blank out comments, so an identifier quoted in prose is not an anchor.

    `/\\* ... \\*/` becomes spaces (the seam writes `/*deterministic=*/`), and `//`
    to end of line becomes spaces. Length is preserved, so offsets stay the ones a
    reader sees in the file only when there are no comments before them — which is
    why the offsets are reported on the stripped text and never compared to line
    numbers.
    """
    out = list(span)
    i, n = 0, len(span)
    while i < n:
        if span.startswith("//", i):
            while i < n and span[i] != "\n":
                out[i] = " "
                i += 1
        elif span.startswith("/*", i):
            while i < n and not span.startswith("*/", i):
                out[i] = " "
                i += 1
            for _ in range(2):
                if i < n:
                    out[i] = " "
                    i += 1
        elif span[i] == '"':
            i += 1
            while i < n and span[i] != '"':
                i += 2 if span[i] == "\\" else 1
            i += 1
        else:
            i += 1
    return "".join(out)


def order_findings(span, anchors=ANCHORS):
    """Every way the span violates the order, and the offsets when it does not."""
    stripped = strip_comments(span)
    offsets, problems = {}, []
    for name, needle in anchors:
        count = stripped.count(needle)
        if count != 1:
            problems.append("%s: %d occurrences of %r, expected 1" % (name, count, needle))
            continue
        offsets[name] = stripped.index(needle)
    if not problems:
        # In DECLARATION order, which is the order the seam must keep: comparing
        # offsets after sorting them by offset would compare each anchor with itself.
        names = [name for name, _ in anchors if name in offsets]
        for a, b in zip(names, names[1:]):
            if offsets[a] > offsets[b]:
                problems.append("%s @%d runs after %s @%d" % (a, offsets[a], b, offsets[b]))
    return offsets, problems


def selftest_cases(span):
    """Variants of the span a correct check must reject (and one it must accept)."""
    def cut(text, needle):
        i = text.index(needle)
        return text[:i] + text[i + len(needle):]

    def move_end(text, needle):
        i = text.index(needle)
        return cut(text, needle) + "\n  " + needle

    cases = [("verbatim", span, True)]
    # A comment that names a later step above an earlier one must not move anything.
    cases.append(("comment-quotes-later-step",
                  span.replace(ANCHORS[0][1], "// see PlanBackwardPhases(p, umma.launch\n  "
                                + ANCHORS[0][1], 1), True))
    cases.append(("native-launch-missing", cut(span, ANCHORS[6][1]), False))
    cases.append(("plan-duplicated", span + "\n  " + ANCHORS[1][1], False))
    cases.append(("refusal-branch-last", move_end(span, ANCHORS[2][1]), False))
    cases.append(("dprev-zero-before-final-check", move_end(span, ANCHORS[5][1])
                  .replace(ANCHORS[4][1], ANCHORS[5][1] + "\n  " + ANCHORS[4][1], 1), False))
    cases.append(("native-launch-first", span.replace(ANCHORS[0][1], ANCHORS[6][1] + "\n  "
                                                      + ANCHORS[0][1], 1), False))
    return cases


def run_order(selftest):
    text = open(HOST_CC, encoding="utf-8").read()
    span = seam_span(text)
    if span is None:
        print("FAIL: no Backward body found in %s" % HOST_CC)
        return 1
    failed = 0
    if selftest:
        for name, variant, should_pass in selftest_cases(span):
            _, problems = order_findings(variant)
            ok = not problems
            verdict = "PASS" if ok == should_pass else "FAIL"
            print("selftest %-30s %s (problems: %s)" % (
                name, verdict, "; ".join(problems) if problems else "none"))
            failed += 0 if ok == should_pass else 1
        print("selftest: %d of %d as expected" % (len(selftest_cases(span)) - failed,
                                                  len(selftest_cases(span))))
    offsets, problems = order_findings(span)
    if problems:
        print("FAIL: seam order")
        for problem in problems:
            print("  " + problem)
        return 1
    print("PASS: seam order  " + "  ".join(
        "%s @%d" % (name, offsets[name]) for name, _ in ANCHORS))
    return 1 if failed else 0


# ---------------------------------------------------------------- probe


def build_in_place():
    env = dict(os.environ)
    env.pop("XATTN_TARGET_SM", None)
    env.setdefault("MAX_JOBS", "8")
    print("building: python setup.py build_ext --inplace (XATTN_TARGET_SM unset)")
    rc = subprocess.call([sys.executable, "setup.py", "build_ext", "--inplace"],
                         cwd=ROOT, env=env)
    print("build_rc=%d" % rc)
    return rc


def usable(sca, torch):
    """One tiny public call: the only evidence that this build runs on this GPU.

    `import xattn` is not evidence — a tree can carry an .so built for another
    target, and then every call refuses with «this xattn build does not link them».
    """
    g = torch.Generator(device="cpu").manual_seed(7)
    B, N, C, H, D = 1, 8, 8, 1, 32
    t = [torch.randn(B, N, H, D, generator=g, dtype=torch.float32) for _ in range(3)]
    to = lambda x: x.to("cuda", torch.bfloat16)  # noqa: E731
    try:
        sca.flash_sca_fwd(to(t[0]), to(t[1]), to(t[2]), C, backend="auto")
        torch.cuda.synchronize()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0])


def load_xattn():
    sys.path.insert(0, ROOT)
    import torch
    import xattn.ops.sliding_chunk_attention as sca
    ok, err = usable(sca, torch)
    if ok:
        return sca, torch, "imported"
    print("extension present but unusable on this device: %s" % err)
    if build_in_place() != 0:
        return None, None, "build_failed"
    for mod in [m for m in list(sys.modules) if m.startswith("xattn")]:
        del sys.modules[mod]
    import xattn.ops.sliding_chunk_attention as sca
    import torch
    ok, err = usable(sca, torch)
    if not ok:
        print("still unusable after the build: %s" % err)
        return None, None, "unusable_after_build"
    return sca, torch, "built"


def sha(t):
    """sha256 of the raw bytes: a comparison of bytes, never of a converted value."""
    if t is None:
        return None
    import torch
    b = t.detach().to("cpu").contiguous()
    raw = b.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def stats(t):
    if t is None:
        return None
    return {"sha": sha(t), "shape": list(t.shape), "dtype": str(t.dtype),
            "max_abs": float(t.detach().abs().max().item()) if t.numel() else 0.0,
            "nonzero": int((t.detach() != 0).sum().item()), "numel": int(t.numel())}


def make_case(torch, name, with_prev, dead_prev, level, deterministic):
    B, N, C = 1, 128, 64
    Hq, Hkv, D, Dv = 4, 2, 128, 128
    g = torch.Generator(device="cpu").manual_seed(
        {"no-prev": 11, "live-prev": 12, "dead-prev": 13, "dead-prev-bos0": 13}[name])
    q = torch.randn(B, N, Hq, D, generator=g, dtype=torch.float32)
    k = torch.randn(B, N, Hkv, D, generator=g, dtype=torch.float32)
    v = torch.randn(B, N, Hkv, Dv, generator=g, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True)          # unit-norm rows, as gekko does
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.nn.functional.silu(v) + 0.1
    dy = torch.randn(B, N, Hq, Dv, generator=g, dtype=torch.float32)
    dy = dy / dy.norm(dim=-1, keepdim=True)
    prev_k = prev_v = bos = None
    if with_prev:
        prev_k = torch.randn(B, C, Hkv, D, generator=g, dtype=torch.float32)
        prev_v = torch.randn(B, C, Hkv, Dv, generator=g, dtype=torch.float32)
        prev_k = prev_k / prev_k.norm(dim=-1, keepdim=True)
        prev_v = torch.nn.functional.silu(prev_v) + 0.1
        bos = torch.zeros(B, C + N, dtype=torch.bool)
        if dead_prev:
            # The dead-prev shape: a BOS at the first current token closes the
            # previous plane as another segment. `dead_prev == 2` adds a BOS at
            # position 0 as well — the one difference between the two, on the same
            # seed — so the pair says whether the leading unmarked run matters.
            bos[:, C] = True
            if dead_prev == 2:
                bos[:, 0] = True
    to = lambda t, dt=torch.bfloat16: None if t is None else t.to("cuda", dt)  # noqa: E731
    return dict(name=name, q=to(q), k=to(k), v=to(v), dy=to(dy), prev_k=to(prev_k),
                prev_v=to(prev_v), bos_mask=None if bos is None else bos.to("cuda"),
                chunk_size=C, level=level, deterministic=deterministic)


def phase_counts():
    # The bindings live on the ops submodule. Absence is a probe failure, not a
    # missing-data pass, and dispatch-entry counters cannot replace product tags.
    import xattn_cuda
    rows = xattn_cuda.ops._flash_sca_native_phase_tags()
    return {tuple(row[:4]): int(row[4]) for row in rows}


def run_case(torch, sca, case):
    # Hash the actual operands, including the packing, before EACH call. Seeds
    # alone do not establish a matched pair across two extension builds.
    inputs = {name: None if case[name] is None else {
        "sha": sha(case[name]), "shape": list(case[name].shape), "dtype": str(case[name].dtype)}
        for name in ("q", "k", "v", "dy", "prev_k", "prev_v", "bos_mask")}
    tags_before = phase_counts()
    kw = dict(backend="auto", reset_chunk_pos_per_seq=False,
              high_precision_level=case["level"])
    mask = case["bos_mask"]
    y, y_fp32, lse = sca.flash_sca_fwd(case["q"], case["k"], case["v"], case["chunk_size"],
                                       scale=None, prev_k=case["prev_k"],
                                       prev_v=case["prev_v"], bos_mask=mask, **kw)
    y_saved = y if case["level"] == 0 else y_fp32
    dq, dk, dv, dprev_k, dprev_v = sca.flash_sca_bwd(
        case["dy"], case["q"], case["k"], case["v"], y_saved, lse, case["chunk_size"],
        scale=None, prev_k=case["prev_k"], prev_v=case["prev_v"], bos_mask=mask,
        deterministic=case["deterministic"], **kw)
    torch.cuda.synchronize()
    tags_after = phase_counts()
    tag_deltas = [list(key) + [tags_after.get(key, 0) - tags_before.get(key, 0)]
                  for key in sorted(set(tags_after) | set(tags_before))
                  if tags_after.get(key, 0) != tags_before.get(key, 0)]
    # Anti-vacuity for the mask itself: a mask that is built and then dropped looks
    # exactly like a mask that hides nothing, and the previous plane stays visible,
    # so dPrev comes back non-zero and reads as a kernel defect. On a fixture whose
    # previous plane is meant to be another segment, the forward WITH the mask must
    # differ from the forward without it; `same` there means the mask never arrived.
    mask_effect = None
    if mask is not None:
        y_plain, _, _ = sca.flash_sca_fwd(case["q"], case["k"], case["v"], case["chunk_size"],
                                          scale=None, prev_k=case["prev_k"],
                                          prev_v=case["prev_v"], bos_mask=None, **kw)
        torch.cuda.synchronize()
        mask_effect = "same" if sha(y_plain) == sha(y) else "differs"
    return {"y": stats(y), "y_fp32": stats(y_fp32), "lse": stats(lse), "dq": stats(dq),
            "dk": stats(dk), "dv": stats(dv), "dprev_k": stats(dprev_k),
            "dprev_v": stats(dprev_v), "y_fp32_is_none": y_fp32 is None,
            "mask_effect": mask_effect, "deterministic": case["deterministic"],
            "inputs": inputs, "options": dict(kw, chunk_size=case["chunk_size"], scale=None),
            "phase_tags": tag_deltas,
            "body_arm": os.environ.get("XATTN_SCA_BWD_ARM", "<unset>")}


def tree_fingerprint():
    out = {}
    try:
        out["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
        out["git_status"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip().splitlines()
    except Exception:  # noqa: BLE001
        out["git_head"] = None
    for rel in ("csrc/flash_sca/native/host.cc", "csrc/flash_sca/native/umma_prepare.h",
                "csrc/flash_sca/native/umma_bwd.h", "csrc/flash_sca/native/umma_bwd.cu"):
        p = os.path.join(ROOT, rel)
        out[rel] = (hashlib.sha256(open(p, "rb").read()).hexdigest()
                    if os.path.exists(p) else "absent")
    return out


# The traced xllm geometries, as the caller actually calls them: BF16,
# level 0, contiguous, `bos_mask` from the packing and `segment_idx=None`. The
# small fixtures above are the contract's shapes; these three are the shapes the
# seam will actually be asked to serve, so a byte-identity claim that only holds
# at N=128 would be a claim about the harness.
TRACED = {
    # name: (B, N, C, Hq, Hkv, D, Dv, with_prev, seed)
    "traced-0.9b-train": (4, 65536, 4096, 8, 4, 256, 256, False, 901),
    "traced-4b-train": (1, 65536, 4096, 32, 8, 128, 128, False, 401),
    "traced-0.9b-prefill": (1, 16384, 4096, 8, 4, 256, 256, True, 902),
}


def bos_packing(torch, B, total, seed):
    """A packing in the traced regime: contiguous documents of 1024-2047 rows, so a
    65536-row call carries ~42-49 segments, as the traced packings do. True at the
    first row of every segment and at row 0. For a prefill call the caller's own
    first row is a BOS too (xllm's `get_bos_mask_from_eos` marks the first token of
    every call), which is what makes the previous plane another segment."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    bos = torch.zeros(B, total, dtype=torch.bool)
    for b in range(B):
        row = 0
        while row < total:
            bos[b, row] = True
            row += int(torch.randint(1024, 2048, (1,), generator=g).item())
    return bos


def make_traced_case(torch, name, level, deterministic):
    """The traced geometries, built on the device: at 65536 rows a CPU build means
    several GB of temporaries per plane, and the answers here are about bytes, not
    about how the inputs were made. Conditioning follows the oracle's (unit-norm Q/K
    rows, V = silu(normal) + 0.1, unit-norm dY per token)."""
    B, N, C, Hq, Hkv, D, Dv, with_prev, seed = TRACED[name]
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, N, Hq, D, generator=g, device="cuda", dtype=torch.float32)
    k = torch.randn(B, N, Hkv, D, generator=g, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N, Hkv, Dv, generator=g, device="cuda", dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.nn.functional.silu(v) + 0.1
    dy = torch.randn(B, N, Hq, Dv, generator=g, device="cuda", dtype=torch.float32)
    dy = dy / dy.norm(dim=-1, keepdim=True)
    prev_k = prev_v = None
    if with_prev:
        prev_k = torch.randn(B, C, Hkv, D, generator=g, device="cuda", dtype=torch.float32)
        prev_v = torch.randn(B, C, Hkv, Dv, generator=g, device="cuda", dtype=torch.float32)
        prev_k = prev_k / prev_k.norm(dim=-1, keepdim=True)
        prev_v = torch.nn.functional.silu(prev_v) + 0.1
    total = (C + N) if with_prev else N
    bos = bos_packing(torch, B, total, seed)
    if with_prev:
        bos[:, C] = True  # the call's first current row: the previous plane is another segment
    to = lambda t: None if t is None else t.to(torch.bfloat16)  # noqa: E731
    return dict(name=name, q=to(q), k=to(k), v=to(v), dy=to(dy), prev_k=to(prev_k),
                prev_v=to(prev_v), bos_mask=bos.to("cuda"), chunk_size=C, level=level,
                deterministic=deterministic)


# (name, with_prev, dead_prev, level, deterministic). The traced rows are the
# run's own shapes; det=1 is kept on one of them because the deterministic request
# is a separate group in the report and must be byte-identical on both sides too.
CASES = [("no-prev", False, False, 0, False),
         ("live-prev", True, False, 0, False),
         ("live-prev", True, False, 0, True),
         ("dead-prev", True, 1, 0, False),
         ("dead-prev-bos0", True, 2, 0, False),
         ("no-prev", False, False, 1, False),
         ("live-prev", True, False, 1, False),
         ("traced-0.9b-train", False, False, 0, False),
         ("traced-0.9b-train", False, False, 0, True),
         ("traced-4b-train", False, False, 0, False),
         ("traced-0.9b-prefill", True, False, 0, False)]


def run_probe(label, out_path):
    # The arm is a request, not serving-body evidence; every call still records
    # its product-tag deltas. Source-checked inference avoids the diagnostic
    # snapshot which the body-only base swap deliberately does not maintain.
    from pair_compare import generic_source_evidence, source_errors
    if os.environ.get("XATTN_SCA_BWD_ARM") != "generic":
        print("FAIL: probe requires explicit XATTN_SCA_BWD_ARM=generic")
        return 1
    try:
        source = generic_source_evidence(ROOT)
        problems = source_errors(source)
    except (OSError, ValueError) as exc:
        print("FAIL: cannot check generic dQ source: %s" % exc)
        return 1
    if problems:
        print("FAIL: " + "; ".join(problems))
        print(json.dumps(source, sort_keys=True))
        return 1
    sca, torch, how = load_xattn()
    if sca is None:
        print("FAIL: %s; nothing measured" % how)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible")
        return 1
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("import=%s device=%s capability=%s torch=%s" % (how, name, cap, torch.__version__))

    import xattn_cuda
    with open(xattn_cuda.__file__, "rb") as fh:
        extension_sha256 = hashlib.sha256(fh.read()).hexdigest()
    result = {"label": label, "how": how, "device": name, "capability": list(cap),
              "torch": torch.__version__, "tree": tree_fingerprint(), "cases": {},
              "extension_sha256": extension_sha256,
              "body_arm": os.environ.get("XATTN_SCA_BWD_ARM", "<unset>"),
              "byte_hash": "sha256_raw_dtype_bytes", "calls_per_case": 2,
              "generic_dq_source": source}
    failures = 0
    for (cname, with_prev, dead_prev, level, det) in CASES:
        key = "%s/L%d/det=%d" % (cname, level, int(det))
        try:
            case = (make_traced_case(torch, cname, level, det) if cname.startswith("traced-")
                    else make_case(torch, cname, with_prev, dead_prev, level, det))
            row = run_case(torch, sca, case)
            # Re-execute, do not duplicate hashes: deterministic planes must
            # repeat byte-exactly and atomic dQ variability must be visible.
            row["repeats"] = [run_case(torch, sca, case)]
            result["cases"][key] = row
            for call in [row] + row["repeats"]:
                dead = cname.startswith("dead-prev") or cname == "traced-0.9b-prefill"
                if dead and call["mask_effect"] != "differs":
                    failures += 1
                    call["mask_effect_failure"] = (
                        "a mask that hides the previous plane left y unchanged: it never "
                        "reached the call")
        except Exception:  # noqa: BLE001
            failures += 1
            result["cases"][key] = {"error": traceback.format_exc().strip().splitlines()[-1]}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    print("wrote %s" % out_path)
    print("%-18s %-34s %-12s %-11s %-7s %s" % (
        "case", "y", "dq", "nz(dprev_k)", "nz(dq)", "mask"))
    for key, row in result["cases"].items():
        if "error" in row:
            print("%-18s ERROR %s" % (key, row["error"]))
            continue
        print("%-18s %-34s %-12s %-11d %-7d %s" % (
            key, row["y"]["sha"], row["dq"]["sha"],
            row["dprev_k"]["nonzero"] if row["dprev_k"] else -1, row["dq"]["nonzero"],
            row.get("mask_effect") or "-"))
        if "mask_effect_failure" in row:
            print("  FAIL: %s" % row["mask_effect_failure"])
    print("cases: %d, failed: %d" % (len(CASES), failures))
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------- base side


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


# Replace ONLY the post-allocation Backward span. The current host's parser and
# diagnostic bindings outside that span still need BackwardArm/BackwardPhases and
# umma::bwd_host. The old two-hunk inversion removed those headers/alias and no
# longer produces a compilable base. Everything outside the span is kept verbatim;
# this is a generic comparison body, not a reconstruction of an historical commit.
BASE_REQUIRED_HEADERS = (
    '#include "flash_sca/native/umma_bwd.h"',
    '#include "flash_sca/native/umma_prepare.h"',
)
BASE_ANCHORS = [
    ("request", "const BackwardRequest request = BackwardArmOptIn();"),
    ("native-refusal", "TORCH_CHECK(!request.require_native,"),
    ("split", "const bool split = request.arm == BackwardArm::kSplit;"),
    ("stream", "const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());"),
    ("key-range-storage", "p.key_tile_range = key_tile_range.data_ptr<int32_t>();"),
    ("accumulator-condition", "if (!deterministic && !split) {"),
    ("accumulator-zero", "dq_accum = at::zeros(dq.sizes(), q.options().dtype(at::kFloat));"),
    ("accumulator-bind", "p.dq_accum = dq_accum.data_ptr<float>();"),
    ("tile-bounds", "LaunchTileBounds(p, stream);"),
    ("key-range", "if (p.row_lo != nullptr) LaunchKeyTileRange(p, stream);"),
    ("all-generic-stages", "LaunchBackward(p, stream);"),
    ("fold", "if (!deterministic && !split) dq.copy_(dq_accum.mul_(p.scale));"),
    ("count", "g_backward_launches.fetch_add(1, std::memory_order_relaxed);"),
]
BASE_FORBIDDEN_CALLS = ("PrepareBackwardUmma(", "PlanBackwardPhases(",
                        "CheckBackwardPhases(", "LaunchBackwardPhases(")
SEED_BLOCK = os.path.join(HERE, "seed_block.cc")


def base_variant(text, seed):
    """Validate both source spans and return a body-only swap, without writing."""
    for needle in BASE_REQUIRED_HEADERS + ("BackwardRequest BackwardArmOptIn()",):
        if strip_comments(text).count(needle) != 1:
            raise ValueError("expected exactly one current host declaration: %r" % needle)
    span = seam_span(text)
    if span is None:
        raise ValueError("no Backward body")
    _, problems = order_findings(span)
    if problems:
        raise ValueError("seam order: " + "; ".join(problems))
    # seam_span starts at 'if', not the line's indentation; retain the original
    # prefix rather than indenting the guard a second time on every swap.
    seed = seed.strip()
    if seam_span(seed) != seed:
        raise ValueError("seed must contain exactly the batch guard through final return")
    _, problems = order_findings(seed, BASE_ANCHORS)
    stripped_seed = strip_comments(seed)
    problems += ["generic seed contains %s" % needle for needle in BASE_FORBIDDEN_CALLS
                 if needle in stripped_seed]
    if problems:
        raise ValueError("generic seed: " + "; ".join(problems))
    start = text.rfind(GUARD)
    return text[:start] + seed + text[start + len(span):]


def swap_to_base(path=HOST_CC, dry_run=False):
    """Keep current headers/helpers; replace only Backward's schedule with the seed."""
    text = open(path, encoding="utf-8").read()
    seed = open(SEED_BLOCK, encoding="utf-8").read()
    try:
        out = base_variant(text, seed)
    except ValueError as exc:
        print("FAIL: %s: %s" % (path, exc))
        return 1
    if not dry_run:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
    print("%s: %s -> %s (%d -> %d bytes); headers/helpers retained" % (
        "validated base swap (no write)" if dry_run else "swapped Backward body to base",
        sha_text(text), sha_text(out), len(text), len(out)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    o = sub.add_parser("order", help="assert the seam's step order in host.cc")
    o.add_argument("--selftest", action="store_true")
    p = sub.add_parser("probe", help="record the public backward's bytes for one build")
    p.add_argument("--label", required=True)
    p.add_argument("--out", default=None)
    w = sub.add_parser("swap", help="replace only host.cc's Backward schedule with the generic base")
    w.add_argument("--to", default="base", choices=["base"])
    w.add_argument("--dry-run", action="store_true",
                   help="validate the body-only swap and print hashes without writing or building")
    args = ap.parse_args()
    if args.what == "swap":
        return swap_to_base(dry_run=args.dry_run)
    if args.what == "order":
        return run_order(args.selftest)
    out = args.out or os.path.join(os.environ.get("OUT", "/tmp"),
                                   "seam_inertness_%s.json" % args.label)
    return run_probe(args.label, out)


if __name__ == "__main__":
    sys.exit(main())
