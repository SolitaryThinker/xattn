#!/usr/bin/env python3
"""Fail-closed, forced-generic BACKWARD seam comparison (not an FP64 oracle).

Every stable-writer plane, including every deterministic plane, must have the
same raw-dtype bytes across builds AND within each build's repeated calls.
Det-off dQ is report-only ONLY when per-call product tags and checked generic
source identify its unordered FP32 atomic writer. Two matching repeats do not
turn that writer into a deterministic one. Its nonzero count is reported too:
rounding/cancellation can change that count, so it is not an identity guarantee.

Explicit contract None is checked, never counted as an identical tensor; an
absent key is an error. Live/dead dPrevK AND dPrevV and the dead-prev mask control
must be non-vacuous on every call. Source/tag evidence identifies a host launch
path, not executed SASS. No prep diagnostic is trusted: the base source swap
retains a binding whose snapshot is stale for the substituted schedule.

PASS means structural/stable-writer byte checks only, not all-plane equivalence,
numerical correctness, native qualification, or a test of preparation refusal.
Use the unchanged sca-contract/gekko-shapes FP64 checks for numerical correctness.

    python tests_b300/sca/pair_compare.py  # $OUT/seam_inertness_{base,stageb}.json
    python tests_b300/sca/pair_compare.py --selftest  # CPU only, no torch import
"""
import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys

TENSORS = ("y", "y_fp32", "lse", "dq", "dk", "dv", "dprev_k", "dprev_v")
HASH_KIND = "sha256_raw_dtype_bytes"
SOURCE_RULE = "generic-dq-tags-source-v1"
# Exact pinned dQ launch branch. A skipped row is evidence of an accumulator,
# not an absent gradient. Whitespace/comments may change; this branch may not.
DQ_STAGE = """
void BackwardDqStage(const Params& p, cudaStream_t stream) {
  if (p.dq_accum == nullptr) {
    using Config = detail::DqConfig<kD, kDv>;
    auto kernel = detail::DqKernel<E, kD, kDv>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Config::kSmemBytes));
    const dim3 grid(static_cast<unsigned>((p.q_length + Config::kBM - 1) / Config::kBM),
                    static_cast<unsigned>(p.query_heads), static_cast<unsigned>(p.batch));
    kernel<<<grid, Config::kThreads, Config::kSmemBytes, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    RecordPhase(Phase::kDq, Body::kMmaSync, Body::kMmaSync, State::kRan);
  } else {
    RecordPhase(Phase::kDq, Body::kNone, Body::kNone, State::kSkipped);
  }
}
"""
# This is a narrow source lint for the mode inference, not a C++ control-flow
# proof or a claim that the on-disk source is the loaded binary. Pin the files
# and require the same generic implementation on both builds. The pair driver
# builds each side explicitly; device/SASS proof belongs to qualification.
SOURCE_CHECKS = {
    "launch.cuh": {
        "dq_tag_branch": (DQ_STAGE, 1),
        "wide_key_body": ("auto kernel = detail::DkdvKernel<E, kD, kDv>;", 1),
        "reg_key_body": ("auto kernel = detail::DkdvRegKernel<E, kD, kDv>;", 1),
        "key_tags": ("RecordPhase(Phase::kDk, Body::kMmaSync, Body::kMmaSync, State::kRan);"
                     "RecordPhase(Phase::kDv, Body::kMmaSync, Body::kMmaSync, State::kRan);", 2),
    },
    "kernels.cuh": {
        "atomic_pair": (r'''void ReduceAdd2(float* address, float x, float y) {
          asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n"
                       ::"l"(address), "f"(x), "f"(y) : "memory");
        }''', 1),
        "accumulator_output": ("float* out = p.dq_accum + "
                               "((b * p.q_length + row) * p.query_heads + hq) * p.qk_dim;", 1),
        "accumulator_add": ("if (col < p.qk_dim) "
                            "ReduceAdd2(out + col, acc[i][2 * h], acc[i][2 * h + 1]);", 1),
        "wide_accumulate": ("""if (p.dq_accum != nullptr) {
          const int64_t hq = hk * p.group + step / span;
          #pragma unroll 1
          for (int c = 0; c < kD / 2; c += 64) {
            AccumulateDq<E, kD, kBN, (kD / 2 < 64 ? kD / 2 : 64)>(p, sS, sK, kg * 16,
              half * (kD / 2) + c, b, hq, m0);
          }
        }""", 1),
        "reg_accumulator": ("const bool fused_dq = p.dq_accum != nullptr;", 1),
        "reg_accumulate": ("""if (fused_dq) {
          __syncthreads();
          const int64_t hq = hk * p.group + step / span;
          AccumulateDq<E, kD, kBN, kD / 2>(p, sS, sK, (warp & 3) * 16,
            (warp >> 2) * (kD / 2), b, hq, m0);
        }""", 1),
    },
}


def _code(text):
    return "".join(re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S).split())


def generic_source_evidence(root):
    """CPU-only source check, called by the probe BEFORE collecting either arm."""
    evidence = {"rule": SOURCE_RULE, "files": {}, "checks": {}}
    for name, checks in SOURCE_CHECKS.items():
        path = os.path.join(root, "csrc", "flash_sca", "native", name)
        with open(path, "rb") as fh:
            raw = fh.read()
        evidence["files"][name] = hashlib.sha256(raw).hexdigest()
        code = _code(raw.decode("utf-8"))
        for check, (snippet, count) in checks.items():
            evidence["checks"][name + ":" + check] = code.count(_code(snippet)) == count
    return evidence


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def source_errors(evidence):
    if not isinstance(evidence, dict) or evidence.get("rule") != SOURCE_RULE:
        return ["missing/unknown generic dQ source rule"]
    files, checks = evidence.get("files"), evidence.get("checks")
    expected = {name + ":" + check for name, rows in SOURCE_CHECKS.items() for check in rows}
    bad = []
    if (not isinstance(files, dict) or set(files) != set(SOURCE_CHECKS)
            or not all(_digest(x) for x in files.values())):
        bad.append("missing/invalid generic source sha256 pins")
    if (not isinstance(checks, dict) or set(checks) != expected
            or not all(x is True for x in checks.values())):
        bad.append("generic dQ source checks missing or failed")
    return bad


def case_specs():
    # The probe module is CPU-importable. This manifest, not the JSON's own
    # row list, prevents deletion on BOTH sides from becoming a passing pair.
    from seam_inertness_probe import CASES, TRACED
    specs = {}
    for name, prev, _dead, level, det in CASES:
        if name in TRACED:
            b, n, c, hq, hkv, d, dv, prev, _seed = TRACED[name]
        else:
            b, n, c, hq, hkv, d, dv = 1, 128, 64, 4, 2, 128, 128
        shapes = {"y": [b, n, hq, dv], "y_fp32": [b, n, hq, dv] if level else None,
                  "lse": [b, hq, n], "dq": [b, n, hq, d], "dk": [b, n, hkv, d],
                  "dv": [b, n, hkv, dv], "dprev_k": [b, c, hkv, d] if prev else None,
                  "dprev_v": [b, c, hkv, dv] if prev else None}
        specs["%s/L%d/det=%d" % (name, level, int(det))] = (shapes, det, level, c)
    return specs


def input_shapes(spec, key):
    shapes, _det, _level, c = spec
    b, n = shapes["y"][:2]
    prev = shapes["dprev_k"] is not None
    inputs = {name: shapes[plane] for name, plane in
              (("q", "dq"), ("k", "dk"), ("v", "dv"), ("dy", "y"),
               ("prev_k", "dprev_k"), ("prev_v", "dprev_v"))}
    inputs["bos_mask"] = [b, n + (c if prev else 0)] if prev or key.startswith("traced-") else None
    return inputs


def call_options(level, chunk):
    return {"backend": "auto", "reset_chunk_pos_per_seq": False,
            "high_precision_level": level, "chunk_size": chunk, "scale": None}


def _tags(row):
    tags = row.get("phase_tags")
    if not isinstance(tags, list) or len(tags) != 5:
        raise ValueError("missing/invalid per-call phase_tags (need all five phases)")
    parsed = {}
    for tag in tags:
        if (not isinstance(tag, list) or len(tag) != 5
                or not all(isinstance(x, str) for x in tag[:4])
                or type(tag[4]) is not int or tag[4] != 1 or tag[0] in parsed):
            raise ValueError("malformed/duplicate/non-unit per-call phase tag")
        parsed[tag[0]] = tuple(tag[1:4])
    if set(parsed) != {"forward", "delta", "dq", "dk", "dv"}:
        raise ValueError("missing/unknown product phase")
    for phase, expected in (("delta", ("scalar", "scalar", "ran")),
                            ("dk", ("mma_sync", "mma_sync", "ran")),
                            ("dv", ("mma_sync", "mma_sync", "ran"))):
        if parsed[phase] != expected:
            raise ValueError("%s is not the forced-generic producer: %s" % (phase, parsed[phase]))
    if parsed["forward"] not in (("mma_sync", "mma_sync", "ran"), ("umma", "umma", "ran")):
        raise ValueError("unknown/missing forward producer")
    if parsed["dq"] == ("none", "none", "skipped"):
        mode = "atomic"
    elif parsed["dq"] == ("mma_sync", "mma_sync", "ran"):
        mode = "direct"
    else:
        raise ValueError("unknown/non-generic dQ producer")
    return parsed, mode


def _check_call(row, spec, key):
    shapes, det, level, chunk = spec
    if not isinstance(row, dict):
        return ["missing call row"], None
    bad = []
    if "error" in row:
        bad.append("call error: %s" % row["error"])
    if row.get("body_arm") != "generic":
        bad.append("body_arm must be explicitly generic")
    if row.get("deterministic") is not det:
        bad.append("deterministic must match the manifest boolean")
    if row.get("y_fp32_is_none") is not (level == 0):
        bad.append("y_fp32_is_none must match the level")
    if row.get("options") != call_options(level, chunk):
        bad.append("call options differ from the fixed fixture")
    inputs = row.get("inputs")
    expected_inputs = input_shapes(spec, key)
    if not isinstance(inputs, dict) or set(inputs) != set(expected_inputs):
        bad.append("missing input/packing hashes")
    else:
        for name, shape in expected_inputs.items():
            plane = inputs[name]
            if shape is None:
                if plane is not None:
                    bad.append("input %s must be explicit None" % name)
            elif (not isinstance(plane, dict) or not _digest(plane.get("sha"))
                  or plane.get("shape") != shape or plane.get("dtype") !=
                  ("torch.bool" if name == "bos_mask" else "torch.bfloat16")):
                bad.append("input %s: missing/invalid raw hash, shape or dtype" % name)
    try:
        tags, mode = _tags(row)
        if det and mode != "direct":
            bad.append("deterministic generic dQ must be direct, not atomic/skipped")
    except ValueError as exc:
        bad.append(str(exc))
        tags, mode = None, None
    for tensor, shape in shapes.items():
        if tensor not in row:
            bad.append("missing tensor key %s (not contract None)" % tensor)
            continue
        plane = row[tensor]
        if shape is None:
            if plane is not None:
                bad.append("%s must be explicit contract None" % tensor)
            continue
        if not isinstance(plane, dict):
            bad.append("%s must be a recorded tensor" % tensor)
            continue
        dtype = "torch.float32" if tensor in ("y_fp32", "lse") else "torch.bfloat16"
        numel = math.prod(shape)
        nz, maximum = plane.get("nonzero"), plane.get("max_abs")
        if (not _digest(plane.get("sha")) or plane.get("shape") != shape
                or plane.get("dtype") != dtype or type(plane.get("numel")) is not int
                or plane["numel"] != numel or type(nz) is not int or not 0 <= nz <= numel
                or type(maximum) not in (int, float) or not math.isfinite(maximum)
                or maximum < 0):
            bad.append("%s: missing/malformed raw hash, shape, dtype or finite stats" % tensor)
            continue
        if (nz == 0) != (maximum == 0):
            bad.append("%s: inconsistent zero statistics" % tensor)
        if tensor in ("dprev_k", "dprev_v"):
            if key.startswith("live-prev") and nz == 0:
                bad.append("live-prev %s is zero: vacuous" % tensor)
            if (key.startswith("dead-prev") or key.startswith("traced-0.9b-prefill")) and nz != 0:
                bad.append("dead-prev %s is not exactly zero" % tensor)
    if (key.startswith("dead-prev") or key.startswith("traced-0.9b-prefill")):
        if row.get("mask_effect") != "differs":
            bad.append("dead-prev mask_effect must differ (mask anti-vacuity)")
    return bad, (tags, mode)


def compare(base, seam):
    """Return an auditable report; no torch, GPU or mutable environment needed."""
    report = {"errors": [], "same": 0, "det_same": 0, "different": [], "none": 0, "atomic": []}
    bad = report["errors"]
    specs, calls, modes = case_specs(), {}, {}
    for side, doc in (("base", base), ("stageb", seam)):
        if not isinstance(doc, dict):
            bad.append("%s: document is not an object" % side)
            continue
        for field, expected in (("label", side), ("body_arm", "generic"),
                                ("byte_hash", HASH_KIND), ("calls_per_case", 2),
                                ("capability", [10, 3])):
            if doc.get(field) != expected:
                bad.append("%s: %s must be %r" % (side, field, expected))
        if not _digest(doc.get("extension_sha256")):
            bad.append("%s: missing/invalid loaded extension sha256" % side)
        tree = doc.get("tree")
        if not isinstance(tree, dict) or not _digest(tree.get("csrc/flash_sca/native/host.cc")):
            bad.append("%s: missing/invalid full host.cc source sha256" % side)
        bad.extend("%s: %s" % (side, x) for x in source_errors(doc.get("generic_dq_source")))
        rows = doc.get("cases")
        if not isinstance(rows, dict):
            bad.append("%s: missing cases object" % side)
            continue
        if set(rows) != set(specs):
            bad.append("%s: case manifest mismatch; missing=%s extra=%s" %
                       (side, sorted(set(specs) - set(rows)), sorted(set(rows) - set(specs))))
        for key, spec in specs.items():
            row = rows.get(key)
            repeats = row.get("repeats") if isinstance(row, dict) else None
            if not isinstance(repeats, list) or len(repeats) != 1:
                bad.append("%s %s: need one independent repeated call" % (side, key))
                repeats = []
            calls[side, key] = [row] + repeats
            for i, call in enumerate(calls[side, key]):
                problems, evidence = _check_call(call, spec, key)
                bad.extend("%s %s call%d: %s" % (side, key, i, x) for x in problems)
                modes[side, key, i] = evidence
    # No malformed/missing records can license the atomic exemption. In
    # particular two empty JSONs, missing hashes or stale old-schema files fail.
    if bad:
        return report
    if base["generic_dq_source"]["files"] != seam["generic_dq_source"]["files"]:
        bad.append("generic implementation source pins differ between the builds")
        return report
    for key, (_shapes, det, _level, _chunk) in specs.items():
        pairs = (("base self", ("base", 0), ("base", 1)),
                 ("stageb self", ("stageb", 0), ("stageb", 1)),
                 ("cross call0", ("base", 0), ("stageb", 0)),
                 ("cross call1", ("base", 1), ("stageb", 1)))
        for label, (sa, ia), (sb, ib) in pairs:
            a, b = calls[sa, key][ia], calls[sb, key][ib]
            ta, ma = modes[sa, key, ia]
            tb, mb = modes[sb, key, ib]
            if ta != tb or ma != mb:
                bad.append("%s %s: serving tags/write modes differ" % (key, label))
                continue
            if a["inputs"] != b["inputs"] or a["options"] != b["options"]:
                bad.append("%s %s: inputs/packing/options differ; not a matched pair" % (key, label))
                continue
            for tensor in TENSORS:
                pa, pb = a[tensor], b[tensor]
                if pa is None and pb is None:
                    report["none"] += 1  # checked None, not byte-equivalent tensors
                    continue
                # All non-null planes were validated above, including dtype and
                # shape. Do NOT read any row dq_mode/preparation snapshot here.
                equal = pa["sha"] == pb["sha"]
                if tensor == "dq" and not det and ma == mb == "atomic":
                    report["atomic"].append({"case": key, "comparison": label, "equal": equal,
                                             "base": pa, "other": pb})
                elif equal:
                    report["same"] += 1
                    report["det_same"] += int(det)
                else:
                    report["different"].append((key, label, tensor, pa["sha"], pb["sha"]))
    return report


def print_report(report):
    for error in report["errors"]:
        print("  FAIL " + error)
    for key, label, tensor, a, b in report["different"]:
        print("  DIFFER %s %s %s %s != %s" % (key, label, tensor, a, b))
    for row in report["atomic"]:
        a, b = row["base"], row["other"]
        print("  ATOMIC-DQ %s %s %s: %s vs %s; nonzero=%d/%d max_abs=%g/%g (UNASSERTED)" %
              (row["case"], row["comparison"], "same" if row["equal"] else "DIFFERS",
               a["sha"], b["sha"], a["nonzero"], b["nonzero"], a["max_abs"], b["max_abs"]))
    print("stable-writer byte comparisons: %d same, %d differ; explicit None cells: %d "
          "(not tensors, not counted identical)" %
          (report["same"], len(report["different"]), report["none"]))
    print("deterministic byte comparisons (no bounds row): %d same, %d differ" %
          (report["det_same"], sum(key.endswith("/det=1") for key, *_ in report["different"])))
    print("det-off atomic dQ: %d comparisons, %d differ; UNASSERTED even when repeats match; "
          "needs the independent fixed-FP64 gate" %
          (len(report["atomic"]), sum(not x["equal"] for x in report["atomic"])))
    ok = not report["errors"] and not report["different"]
    print("PAIR: %s (structural/stable-writer byte checks; forced-generic backward only)" %
          ("PASS" if ok else "FAIL"))
    print("Numerical correctness: NOT TESTED. No all-plane equivalence or native/device proof claimed.")
    return 0 if ok else 1


def selftest():
    """Synthetic CPU records exercise the comparator, not attention numerics."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    source = generic_source_evidence(root)
    problems = source_errors(source)
    if problems:
        print("SELFTEST FAIL: held source: " + "; ".join(problems))
        print(json.dumps(source, sort_keys=True))
        return 1
    doc = {"label": "base", "body_arm": "generic", "byte_hash": HASH_KIND,
           "capability": [10, 3],
           "calls_per_case": 2, "generic_dq_source": source, "cases": {},
           "extension_sha256": "e" * 64, "tree": {"csrc/flash_sca/native/host.cc": "f" * 64}}
    for key, spec in case_specs().items():
        shapes, det, level, chunk = spec
        dead = key.startswith(("dead-prev", "traced-0.9b-prefill"))
        inputs = {name: None if shape is None else {"sha": "c" * 64, "shape": shape,
                  "dtype": "torch.bool" if name == "bos_mask" else "torch.bfloat16"}
                  for name, shape in input_shapes(spec, key).items()}
        row = {"body_arm": "generic", "deterministic": det, "y_fp32_is_none": level == 0,
               "inputs": inputs, "options": call_options(level, chunk),
               "mask_effect": "differs" if dead else "same", "phase_tags": [
                   ["forward", "mma_sync", "mma_sync", "ran", 1],
                   ["delta", "scalar", "scalar", "ran", 1],
                   ["dq", "mma_sync", "mma_sync", "ran", 1] if det else
                   ["dq", "none", "none", "skipped", 1],
                   ["dk", "mma_sync", "mma_sync", "ran", 1],
                   ["dv", "mma_sync", "mma_sync", "ran", 1]]}
        for tensor, shape in shapes.items():
            zero = dead and tensor.startswith("dprev_")
            row[tensor] = None if shape is None else {
                "sha": "a" * 64, "shape": shape, "numel": math.prod(shape),
                "dtype": "torch.float32" if tensor in ("lse", "y_fp32") else "torch.bfloat16",
                "nonzero": 0 if zero else math.prod(shape), "max_abs": 0.0 if zero else 1.0}
        row["repeats"] = [copy.deepcopy(row)]
        doc["cases"][key] = row
    k = "live-prev/L0/det=0"
    d = "live-prev/L0/det=1"
    tests = []

    def test(name, mutate, expected):
        a, b = copy.deepcopy(doc), copy.deepcopy(doc)
        b["label"] = "stageb"
        mutate(a, b)
        result = compare(a, b)
        ok = not result["errors"] and not result["different"]
        passed = ok == expected
        tests.append(passed)
        print("selftest %-44s %s" % (name, "PASS" if passed else "FAIL"))
        return result

    def change(b, key=k, tensor="dq", repeat=False):
        row = b["cases"][key]
        if repeat:
            row = row["repeats"][0]
        row[tensor]["sha"] = "b" * 64

    def direct(a, b):
        for obj in (a, b):
            for row in (obj["cases"][k], obj["cases"][k]["repeats"][0]):
                row["phase_tags"][2] = ["dq", "mma_sync", "mma_sync", "ran", 1]
        change(b)

    test("valid complete pair", lambda a, b: None, True)
    r = test("atomic cross differences report only", lambda a, b: change(b), True)
    tests.append(any(not x["equal"] for x in r["atomic"]) and r["none"] > 0)
    test("atomic self differences report only", lambda a, b: change(b, repeat=True), True)
    test("atomic nonzero count not an invariant",
         lambda a, b: b["cases"][k]["dq"].update(sha="b" * 64, nonzero=1), True)
    test("det-off direct dQ differences fail", direct, False)
    test("deterministic cross differences fail", lambda a, b: change(b, key=d), False)
    test("deterministic self differences fail", lambda a, b: change(b, key=d, repeat=True), False)
    for tensor in TENSORS:
        test("L1 stable plane " + tensor, lambda a, b, t=tensor:
             change(b, key="live-prev/L1/det=0", tensor=t) if t != "dq" else
             change(b, key=d, tensor=t), False)
    test("missing row BOTH sides", lambda a, b: (a["cases"].pop(k), b["cases"].pop(k)), False)
    test("extra row", lambda a, b: b["cases"].update(extra={}), False)
    test("missing tensor BOTH sides", lambda a, b:
         (a["cases"][k].pop("dq"), b["cases"][k].pop("dq")), False)
    test("None key absent != explicit None", lambda a, b: b["cases"][k].pop("y_fp32"), False)
    test("required tensor replaced by None", lambda a, b: b["cases"][k].update(dq=None), False)
    test("None on only one side", lambda a, b: b["cases"][k].update(y_fp32=b["cases"][k]["y"]), False)
    test("missing hash", lambda a, b: b["cases"][k]["dq"].pop("sha"), False)
    test("truncated old hash", lambda a, b: b["cases"][k]["dq"].update(sha="a" * 32), False)
    test("wrong dtype", lambda a, b: b["cases"][k]["dq"].update(dtype="torch.float32"), False)
    test("nonfinite stats", lambda a, b: b["cases"][k]["dq"].update(max_abs=float("nan")), False)
    test("missing repeats", lambda a, b: b["cases"][k].pop("repeats"), False)
    test("repeat row error", lambda a, b: b["cases"][k]["repeats"][0].update(error="launch"), False)
    test("missing tags", lambda a, b: b["cases"][k].pop("phase_tags"), False)
    test("duplicate phase", lambda a, b: b["cases"][k]["phase_tags"].__setitem__(2,
         ["dk", "mma_sync", "mma_sync", "ran", 1]), False)
    test("non-unit count", lambda a, b: b["cases"][k]["phase_tags"][0].__setitem__(4, 2), False)
    test("native body despite generic request", lambda a, b: b["cases"][k]["phase_tags"].__setitem__(3,
         ["dk", "umma", "umma", "ran", 1]), False)
    test("forward body mismatch", lambda a, b: b["cases"][k]["phase_tags"].__setitem__(0,
         ["forward", "umma", "umma", "ran", 1]), False)
    test("unset request", lambda a, b: b.update(body_arm="<unset>"), False)
    test("per-call unset request", lambda a, b: b["cases"][k].update(body_arm="<unset>"), False)
    test("det flag mismatch", lambda a, b: b["cases"][k].update(deterministic=True), False)
    test("wrong GPU capability", lambda a, b: b.update(capability=[9, 0]), False)
    test("missing loaded binary pin", lambda a, b: b.pop("extension_sha256"), False)
    test("missing exact host source pin", lambda a, b: b.pop("tree"), False)
    test("missing inputs", lambda a, b: b["cases"][k].pop("inputs"), False)
    test("different input", lambda a, b: b["cases"][k]["inputs"]["q"].update(sha="d" * 64), False)
    test("different packing", lambda a, b: b["cases"][k]["inputs"]["bos_mask"].update(sha="d" * 64), False)
    test("different chunk", lambda a, b: b["cases"][k]["options"].update(chunk_size=32), False)
    test("missing source rule", lambda a, b: b.pop("generic_dq_source"), False)
    test("source check failed", lambda a, b:
         b["generic_dq_source"]["checks"].update({"launch.cuh:dq_tag_branch": False}), False)
    test("generic source mismatch", lambda a, b:
         b["generic_dq_source"]["files"].update({"launch.cuh": "0" * 64}), False)
    for tensor in ("dprev_k", "dprev_v"):
        test("vacuous live " + tensor, lambda a, b, t=tensor:
             b["cases"][k][t].update(nonzero=0, max_abs=0.0), False)
        test("nonzero dead " + tensor, lambda a, b, t=tensor:
             b["cases"]["dead-prev/L0/det=0"][t].update(nonzero=1, max_abs=1.0), False)
    test("mask never arrived", lambda a, b:
         b["cases"]["dead-prev/L0/det=0"].update(mask_effect="same"), False)
    # A stale diagnostic must not overrule the actual per-call tag/source join.
    test("stale dq_mode ignored", lambda a, b:
         (b["cases"][k].update(dq_mode="direct"), change(b)), True)
    test("empty pair fails", lambda a, b: (a.clear(), b.clear()), False)
    print("selftest: %d/%d PASS (CPU comparator/source lint only)" % (sum(tests), len(tests)))
    return 0 if all(tests) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    out = os.environ.get("OUT", "/tmp")
    try:
        with open(os.path.join(out, "seam_inertness_base.json"), encoding="utf-8") as fh:
            base = json.load(fh)
        with open(os.path.join(out, "seam_inertness_stageb.json"), encoding="utf-8") as fh:
            seam = json.load(fh)
    except (OSError, ValueError) as exc:
        print("PAIR: FAIL (missing/unreadable evidence: %s)" % exc)
        return 1
    return print_report(compare(base, seam))


if __name__ == "__main__":
    sys.exit(main())
