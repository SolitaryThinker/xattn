"""Refusal probe for the backward launch preparation (umma_prepare.h).

The preparation's job is to refuse a native backward call *before* anything is
written. This probe measures its predicates: one valid fixture, one thing
broken per case, and the refusal each case produced. Every case runs with
D=Dv=128 (a mounted bucket) and D=Dv=64 (an explicitly unmounted bucket).

Two builds of the same source:
  * ``-DXATTN_UMMA_PREPARE_FORCE_BUCKET=1`` bypasses the bucket leg, so the
    later refusals are reachable in both buckets and must match each case;
  * the default build is the shipped predicate: 128/128 must reach the same
    later refusals, while every 64/64 case must report ``bucket`` before any
    later predicate can inspect its broken operand.

These are preparation checks, not leaf launches. The expected statuses are
fixed below, not inferred from the registry being tested. ``kCapability`` is
the one status this host cannot produce: it precedes the bucket check and
needs a device that is not compute capability 10.3; the probe only runs on one
that is.
"""
import os
import sys

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.abspath(os.path.join(HERE, "..", "..", "csrc"))
SRC = os.path.join(HERE, "prepare_refusal_check.cc")
CU_SRC = os.path.join(HERE, "prepare_refusal_check_stub.cu")

# One case per refusal the forced build can reach. The values are
# RefusalName()'s spellings, not the enumerator names: the probe compares the
# string the header reports, so a rename in either place fails here.
EXPECTED_FORCED = {
    "valid": "none",
    "valid_packed_prev": "none",
    "sorted_order": "order_arm",
    "head_fastest": "order_arm",
    "lse_dtype": "state",
    "lse_shape": "state",
    "delta_not_contiguous": "state",
    "saved_shape": "state",
    "dy_dtype": "state",
    "dy_shape": "state",
    "tile_count": "tables",
    "key_tile_count": "tables",
    "dq_shape": "layout",
    "dk_not_contiguous": "layout",
    "dy_unaligned_stride": "state",
    "dprev_pair": "layout",
    "dprev_pair_missing_k": "layout",
    "q_unaligned_stride": "map_source",
    "prev_unaligned_stride": "map_source",
    "prev_shape": "layout",
    "prev_dtype": "layout",
    "input_prev_missing_v": "layout",
    "input_prev_missing_k": "layout",
    "dprev_both_null": "none",
    "dq_accum_extent": "dq_accum",
}
MOUNTED_DIMENSION = 128
UNMOUNTED_DIMENSION = 64
EXPECTED_UNMOUNTED = {case: "bucket" for case in EXPECTED_FORCED}


def build(name, force_bucket):
    flags = ["-DXATTN_UMMA_PREPARE_FORCE_BUCKET=1"] if force_bucket else []
    # Two build details the probe needs and a plain host extension does not:
    #  * the preparation is a host-only header but it takes the device's
    #    properties from ATen's CUDA context, so the module must link
    #    torch_cuda -- which torch's builder does only for an extension with a
    #    CUDA source, hence the stub .cu below;
    #  * the host compile of the .cc still needs the toolkit's include path,
    #    because a host-only extension gets none.
    cuda_home = os.environ.get("CUDA_HOME") or torch.utils.cpp_extension.CUDA_HOME
    include_paths = [CSRC]
    if cuda_home:
        include_paths.append(os.path.join(cuda_home, "include"))
    return load(
        name=name,
        sources=[SRC, CU_SRC],
        extra_include_paths=include_paths,
        extra_cflags=flags,
        with_cuda=True,
        verbose=bool(os.environ.get("PREPARE_PROBE_VERBOSE")),
    )


def check(module, expected, label, dimension):
    observed = module.run_checks(dimension)
    failures = []
    for case in sorted(expected):
        want, got = expected[case], observed.get(case, "<missing>")
        if want != got:
            failures.append("%s[%s]: expected %s, observed %s" % (label, case, want, got))
    extra = sorted(set(observed) - set(expected))
    for case in extra:
        failures.append("%s[%s]: unexpected case, observed %s" % (label, case, observed[case]))
    return observed, failures


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 2
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (10, 3):
        print("SKIP: probe describes an sm_103a build, device is (%d, %d)" % (major, minor))
        return 2

    results = {}
    failures = []
    for force_bucket, mode in ((True, "forced"), (False, "unforced")):
        module = build("xattn_prepare_refusal_" + mode, force_bucket)
        for dimension in (MOUNTED_DIMENSION, UNMOUNTED_DIMENSION):
            label = "%s_%d" % (mode, dimension)
            # Capability has already been checked. Bucket precedes every other
            # refusal: only the unforced, unmounted fixture stops there.
            expected = (EXPECTED_FORCED if force_bucket or dimension == MOUNTED_DIMENSION
                        else EXPECTED_UNMOUNTED)
            observed, more = check(module, expected, label, dimension)
            results[label] = observed
            failures += more

    columns = ("forced_128", "unforced_128", "forced_64", "unforced_64")
    print("%-26s %s" % ("case", " ".join("%-13s" % label for label in columns)))
    for case in sorted(EXPECTED_FORCED):
        print("%-26s %s" % (case, " ".join("%-13s" % results[label].get(case)
                                           for label in columns)))
    print("kCapability is not observed here: it needs a device that is not (10, 3)")
    if failures:
        print("FAIL")
        for line in failures:
            print("  " + line)
        return 1
    print("PASS: %d cases x 2 buckets x 2 builds; exact refusal statuses checked"
          % len(EXPECTED_FORCED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
