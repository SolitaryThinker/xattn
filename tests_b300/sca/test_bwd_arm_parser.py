"""XATTN_SCA_BWD_ARM, the one backward body-arm parser, through the public backward.

Replaces E1-E17 of the retired XATTN_SCA_BWD parser test (the CheckBodyArm cells an
earlier backward_phases_check.cu ran without a device). The parse is now
host.cc:BackwardArmOptIn and the strict gate sits inside the torch-facing Backward(), so
each old check (named in its test's docstring) runs the public xattn.flash_sca_bwd
(level 0, BF16, deterministic=False) on a cc 10.3 device and reads the process-wide
_flash_sca_last_backward_preparation() record and the phase tags. E14 (require refuses a native claim over a generic product) has no public
equivalent: it needs a mixed plan, and the dQ and dK/dV entries share one bucket list and
accept the same public calls (umma_bwd_dq.cuh:CanLaunchUmmaDq, umma_bwd_dkdv.cuh:Accepted).
The planner's mixed rows stay in backward_phases_check.cu (T-dq-fast, Q7, Q14), which does
not reach host.cc, so the gate's mixed branch (host.cc:558-559) stays untested.

Calls: D = Dv = 128 is served by both registered products. D = Dv = 64 is bucket pair
(64, 64) (dimension_bucket.h:14), which has no backward leaf: umma_bwd_entry.h:14 lists
(128, 128) and (256, 256) only, the registry is built from that list
(instantiations/sca_umma_bwd_registry.cu:35, FindBackwardLeaf :40), and PrepareBackwardUmma
refuses it with Refusal::kBucket (umma_prepare.h:180-181) before any leaf predicate runs;
the generic mma.sync body serves it.
"""

import pytest
import torch
import torch.nn.functional as F

import xattn
from xattn.ops import _sca_native


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="native FlashSCA runs on compute capability 10.3",
)

ARM = "XATTN_SCA_BWD_ARM"
RETIRED = "XATTN_SCA_BWD"
SERVED, UNSERVED = 128, 64  # D = Dv of the two calls (module docstring)
CHUNK = 128
ALL_STAGES, DELTA_ONLY = 7, 1  # backward_stage_mask.h: kBackwardAllStages, kStageDelta
NATIVE_ROWS = {(phase, "umma", "umma", "ran") for phase in ("dq", "dk", "dv")}
REQUIRE = "XATTN_SCA_BWD_ARM=native requires both native backward products"


def _ops():
    xattn_cuda = pytest.importorskip("xattn_cuda")
    ops = getattr(xattn_cuda, "ops", None)
    if ops is None or not hasattr(ops, "_flash_sca_last_backward_preparation"):
        pytest.skip("build does not link the native FlashSCA backward diagnostic")
    return ops


@pytest.fixture(autouse=True)
def _unset_instruments(monkeypatch):
    # Every per-call instrument the native host reads starts unset, whatever the job set.
    for name in (RETIRED, ARM, "XATTN_SCA_ORDER", "XATTN_SCA_LAYOUT", "XATTN_SCA_UMMA_FWD"):
        monkeypatch.delenv(name, raising=False)


def _call(dim):
    """Inputs and level-0 forward of one public call: B=1, N=256, Hq=4, Hkv=2, D=Dv=dim."""
    torch.manual_seed(0)
    q = F.normalize(torch.randn(1, 256, 4, dim, device="cuda"), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, 256, 2, dim, device="cuda"), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, 256, 2, dim, device="cuda").to(torch.bfloat16)
    dy = torch.randn(1, 256, 4, dim, device="cuda").to(torch.bfloat16)
    y, lse = xattn.flash_sca_fwd(q, k, v, CHUNK)
    assert y.dtype == q.dtype, "level 0 returns y in the Q dtype"
    return dy, q, k, v, y, lse


def _backward(call):
    """The public standalone backward, on this thread: one call at a time, so the record
    the next dump returns is this call's (test_e1 checks its stamps)."""
    dy, q, k, v, y, lse = call
    grads = xattn.flash_sca_bwd(dy, q, k, v, y, lse, CHUNK, deterministic=False)
    assert len(grads) == 5 and grads[3] is None and grads[4] is None
    for grad, ref in zip(grads[:3], (q, k, v)):
        assert grad.shape == ref.shape and grad.dtype == ref.dtype
        assert bool(torch.isfinite(grad.float()).all()), "a served backward must be finite"


def _diag(ops):
    return ops._flash_sca_last_backward_preparation()


def _product_rows(ops):
    return {row[:4] for row in ops._flash_sca_native_phase_tags() if row[0] in ("dq", "dk", "dv")}


def _refused(ops, call, *needles):
    """The backward raises RuntimeError naming every needle, and launches nothing."""
    before = _sca_native.launch_counts()
    with pytest.raises(RuntimeError) as excinfo:
        _backward(call)
    for needle in needles:
        assert needle in str(excinfo.value), str(excinfo.value)
    assert _sca_native.launch_counts() == before, "a refused backward must not launch"
    diag = _diag(ops)
    assert diag["launched"] is False, diag
    return diag


def test_e1_unset_is_the_shipped_arm():
    """Old E1-unset-is-the-shipped-arm: neither variable set runs the default arm, which
    selects every eligible producer - so the served call is native in both products and
    the generic arm of E2-E5 is a real override."""
    ops = _ops()
    call = _call(SERVED)
    seq_before = _diag(ops)["record_seq"]
    _backward(call)
    diag = _diag(ops)
    # The record is attributable: a newer sequence number, written on this thread, with the
    # launch counters this call left behind.
    assert diag["record_seq"] > seq_before and diag["recorded_on_this_thread"] is True, diag
    assert diag["record_launch_counts"] == _sca_native.launch_counts(), diag
    assert diag["arm_env"] is None and diag["body_arm"] == "auto" and diag["arm"] == "off", diag
    assert diag["force_generic"] is False and diag["require_native"] is False, diag
    assert diag["launched"] is True and diag["prepared"] is True, diag
    assert diag["native_dq"] is True and diag["native_key"] is True, diag


def test_e2_e5_generic_arm_forces_the_generic_bodies(monkeypatch):
    """Old E2-force-generic-plans, E3-force-generic-mask-is-all,
    E4-force-generic-record-is-empty, E5-force-generic-keeps-the-seed-accumulator:
    XATTN_SCA_BWD=0 then, XATTN_SCA_BWD_ARM=generic now. On a call both leaves serve,
    preparation is skipped (host.cc:536) and every stage is generic. B300 SCA is
    deterministic-only, so even with deterministic=False the generic body writes dQ
    directly (row-owned, fixed order) and keeps no FP32 dQ accumulator (E5)."""
    ops = _ops()
    call = _call(SERVED)
    monkeypatch.setenv(ARM, "generic")
    ops._flash_sca_native_phase_tags_reset()
    _backward(call)  # E2
    diag = _diag(ops)
    assert diag["arm_env"] == "generic" and diag["body_arm"] == "forced_generic", diag
    assert diag["refusal"] == "forced_generic" and diag["prepared"] is False, diag
    assert diag["generic_mask"] == ALL_STAGES, diag  # E3
    assert not (diag["native_dq"] or diag["native_key"] or diag["native_atomic_dq"]), diag  # E4
    # No native key tile ran, so key_rows is the bucket's preferred tile (128 at D=Dv=128),
    # not a launch: the record's rule is to read it only beside native_key true.
    assert diag["key_rows"] == 128, diag
    assert diag["has_accumulator"] is False and diag["dq_mode"] == "direct", diag  # E5
    rows = _product_rows(ops)
    assert all("umma" not in row for row in rows), rows
    assert rows == {("dq", "mma_sync", "mma_sync", "ran"), ("dk", "mma_sync", "mma_sync", "ran"),
                    ("dv", "mma_sync", "mma_sync", "ran")}, rows


def test_e6_e8_native_arm_passes_when_both_leaves_serve(monkeypatch):
    """Old E6-one-is-require, E7-require-plans-with-a-leaf, E8-require-passes-with-a-leaf:
    XATTN_SCA_BWD=1 then, XATTN_SCA_BWD_ARM=native now. Both products are native, only
    delta is generic, and the direct native dQ needs no accumulator."""
    ops = _ops()
    call = _call(SERVED)
    monkeypatch.setenv(ARM, "native")
    ops._flash_sca_native_phase_tags_reset()
    _backward(call)
    diag = _diag(ops)
    assert diag["arm_env"] == "native" and diag["body_arm"] == "require_native", diag  # E6
    assert diag["require_native"] is True and diag["prepared"] is True, diag
    assert diag["native_dq"] is True and diag["native_key"] is True, diag  # E7
    assert diag["launched"] is True and diag["generic_mask"] == DELTA_ONLY, diag  # E8
    assert diag["has_accumulator"] is False, diag
    assert _product_rows(ops) == NATIVE_ROWS


def test_e9_e13_native_arm_refuses_a_call_no_leaf_serves(monkeypatch):
    """Old E9-require-still-plans-the-generic-fallback, E10-fallback-mask-is-all,
    E11-require-refuses-no-leaf, E12-require-refusal-has-a-reason,
    E13-require-refuses-a-refused-preparation. D = Dv = 64: preparation refuses (kBucket),
    the host records the generic plan it would run, and the strict arm raises before any
    launch (host.cc:569). With the arm unset that generic plan serves the same call."""
    ops = _ops()
    call = _call(UNSERVED)
    monkeypatch.setenv(ARM, "native")
    diag = _refused(ops, call, REQUIRE, "preparation=bucket", "dq=not_compiled",
                    "key=not_compiled")  # E11, E12
    assert diag["arm_env"] == "native" and diag["body_arm"] == "require_native", diag
    assert diag["refusal"] == "bucket" and diag["prepared"] is False, diag  # E13
    # Deterministic-only on B300: the generic fallback plan writes dQ directly.
    assert diag["generic_mask"] == ALL_STAGES and diag["dq_mode"] == "direct", diag  # E9, E10
    monkeypatch.delenv(ARM)
    _backward(call)
    diag = _diag(ops)
    assert diag["arm_env"] is None and diag["body_arm"] == "auto", diag
    assert diag["refusal"] == "bucket" and diag["launched"] is True, diag
    assert diag["generic_mask"] == ALL_STAGES, diag


@pytest.mark.parametrize("dim", (SERVED, UNSERVED), ids=("served", "unserved"))
@pytest.mark.parametrize("arm, body_arm", ((None, "auto"), ("generic", "forced_generic"),
                                           ("fused", "auto"), ("split", "split")))
def test_e15_e16_other_arms_never_refuse(monkeypatch, dim, arm, body_arm):
    """Old E15-force-generic-never-refuses, E16-default-never-refuses. Only the strict arm
    refuses: unset, "generic", "fused" (the default by name) and "split" (direct dQ, no
    native requirement) serve both the served and the unserved call."""
    ops = _ops()
    call = _call(dim)
    if arm is not None:
        monkeypatch.setenv(ARM, arm)
    _backward(call)
    diag = _diag(ops)
    assert diag["arm_env"] == arm and diag["body_arm"] == body_arm, diag
    assert diag["launched"] is True and diag["require_native"] is False, diag
    assert diag["arm"] == ("split" if arm == "split" else "off"), diag
    if arm == "split":
        assert diag["dq_mode"] == "direct" and diag["has_accumulator"] is False, diag


@pytest.mark.parametrize("value", ("2", "Native", "native ", "auto", "0", "1"), ids=repr)
def test_e17_no_invented_arms(monkeypatch, value):
    """Old E17-unknown-arm-refused: exact labels only (host.cc:BackwardArmOptIn). The empty
    string is not among them: like XATTN_SCA_BWD, XATTN_SCA_BWD_ARM treats it as unset
    (test_arm_empty_is_unset). The parse runs before preparation."""
    ops = _ops()
    call = _call(SERVED)
    monkeypatch.setenv(ARM, value)
    diag = _refused(ops, call, "XATTN_SCA_BWD_ARM must be unset", f"got '{value}'")
    assert diag["refusal"] == "not_called", diag


@pytest.mark.parametrize("arm", (None, "native"))
@pytest.mark.parametrize("value", ("0", "1", "2", "x"))
def test_retired_xattn_sca_bwd_raises(monkeypatch, value, arm):
    """A set XATTN_SCA_BWD raises and names its replacement, even beside a valid
    XATTN_SCA_BWD_ARM, so a script written for it cannot time the default arm under its
    label (host.cc:128-132). The retired name is refused FIRST - and that is exactly why
    both raw values are stored above the refusal: the dump read after the raise must
    still answer what the environment said. This cell used to assert `arm_env is None`
    with the docstring "It is checked first, so the arm is never read", which is the
    defect written down as a requirement: a refused call that cannot say whether the arm
    was unset or `native` cannot tell a script which of the two it asked for. Now both
    values are read back, and `retired_env` carries the retired name's own value."""
    ops = _ops()
    call = _call(SERVED)
    monkeypatch.setenv(RETIRED, value)
    if arm is not None:
        monkeypatch.setenv(ARM, arm)
    diag = _refused(ops, call, "XATTN_SCA_BWD is retired", "XATTN_SCA_BWD_ARM=generic",
                    "XATTN_SCA_BWD_ARM=native", f"got XATTN_SCA_BWD='{value}'")
    assert diag["refusal"] == "not_called", diag
    assert diag["retired_env"] == value, diag
    assert diag["arm_env"] == arm, diag


def test_retired_xattn_sca_bwd_empty_is_unset(monkeypatch):
    """XATTN_SCA_BWD='' is accepted as unset (host.cc:129): the default arm runs."""
    ops = _ops()
    monkeypatch.setenv(RETIRED, "")
    _backward(_call(SERVED))
    diag = _diag(ops)
    assert diag["arm_env"] is None and diag["body_arm"] == "auto" and diag["launched"], diag


def test_arm_empty_is_unset(monkeypatch):
    """XATTN_SCA_BWD_ARM='' is accepted as unset, as XATTN_SCA_BWD='' is: the default arm
    runs and serves both products natively. The raw value is still recorded, so the dump
    says the variable was set ("" rather than None)."""
    ops = _ops()
    monkeypatch.setenv(ARM, "")
    _backward(_call(SERVED))
    diag = _diag(ops)
    assert diag["arm_env"] == "" and diag["body_arm"] == "auto" and diag["arm"] == "off", diag
    assert diag["launched"] is True and diag["require_native"] is False, diag
    assert diag["native_dq"] is True and diag["native_key"] is True, diag


def test_owned_copy_diagnostic_keeps_the_parsed_arm(monkeypatch):
    """The diagnostic owns a copy of the value its call parsed
    (host.cc:61-63, 497). Changing or deleting XATTN_SCA_BWD_ARM with no backward in
    between leaves it unchanged; the next call parses the variable again."""
    ops = _ops()
    call = _call(SERVED)
    monkeypatch.setenv(ARM, "native")
    _backward(call)
    assert _diag(ops)["arm_env"] == "native"
    monkeypatch.setenv(ARM, "generic")
    diag = _diag(ops)
    assert diag["arm_env"] == "native" and diag["body_arm"] == "require_native", diag
    monkeypatch.delenv(ARM)
    diag = _diag(ops)
    assert diag["arm_env"] == "native" and diag["body_arm"] == "require_native", diag
    _backward(call)
    diag = _diag(ops)
    assert diag["arm_env"] is None and diag["body_arm"] == "auto", diag
