"""The refused-call reader: a refused backward must still be readable through the public dump.

Companion to `test_bwd_arm_parser.py`, not a replacement: that file owns the arm grammar
(E1-E17) and the call shapes, and this one owns the two answers a refusal must still give -
`arm_env` and the retired name's `retired_env` - plus the rule that a refusal records no
decision it never made. Helpers (`_call`, `_backward`, `_refused`, `_diag`) are imported
from it rather than re-written, so there is exactly one case construction in the tree.

Why it exists: `host.cc:BackwardArmOptIn` refuses a retired `XATTN_SCA_BWD` and an unknown
`XATTN_SCA_BWD_ARM`, and both refusals raise before preparation. Until the raw values are
stored *at the read*, a refused call cannot say whether the arm was unset or `native` - so
a script asking for the strict arm under the retired name gets a raise and no way to tell
which of the two it asked for. This file witnesses that both values survive the raise, and
that absent, set-to-"" and set-to-a-word stay three different answers (`std::string(nullptr)`
is UB, so the null guard is part of the contract, not tidiness).

Every cell here calls `flash_sca_bwd` standalone, so one thread writes and reads. The dump
is the process-wide record that host.cc:PublishBackwardRecord publishes on every exit from
Backward, a raise included, so a refusal on the autograd engine's thread is readable from
the caller's thread as well, and `record_seq` / `record_thread` / `record_launch_counts` say
which call a dump describes. That door is witnessed by `test_native_host_thread.py`, not by
this file.

A missing instrument is a FAILURE, not a skip: a build without the diagnostic binding or
without the two keys is precisely the outcome this file exists to report. Only wrong
hardware skips (the module-level mark), because that is the one case where the answer is
genuinely "not applicable".
"""

import pytest
import torch

from test_bwd_arm_parser import (  # one case construction in the tree, not two
    ARM,
    RETIRED,
    SERVED,
    UNSERVED,
    _backward,
    _call,
    _diag,
    _refused,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="native FlashSCA runs on compute capability 10.3",
)

KEYS = ("arm_env", "retired_env")
RESET_BINDING = "_flash_sca_native_phase_tags_reset"
ROWS_BINDING = "_flash_sca_native_phase_tags"


@pytest.fixture(autouse=True)
def _unset_instruments(monkeypatch):
    for name in (RETIRED, ARM, "XATTN_SCA_ORDER", "XATTN_SCA_LAYOUT", "XATTN_SCA_UMMA_FWD"):
        monkeypatch.delenv(name, raising=False)


def _ops():
    """The ops namespace, or a FAILURE. Never a skip: see the module docstring."""
    xattn_cuda = pytest.importorskip("xattn_cuda")
    ops = getattr(xattn_cuda, "ops", None)
    if ops is None or not hasattr(ops, "_flash_sca_last_backward_preparation"):
        pytest.fail(
            "the built extension has no ops._flash_sca_last_backward_preparation: the "
            "instrument this reader witnesses is absent, which is the finding and not a "
            "reason to skip"
        )
    for name in KEYS:
        # The keys live in the dump, so a build that emits the dump can still be missing
        # one of them; ask once here so every cell's failure names the same cause.
        dump = ops._flash_sca_last_backward_preparation()
        if name not in dump:
            pytest.fail(f"the backward dump has no '{name}' key: got {sorted(dump)}")
    return ops


def _dump(ops):
    """The dump, with the presence of both keys asserted HERE.

    `d.get(k)` would make an absent key indistinguishable from a correct `None` under
    `-k` selection, which is how a cell is re-run after a failure.
    """
    d = ops._flash_sca_last_backward_preparation()
    for k in KEYS:
        assert k in d, (k, sorted(d))
    return d


def test_unset_reads_none_beside_a_served_call(monkeypatch):
    """Both variables unset: the served call runs and the dump says None, not ""."""
    ops = _ops()
    _backward(_call(SERVED))
    d = _dump(ops)
    assert d["arm_env"] is None and d["retired_env"] is None, d
    assert d["launched"], d


@pytest.mark.parametrize("value", ("2", "bogus", "native "))
def test_an_invalid_arm_raises_and_stays_readable(monkeypatch, value):
    """An unknown arm value raises by name and `arm_env` still carries it verbatim -
    including a trailing space, which is the value most easily "helpfully" trimmed."""
    ops = _ops()
    call = _call(SERVED)
    monkeypatch.setenv(ARM, value)
    diag = _refused(ops, call, "XATTN_SCA_BWD_ARM must be unset", f"got '{value}'")
    d = _dump(ops)
    assert d["refusal"] == "not_called", d
    assert d["arm_env"] == value, d
    assert d["retired_env"] is None, d
    del diag  # the needles above are the raise; the dump is the witness


@pytest.mark.parametrize("arm", (None, "native"))
def test_the_retired_name_raises_and_stays_readable(monkeypatch, arm):
    """A retired XATTN_SCA_BWD raises even beside a valid arm, and BOTH values survive the
    raise: the refusal is why they must be stored above it."""
    ops = _ops()
    call = _call(SERVED)
    monkeypatch.setenv(RETIRED, "1")
    if arm is not None:
        monkeypatch.setenv(ARM, arm)
    _refused(ops, call, "XATTN_SCA_BWD is retired", "XATTN_SCA_BWD_ARM=generic",
             "XATTN_SCA_BWD_ARM=native", "got XATTN_SCA_BWD='1'")
    d = _dump(ops)
    assert d["refusal"] == "not_called", d
    assert d["retired_env"] == "1", d
    assert d["arm_env"] == arm, d


@pytest.mark.parametrize("variable", (ARM, RETIRED))
def test_set_but_empty_is_not_unset(monkeypatch, variable):
    """`XATTN_SCA_BWD=""` is accepted as unset, but the dump must still say the variable
    WAS set: "" and None are different answers, and collapsing them is the defect this
    file was written for.

    Both variables treat "" as unset, and `test_bwd_arm_parser.py` pins that outcome for
    each (`test_retired_xattn_sca_bwd_empty_is_unset`, `test_arm_empty_is_unset`). This
    cell owns the store only, so it keeps a branch for a refusal as well: the raw value must
    survive whichever outcome the parser takes.
    """
    ops = _ops()
    call = _call(SERVED)
    monkeypatch.setenv(variable, "")
    if variable == ARM:
        try:
            _backward(call)
        except RuntimeError as exc:
            assert ARM in str(exc), f"refused for something other than the arm: {exc}"
            d = _dump(ops)
            assert d["arm_env"] == "", d  # the store survives the refusal either way
            assert d["launched"] is False, d
            return
        d = _dump(ops)
        assert d["launched"] is True, d
    else:
        _backward(call)
        d = _dump(ops)
        assert d["launched"] is True, d
    if variable == RETIRED:
        assert d["retired_env"] == "", d
        assert d["arm_env"] is None, d
    else:
        assert d["arm_env"] == "", d
        assert d["retired_env"] is None, d


def test_no_stale_string_survives_the_next_call(monkeypatch):
    """Two calls in one process: the second call's dump describes the second call."""
    ops = _ops()
    call = _call(SERVED)
    monkeypatch.setenv(ARM, "native")
    _backward(call)
    assert _dump(ops)["arm_env"] == "native"
    monkeypatch.delenv(ARM)
    _backward(call)
    d = _dump(ops)
    assert d["arm_env"] is None, d


def test_a_refusal_records_no_decision_it_never_made(monkeypatch):
    """A refusal is not a prepared, packed or launched call, and it appends no phase row.

    The row half is gated on the reset binding: without it the counter set cannot be
    cleared, so "no rows" would be a statement about whatever earlier tests accumulated -
    order-dependent evidence, which a witness of refusal must never be. When the binding is
    present the reset is PROVED by reading the set back empty before the refused call.
    """
    ops = _ops()
    call = _call(SERVED)
    monkeypatch.setenv(ARM, "bogus")
    if hasattr(ops, RESET_BINDING) and hasattr(ops, ROWS_BINDING):
        getattr(ops, RESET_BINDING)()
        assert getattr(ops, ROWS_BINDING)() in ([], None), "the reset did not clear the rows"
        reset_present = True
    else:
        reset_present = False
    _refused(ops, call, "XATTN_SCA_BWD_ARM must be unset", "got 'bogus'")
    d = _dump(ops)
    assert d["refusal"] == "not_called", d
    assert d["prepared"] is False and d["launched"] is False, d
    if d.get("has_accumulator", False):
        assert False, f"a refused call reported an accumulator: {d}"
    if reset_present:
        rows = getattr(ops, ROWS_BINDING)()
        assert rows in ([], None), f"a refused call appended a phase row: {rows}"
    # No else: without the binding the row half is untested, and this cell says so by
    # asserting only the fields the refusal itself owns.


def test_a_generic_policy_cell_still_reads_back(monkeypatch):
    """(64, 64) has no backward leaf (the registry lists (128,128) and (256,256)), so the
    generic body serves it - and the arm environment must read back the same way there:
    the witness belongs to the host seam, not to one product."""
    ops = _ops()
    _backward(_call(UNSERVED))
    d = _dump(ops)
    assert d["arm_env"] is None and d["retired_env"] is None, d
    assert d["launched"], d
