"""Opt-in complete shape/long matrices, reproducible process-level sharding, and the proof that every
SoftDelta test ran on the native Blackwell route (copy of tests_b300/sca_main/conftest.py's options)."""
import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption('--full-matrix', action='store_true', help='Include every declared D/V bucket.')
    parser.addoption('--long', action='store_true', help='Include the full-size training calls (FP64 blocks).')
    parser.addoption('--shard-count', type=int, default=1)
    parser.addoption('--shard-index', type=int, default=0)
    parser.addoption('--inventory-output', help='Write selected node IDs and canonical coverage hash.')


def pytest_configure(config):
    config.addinivalue_line('markers', 'full_matrix: complete D/V correctness domain')
    config.addinivalue_line('markers', 'long: full-size training calls')
    config.addinivalue_line('markers', 'torch_reference: an explicit backend="torch" call; must not run native')
    torch.set_num_threads(1)
    count, index = config.getoption('--shard-count'), config.getoption('--shard-index')
    if count < 1 or not 0 <= index < count:
        raise pytest.UsageError('Require shard-count > 0 and 0 <= shard-index < shard-count')


def pytest_collection_modifyitems(config, items):
    count, index = config.getoption('--shard-count'), config.getoption('--shard-index')
    eligible, selected, deselected = [], [], []
    for item in items:
        if (item.get_closest_marker('full_matrix') and not config.getoption('--full-matrix')) or (
                item.get_closest_marker('long') and not config.getoption('--long')):
            deselected.append(item)
            continue
        eligible.append(item.nodeid)
        if int.from_bytes(hashlib.sha256(item.nodeid.encode()).digest()[:8], 'big') % count == index:
            selected.append(item)
        else:
            deselected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
    output = config.getoption('--inventory-output')
    if output:
        canonical = sorted(eligible)
        Path(output).write_text(json.dumps({
            'shard_count': count, 'shard_index': index,
            'coverage_sha256': hashlib.sha256(json.dumps(canonical, separators=(',', ':')).encode()).hexdigest(),
            'canonical_nodeids': canonical, 'selected_nodeids': [item.nodeid for item in selected],
        }, indent=2) + '\n')


# Every test must have run on the native Blackwell route, or only been refused before anything ran.
# Read around each test: the native FlashSCA launch counters, the route records of the public SoftDelta
# calls (`softdelta_route`, resolved native) and of the reads they make (`sca_route` of
# _flash_sca_fwd/_flash_sca_bwd, resolved native), and a spy on the PyTorch reference
# (softdelta_attention._torch_softdelta), which must never run. A test whose only calls were refused
# (softdelta_main_cases.expect_refusal) must have launched nothing. A test marked torch_reference makes an
# explicit backend='torch' call and must run the reference and nothing native. Set SOFTDELTA_MAIN_ROUTE_LOG
# to keep one JSON line per test, with the native per-phase dispatch tags.
@pytest.fixture(autouse=True)
def _native_route_proof(request, monkeypatch):
    import softdelta_main_cases as cases
    from xattn.ops import _native_dispatch, _sca_native
    module = importlib.import_module('xattn.ops.softdelta_attention')
    reference_calls = []
    original = module._torch_softdelta

    def spy(*args, **kwargs):
        reference_calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, '_torch_softdelta', spy)
    _native_dispatch.set_route_trace(True)
    _native_dispatch.clear_route_trace()
    ops = _sca_native._extension_ops()
    ops._flash_sca_native_phase_tags_reset()
    cases.REFUSALS.clear()
    before = _sca_native.launch_counts()
    yield
    after = _sca_native.launch_counts()
    trace = _native_dispatch.route_trace()
    _native_dispatch.set_route_trace(False)
    softdelta = sorted({f"{r.get('api_name')}:{r.get('resolved')}" for r in trace if r.get('kind') == 'softdelta_route'})
    reads = sorted({f"{r.get('api_name')}:{r.get('resolved')}" for r in trace if r.get('kind') == 'sca_route'})
    record = {'test': request.node.nodeid, 'native_fwd_launches': after[0] - before[0],
              'native_bwd_launches': after[1] - before[1], 'softdelta_routes': softdelta, 'read_routes': reads,
              'reference_calls': len(reference_calls), 'refusals': list(cases.REFUSALS),
              'native_phase_tags': sorted({'/'.join(map(str, row)) for row in ops._flash_sca_native_phase_tags()})}
    path = os.environ.get('SOFTDELTA_MAIN_ROUTE_LOG')
    if path:
        with open(path, 'a') as handle:
            handle.write(json.dumps(record) + '\n')
    if request.node.get_closest_marker('torch_reference'):
        assert record['reference_calls'] > 0 and record['native_fwd_launches'] == 0, f'not the reference: {record}'
        assert all(name.endswith(':torch') for name in softdelta), f'not the reference: {record}'
        return
    assert record['reference_calls'] == 0, f'the PyTorch reference ran: {record}'
    assert all(name.endswith(':native') for name in softdelta + reads), f'non-native route: {record}'
    if record['native_fwd_launches'] == 0:
        # Only refusals: each raised before anything ran.
        assert record['refusals'] and record['native_bwd_launches'] == 0 and not softdelta and not reads, (
            f'no native forward launch: {record}')
    if any(name.startswith('_flash_sca_bwd:') for name in reads):
        assert record['native_bwd_launches'] > 0, f'no native backward launch: {record}'
