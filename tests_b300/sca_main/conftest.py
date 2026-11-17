"""Opt-in complete shape/long matrices and reproducible process-level sharding."""
import hashlib
import json
from pathlib import Path

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption('--full-matrix', action='store_true', help='Include every declared D/V bucket.')
    parser.addoption('--long', action='store_true', help='Include long-sequence FP64 sampled-row checks.')
    parser.addoption('--shard-count', type=int, default=1)
    parser.addoption('--shard-index', type=int, default=0)
    parser.addoption('--inventory-output', help='Write selected node IDs and canonical coverage hash.')


def pytest_configure(config):
    config.addinivalue_line('markers', 'full_matrix: complete D/V correctness domain')
    config.addinivalue_line('markers', 'long: long-sequence numerical coverage')
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


# B300 copy: every test must have run on the native Blackwell kernels. The
# native extension counts its forward/backward launches per process, and the
# precision backend records which implementation each level-1 backward
# resolved; both are read around each test, together with the public entry
# points' route records (none may resolve to the generic or matched-reset
# routes) and the native per-phase dispatch tags. Set SCA_MAIN_ROUTE_LOG to a
# path to keep one JSON line per test.
@pytest.fixture(autouse=True)
def _native_route_proof(request):
    import os
    from xattn.ops import _native_dispatch, _precision_backend, _sca_native
    _native_dispatch.set_route_trace(True)
    _native_dispatch.clear_route_trace()
    ops = _sca_native._extension_ops()
    ops._flash_sca_native_phase_tags_reset()
    before = _sca_native.launch_counts()
    del _precision_backend._ROUTE_TRACE[:]
    yield
    after = _sca_native.launch_counts()
    resolved = [record.get('resolved') for record in _precision_backend.route_trace()]
    routes = sorted({str(r.get('resolved')) for r in _native_dispatch.route_trace()
                     if r.get('kind') == 'sca_route'})
    _native_dispatch.set_route_trace(False)
    record = {'test': request.node.nodeid, 'native_fwd_launches': after[0] - before[0],
              'native_bwd_launches': after[1] - before[1], 'precision_backward': resolved,
              'sca_routes': routes,
              'native_phase_tags': [list(map(str, row)) for row in ops._flash_sca_native_phase_tags()]}
    path = os.environ.get('SCA_MAIN_ROUTE_LOG')
    if path:
        with open(path, 'a') as handle:
            handle.write(json.dumps(record) + '\n')
    assert record['native_fwd_launches'] > 0, f'no native forward launch: {record}'
    assert all(name == 'native' for name in resolved), f'non-native precision backward: {record}'
    assert not set(routes) & {'generic', 'matched_reset', 'None'}, f'non-native SCA route: {record}'
