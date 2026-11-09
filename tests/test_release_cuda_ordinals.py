"""Container GPU device nodes keep host minor numbers; CUDA uses logical ordinals."""
from types import SimpleNamespace

import pytest

from glossapi import capabilities
from glossapi.ocr import defaults


def hardware(monkeypatch, nodes):
    class DevicePath:
        def __init__(self, path):
            self.path = path

        def glob(self, pattern):
            assert self.path == '/dev' and pattern == 'nvidia[0-9]*'
            return [SimpleNamespace(name=f'nvidia{index}') for index in nodes]

        def exists(self):
            return self.path == '/dev/nvidia0' and 0 in nodes

    monkeypatch.setattr(capabilities, 'Path', DevicePath)
    monkeypatch.setattr(defaults, 'Path', DevicePath)
    monkeypatch.setattr(capabilities.sys, 'platform', 'linux')
    monkeypatch.setattr(capabilities, 'installed_version', lambda name: '2.14.1' if name == 'torch' else None)


@pytest.mark.parametrize('nodes', [(7,), (3, 7), (0,), (0, 1), ()])
def test_sparse_host_nodes_expose_contiguous_cuda_ordinals(monkeypatch, nodes):
    hardware(monkeypatch, nodes)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert capabilities._nvidia_devices() == list(range(len(nodes)))
    rows = [row for row in capabilities.devices() if row['kind'] == 'cuda' and row['hardware'] == 'present']
    assert [row['device_id'] for row in rows] == [f'cuda:{i}' for i in range(len(nodes))]
    assert defaults.planned_device() == ('cuda:0' if nodes else 'cpu')
    assert all(row['state'] == 'unknown_until_probe' for row in rows)


@pytest.mark.parametrize('mask,expected', [
    ('', []), ('  ', []), ('-1', []), ('none', []), ('nodevfiles', []),
    ('7', [0]), ('GPU-first', [0]), ('7,3', [0, 1]), ('GPU-first,GPU-second', [0, 1]),
])
def test_visibility_mask_keeps_logical_ids_and_explicit_disabling(monkeypatch, mask, expected):
    hardware(monkeypatch, (3, 7))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    assert capabilities._nvidia_devices() == expected
    assert defaults.planned_device() == ('cuda:0' if expected else 'cpu')
