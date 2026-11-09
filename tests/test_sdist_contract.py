"""Distribution regression: run with GLOSSAPI_SDIST after `python -m build`.

The source archive must carry every literal Rust include_bytes!/include_str! asset,
including the native layout reader's first-use PDFium self-test. A source install
alone does not prove the archive is rebuildable.
"""
import os
from pathlib import Path
import re
import tarfile

import pytest


@pytest.mark.skipif(not os.environ.get('GLOSSAPI_SDIST'), reason='requires a built source distribution: GLOSSAPI_SDIST')
def test_sdist_contains_every_rust_compile_time_asset():
    with tarfile.open(os.environ['GLOSSAPI_SDIST']) as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        roots = {name.split('/')[0] for name in members}
        assert len(roots) == 1
        root = roots.pop()
        notice = archive.extractfile(members[root + "/FONT_TABLES_LICENSES.txt"]).read()
        assert b"Adobe" in notice and b"FontTools" in notice and b"Unicode" in notice
        checked = []
        for name, member in members.items():
            if not name.startswith(root + '/rust/') or not name.endswith('.rs'):
                continue
            source = archive.extractfile(member).read().decode('utf-8')
            for relative in re.findall(r'include_(?:bytes|str)!\(\s*"([^"]+)"\s*\)', source):
                # Resolve lexically inside the archive; never extract or trust a host path.
                parts = list(Path(name).parent.parts)
                for part in Path(relative).parts:
                    if part == '..':
                        parts.pop()
                    elif part != '.':
                        parts.append(part)
                asset = '/'.join(parts)
                assert asset in members, f'{name} embeds {relative}, but {asset} is absent from the sdist'
                assert members[asset].size > 0
                checked.append(asset)
        assert any(name.endswith('/fixtures/layout_selftest.pdf') for name in checked), 'layout self-test was not checked'


@pytest.mark.skipif(not os.environ.get('GLOSSAPI_WHEEL') or __import__('sys').platform != 'darwin',
                    reason='requires a built macOS wheel: GLOSSAPI_WHEEL')
def test_macos_wheel_tag_matches_every_native_binary(tmp_path):
    import subprocess
    import zipfile

    wheel = Path(os.environ['GLOSSAPI_WHEEL'])
    tag = wheel.stem.rsplit('-', 1)[-1]
    required = {'arm64', 'x86_64'} if tag.endswith('_universal2') else {tag.rsplit('_', 1)[-1]}
    # x86_64 has an underscore; preserve it as a single architecture.
    if tag.endswith('_x86_64'):
        required = {'x86_64'}
    with zipfile.ZipFile(wheel) as archive:
        binaries = [name for name in archive.namelist() if name.endswith('.so')]
        assert len(binaries) == 3
        for name in binaries:
            target = tmp_path / Path(name).name
            content = archive.read(name)
            assert not any(path in content for path in (b'/Users/', b'/home/foivos')), f'personal build path in {name}'
            target.write_bytes(content)
            actual = set(subprocess.check_output(['lipo', '-archs', str(target)], text=True).split())
            assert required <= actual, f'{wheel.name} advertises {required}, but {name} contains {actual}'
