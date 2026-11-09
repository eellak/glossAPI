"""Build host-native Rust wheels correctly with a universal2 macOS Python.

setuptools derives the default platform from the interpreter, whereas cargo builds
one host architecture unless ARCHFLAGS/explicit platform selection requests another.
"""
import os
from pathlib import Path
import platform
import sys

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools_rust import build_rust


class NativeWheel(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()
        if (sys.platform == 'darwin' and plat.endswith('_universal2')
                and not self.plat_name_supplied and not os.environ.get('ARCHFLAGS')):
            plat = plat.removesuffix('_universal2') + '_' + platform.machine()
        return python, abi, plat


class PortableRust(build_rust):
    def run(self):
        # Rust panic/source locations otherwise embed the builder's home and Cargo
        # registry paths even in stripped release binaries. Keep useful, portable
        # locations without distributing personal machine paths.
        previous = os.environ.get('CARGO_ENCODED_RUSTFLAGS')
        root = Path(__file__).resolve().parent
        cargo = Path(os.environ.get('CARGO_HOME', Path.home() / '.cargo')).resolve()
        mappings = [(Path.home(), '/build/user'), (cargo, '/build/cargo'), (root, '/build/glossapi')]
        # Cargo splits RUSTFLAGS by whitespace, not shell quoting. Its encoded
        # form keeps checkout/home paths containing spaces as single arguments.
        flags = (previous.split('\x1f') if previous else []) if previous is not None else os.environ.get('RUSTFLAGS', '').split()
        flags.extend(f'--remap-path-prefix={source}={target}' for source, target in mappings)
        os.environ['CARGO_ENCODED_RUSTFLAGS'] = '\x1f'.join(flags)
        try:
            super().run()
        finally:
            if previous is None:
                os.environ.pop('CARGO_ENCODED_RUSTFLAGS', None)
            else:
                os.environ['CARGO_ENCODED_RUSTFLAGS'] = previous


setup(cmdclass={'bdist_wheel': NativeWheel, 'build_rust': PortableRust})
