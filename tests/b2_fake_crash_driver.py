"""Fresh-exec fake workload for independent crash-loop certification."""
import json
import os
from pathlib import Path
import resource
import signal
import sys
from types import SimpleNamespace

from glossapi.scripts import batched_stability_benchmark as b
from glossapi.scripts.docling_cost_benchmark import fake_docling, _FakeStatus

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
root = Path(sys.argv[sys.argv.index('--out') + 1])
b.reference.page_count = lambda path: 2
b.reference.docling_version = lambda: '2.81.0'
b.reference.equality = lambda root, names, fidelity: [dict(file=n, passes_gate=True) for n in names]

class Adapter:
    def __init__(self, config, device):
        self.config = config
        self.pipeline = SimpleNamespace(pipeline_options=SimpleNamespace(model_dump=lambda **kw: {'do_ocr': False, 'layout_batch_size': 4 if config['mode']=='sequential' else 8}))
        self.converter = SimpleNamespace(convert=self.convert)

    def convert(self, path, **kwargs):
        b.append_jsonl(root/'conversions.jsonl', dict(file=path.name, mode=self.config['mode'], resume='--resume' in sys.argv))
        # Signal while inside the convert/unload surface, before it returns.
        marker = root/'already_crashed'
        if self.config['mode']=='sequential' and path.name=='b.pdf' and not marker.exists():
            marker.touch()
            os.kill(os.getpid(), signal.SIGABRT)
        return SimpleNamespace(input=SimpleNamespace(file=path), pages=[1,2], errors=[], status=_FakeStatus.SUCCESS)

    def export(self, result, out):
        (out/f'{result.input.file.stem}.md').write_text('identical output')

    def failure(self, *args):
        pass

b.make_adapter = Adapter
# This probe targets orchestration/receipt logic; run both arms with the same
# cheap scoped document primitive instead of model/parser work.
b.batched_documents = b.sequential_documents
original_launch = b.launch_process
def fresh_launch(command):
    b.append_jsonl(root/'launches.jsonl', dict(command=command))
    return original_launch([sys.executable, __file__, *command[3:]])
b.launch_process = fresh_launch
with fake_docling('baseline', {}):
    raise SystemExit(b.main(sys.argv[1:]))
