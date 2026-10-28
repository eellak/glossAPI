"""Ship tracked+new working files via rsync, with a content receipt; never write git."""
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ''):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'tools.harness'
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from .common import REPO, code_receipt, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node-dir', required=True); p.add_argument('--receipt', required=True)
    a = p.parse_args()
    node = Path(a.node_dir).resolve()
    receipt = code_receipt()
    remote = f"/home/ubuntu/runs/harness-code-{receipt['commit'][:12]}-{receipt['tree_sha256'][:16]}/repo"
    ssh = ['ssh', '-i', str(Path.home()/'.ssh/prime_intellect_key'), '-o', f'UserKnownHostsFile={node}/known_hosts', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15']
    host = 'ubuntu@195.242.31.40'
    subprocess.run([*ssh, host, f'mkdir -p {shlex.quote(remote)}'], check=True)
    with tempfile.TemporaryDirectory(prefix='glossapi-ship-') as t:
        listing = Path(t)/'files'
        listing.write_bytes(('\0'.join(receipt['files'])+'\0').encode())
        subprocess.run(['rsync', '-rlt', '--checksum', '--link-dest=/home/ubuntu/glossapi', '--from0', f'--files-from={listing}', '-e', shlex.join(ssh), str(REPO)+'/', host+':'+remote+'/'], check=True)
        bound = Path(t)/'.code-receipt.json'
        write_json(bound, receipt)
        subprocess.run(['rsync', '-t', '-e', shlex.join(ssh), str(bound), host+':'+remote+'/.code-receipt.json'], check=True)
    receipt['remote_repo'] = remote
    write_json(a.receipt, receipt)
    print(remote)

if __name__ == '__main__': main()
