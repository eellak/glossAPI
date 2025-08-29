#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-yml', required=True, help='Path to inference.yml for Greek rec')
    ap.add_argument('--out', required=True, help='Output keys file path')
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.in_yml).read_text(encoding='utf-8'))
    chars = data['PostProcess']['character_dict']
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for ch in chars:
            f.write(("'" if ch == "'" else ch) + "\n")
    print(f'Wrote {out} with {len(chars)} keys')

if __name__ == '__main__':
    main()

