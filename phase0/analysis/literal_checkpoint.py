"""Create a separate expanded-alphabet checkpoint for future literal-label training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from phase0.analysis import ctcv3 as C
from phase0.analysis.text_contract import LITERAL_TOKENS, expand_head


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    if a.output.exists():
        p.error('Output exists; choose a new checkpoint path')
    model=C.load(C.recipe('hwtmix'),a.source)
    model=expand_head(model.cpu())
    meta=dict(contract='literal-nfc-v1',tokens=LITERAL_TOKENS,model_kwargs=model.kw,
              source=str(a.source.resolve()),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
              trained_new_symbols=False,deployable=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(),metadata=meta),a.output)
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    print(f'Wrote training initialization with {len(LITERAL_TOKENS)} symbols. New symbols are not trained.')


if __name__=='__main__':
    main()
