"""Build a local vocabulary from explicitly supplied working-context files."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
from phase0.analysis.text_contract import legacy_training_text


def build(paths,limit=128):
    if not 1<=limit<=512:
        raise ValueError('limit must be between 1 and 512')
    counts=Counter()
    sources=[]
    excluded=set()
    for path in paths:
        path=Path(path)
        if path.stat().st_size>1_000_000:
            raise ValueError('Context files must be at most 1 MB')
        raw=path.read_bytes()
        text=raw.decode('utf-8')
        words=re.findall(r'[^\W_]+',re.sub(r'([a-z])([A-Z])',r'\1 \2',text),flags=re.UNICODE)
        for word in words:
            if len(word)<3:
                continue
            try:
                normalized=legacy_training_text(word)
            except ValueError:
                excluded.add(word)
                continue
            counts[normalized]+=1
        sources.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest()))
    return dict(version=1,created_at=datetime.now(timezone.utc).isoformat(),sources=sources,
        terms=[w for w,n in counts.most_common(limit)],unsupported_terms=sorted(excluded),
        note='Explicit task context only; unsupported spellings retained for future literal models; no truth input')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('files',nargs='+',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=128)
    a=p.parse_args()
    result=build(a.files,a.limit)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    print(f'Wrote {len(result["terms"])} terms; {len(result["unsupported_terms"])} unsupported')


if __name__=='__main__':
    main()
