"""Held-out-session pilot: residual CTC heads with matched geometry and geometry+RGB inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from phase0.analysis.text_contract import edits
from phase0.analysis import seqctc as S
from phase0.analysis import ctcv4 as V

ROOT = Path('.cache/nextgen/visual_fusion')


def image_features(sid, device):
    folder = S.SESS / sid
    files = [folder / name for name in ('video.mp4','landmarks.parquet','frames.jsonl')]
    signature = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    cache = ROOT/f'{sid}_rgb.npz'
    if cache.exists():
        z = np.load(cache)
        if json.loads(str(z['signature'])) != signature:
            raise ValueError(f'Stale image cache: {cache}')
        return z['features'].astype(np.float32)
    fr = [json.loads(l) for l in files[2].read_text().splitlines()]
    frame_ids = np.array([r['i'] for r in fr])
    tb = pq.read_table(files[1],columns=['i','hand','joint','x','y']).to_pydict()
    pts = np.full((len(fr),2,21,2),np.nan,np.float32)
    rows = np.searchsorted(frame_ids,tb['i'])
    valid = rows < len(fr)
    valid[valid] &= frame_ids[rows[valid]] == np.asarray(tb['i'])[valid]
    pts[rows[valid],np.asarray(tb['hand'])[valid],np.asarray(tb['joint'])[valid]] = np.array([tb['x'],tb['y']]).T[valid]
    mid = float(np.nanmedian(pts[:,:,0,1]))
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net = net.eval().to(device)
    mean = torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    wanted = {int(f):k for k,f in enumerate(frame_ids[::2])}
    features = np.zeros((len(wanted),2,512),np.float32)
    batch, owners = [],[]

    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).permute(0,3,1,2).float().to(device)/255
        with torch.inference_mode():
            z = net((x-mean)/std).cpu().numpy()
        for (k,h),v in zip(owners,z):
            features[k,h] = v
        batch.clear()
        owners.clear()

    cap = cv2.VideoCapture(str(files[0]))
    i,seen = 0,0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            k = wanted[i]
            hands = [p for p in pts[k*2] if np.isfinite(p).all()]
            hands.sort(key=lambda p:p[0,1])
            for j,p in enumerate(hands):
                h = j if len(hands)==2 else int(p[0,1] > mid)
                lo,hi = p.min(0),p.max(0)
                center = (lo+hi)/2
                side = max(16.,float((hi-lo).max())*1.25)
                scale = 96/side
                matrix = np.array([[scale,0,48-scale*center[0]],[0,scale,48-scale*center[1]]],np.float32)
                crop = cv2.warpAffine(frame,matrix,(96,96))
                hull = cv2.convexHull((p@matrix[:,:2].T+matrix[:,2]).astype(np.int32))
                mask = np.zeros((96,96),np.uint8)
                cv2.fillConvexPoly(mask,hull,255)
                mask = cv2.dilate(mask,np.ones((5,5),np.uint8))
                crop[mask==0] = 0
                batch.append(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
                owners.append((k,h))
            seen += 1
            if len(batch)>=64:
                flush()
        i += 1
    cap.release()
    flush()
    del net
    if seen != len(wanted):
        raise ValueError(f'Video/frame mismatch: read {seen}/{len(wanted)} requested frames')
    features = features.reshape(len(features),-1)
    np.savez_compressed(cache,features=features.astype(np.float16),signature=json.dumps(signature))
    print(f'{sid}: image features ready ({len(features)} frames)',flush=True)
    return features


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1530,96,1),nn.GELU(),nn.Dropout(.1),
                                 nn.Conv1d(96,96,5,padding=2),nn.GELU(),nn.Conv1d(96,29,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,base):
        return (base + self.net(x.T[None])[0].T).log_softmax(-1)


def greedy(lp):
    ix = lp.argmax(-1).tolist()
    return ''.join(S.SYMS[c] for j,c in enumerate(ix) if c not in (0,28) and (not j or c != ix[j-1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=150)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',type=Path,default=Path('results/nextgen/visual_fusion.json'))
    a = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    ROOT.mkdir(parents=True,exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'mps':
        torch.mps.set_per_process_memory_fraction(.35)
    sessions = V.load_data(Path('.cache/ctc_v4/desk_data.json'))
    config = dict(steps=a.steps,seed=a.seed,backbone='ResNet18 ImageNet1K v1, frozen',crop=96,
                  evaluation='3 whole-session holdouts; phrase windows; greedy decoding; development only',
                  modes=['geometry','rgb'],not_deployed=True)
    (ROOT/'config.json').write_text(json.dumps(config,indent=2))
    t0 = time.time()
    examples = {}
    models = [V.V3/'runs'/'hwtmix'/f'seed{s}'/'all.pt' for s in range(3)]
    for sid,data in sessions.items():
        rgb = image_features(sid,device)
        st = S.load_session(sid)
        with torch.inference_mode():
            geo = S.featurize(torch.from_numpy(st.A)[None],torch.from_numpy(st.M)[None],False)[0,::2].numpy()
        base = V.ensemble_cont(ROOT,models,sid)
        if len(rgb)!=len(base) or len(geo)!=len(base):
            raise ValueError('Image/geometry/posterior clocks differ')
        times = st.t[::2]
        examples[sid] = []
        for lo,hi,text in data['windows']:
            i,j = np.searchsorted(times,[lo,hi])
            if j<=i:
                raise ValueError('Empty phrase window')
            examples[sid].append(dict(x=np.concatenate([geo[i:j],rgb[i:j]],1),
                                      base=np.array(base[i:j],np.float32),text=text))
    results = {}
    for held in sessions:
        train = [e for sid,es in examples.items() if sid!=held for e in es]
        mu = np.concatenate([e['x'] for e in train]).mean(0)
        sd = np.concatenate([e['x'] for e in train]).std(0).clip(.05)
        results[held] = {}
        for mode in config['modes']:
            torch.manual_seed(a.seed)
            rng = np.random.default_rng(a.seed)
            model = Residual()
            opt = torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
            batches = []
            for e in train:
                x = (e['x']-mu)/sd
                if mode=='geometry':
                    x[:,506:] = 0
                batches.append((torch.tensor(x),torch.tensor(e['base']),torch.tensor(S.text_syms(e['text']),dtype=torch.long)))
            model.train()
            for step in range(a.steps):
                x,base,target = batches[rng.integers(len(batches))]
                lp = model(x,base)
                loss = nn.functional.ctc_loss(lp[:,None],target,[len(lp)],[len(target)])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite fusion training loss')
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
            model.eval()
            rows = []
            for e in examples[held]:
                x = (e['x']-mu)/sd
                if mode=='geometry':
                    x[:,506:]=0
                with torch.inference_mode():
                    pred = greedy(model(torch.tensor(x),torch.tensor(e['base'])))
                    shuffled = x.copy()
                    shuffled[:,506:] = shuffled[rng.permutation(len(x)),506:]
                    shyp = greedy(model(torch.tensor(shuffled),torch.tensor(e['base'])))
                ref=e['text']
                basehyp = greedy(torch.tensor(e['base']))
                rows.append(dict(chars=len(ref),words=len(ref.split()),baseline_ce=edits(ref,basehyp),
                                 ce=edits(ref,pred),we=edits(ref.split(),pred.split()),
                                 shuffled_ce=edits(ref,shyp)))
            nc = sum(r['chars'] for r in rows)
            results[held][mode] = dict(cer=sum(r['ce'] for r in rows)/nc,
                baseline_cer=sum(r['baseline_ce'] for r in rows)/nc,
                shuffled_cer=sum(r['shuffled_ce'] for r in rows)/nc,rows=rows)
            torch.save(dict(state=model.state_dict(),mean=mu,std=sd,heldout=held,config=config),ROOT/f'{held}_{mode}_s{a.seed}.pt')
            print(f'{held} {mode}: CER {results[held][mode]["cer"]:.3f}',flush=True)
    report=dict(config=config,seconds=time.time()-t0,results=results)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2))
    print(f'Pilot complete -> {a.out}',flush=True)


if __name__=='__main__':
    main()
