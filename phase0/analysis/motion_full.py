"""Local frozen-v4 decoder comparison for session-held-out motion residuals."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import numpy as np
import torch
from phase0.analysis import seqctc as S,seqctc2 as S2,ctcv4 as V
from phase0.analysis.motion_residual import MotionResidual,extract,ROOT
L=Path('.cache/llmdec');OUT=Path('.cache/nextgen/motion_full')


def select_pool(P,run):
    from phase0.analysis import v4_suggest as VS
    scores=VS.pool_scores(P,run)
    rank=P['gen_rank'].get(('8b',run['prompt']),np.full(len(P['texts']),99))
    eligible=P['base'] | (rank<run['K']) | P['isfz']
    if any(not np.isfinite(P['lm'][m][eligible]).all() for m in ['8b','p05b']):raise ValueError('Unscored eligible candidates')
    if not np.isfinite(scores).any():raise ValueError('No eligible candidates')
    return P['texts'][int(np.argmax(scores))]


def names():
    return [(sid,mode,'s_motionfull_'+mode+'_'+sid) for sid in V.load_data(Path('.cache/ctc_v4/desk_data.json')) for mode in ['baseline','geometry','flow']]


def prepare():
    from phase0.analysis import autocorrect as AC,llmdec as LD
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    config=dict(variants=['geometry','flow'],blend=.5,boundaries='unchanged session-held-out baseline automatic segments',selection='frozen v4, no retuning',development=True)
    (OUT/'config.json').write_text(json.dumps(config,indent=2));loaded_qwen=False
    for sid,mode,name in names():
        if (L/'sets'/f'{name}.json').exists():continue
        source='s_zz-v4loso-'+sid;data=json.loads((L/'sets'/f'{source}.json').read_text())
        if mode=='baseline':
            for folder,suffix in [('sets','.json'),('sets','_lp.npz'),('llm/8b','.json'),('llm/p05b','.json'),('fuzzy','.json')]:
                shutil.copyfile(L/folder/(source+suffix),L/folder/(name+suffix))
            print('cloned baseline',name,flush=True)
            continue
        full=np.load(f'.cache/decipher/out/zz-v4loso-{sid}_lp.npz');oldz=np.load(L/'sets'/f'{source}_lp.npz')
        st=S.load_session(sid);geo=S.featurize(torch.from_numpy(st.A)[None],torch.from_numpy(st.M)[None],False)[0,::2]
        base=torch.tensor(full['zs'],dtype=torch.float32);base=base.log_softmax(-1);flow=torch.tensor(extract(sid))
        if not len(base)==len(geo)==len(flow):raise ValueError('Clock mismatch')
        preds=[]
        for seed in range(3):
            ck=torch.load(ROOT/f'{sid}_{mode}_s{seed}.pt',map_location='cpu',weights_only=False)
            if ck['heldout']!=sid:raise ValueError('Residual trained on test session')
            meta=json.loads((Path('.cache/ctc_v4/runs/hwtmix')/f'seed{seed}'/f'loso_{sid}'/'meta.json').read_text())
            if sid not in meta['excluded']:raise ValueError('Baseline trained on test session')
            model=MotionResidual();model.load_state_dict(ck['state']);model.eval()
            with torch.inference_mode():preds.append(model(geo,flow,base,mode=='flow').numpy())
        lp=np.mean(preds,0);lp-=np.logaddexp.reduce(lp,axis=1,keepdims=True)
        lp=(lp+full['desk'])/2;lp-=np.logaddexp.reduce(lp,axis=1,keepdims=True)
        np.savez_compressed(OUT/f'{name}_full.npz',zs=full['zs'],desk=lp,times=full['times'])
        arr={};items=[]
        if not loaded_qwen:
            S2._SC['qwen']=AC.NLM('.cache/decipher/qwen2.5-0.5b',device='cpu');loaded_qwen=True
        for original,(lo,hi) in zip(data['items'],data['meta']['segments']):
            it=copy.deepcopy(original);part=lp[lo:hi];ident=it['id'];arr[ident+'__zs']=oldz[ident+'__zs'];arr[ident+'__desk']=part
            nb=LD.beam_nbest(part,S2.charlm(),LD.CHAR['alpha'],LD.CHAR['beta'],LD.CHAR['beam'],LD.CHAR['topn'])
            q=LD.norm(S2.run_decoder(data['qwen'],part))
            it['c']['desk']=dict(char=nb,greedy=LD.norm(S.greedy(part)),qwen=q);it.pop('ref',None);items.append(it)
        data['items']=items;data['meta']['experiment']=config;data['meta'].pop('lines',None)
        np.savez_compressed(L/'sets'/f'{name}_lp.npz',**arr);(L/'sets'/f'{name}.json').write_text(json.dumps(data,indent=2))
        print('prepared',name,len(items),flush=True)


def run():
    env=dict(os.environ,HF_HUB_OFFLINE='1',LLMDEC_MEM_GB='7',LLMDEC_FUZZY_TAG='')
    py='.venv/bin/python';mpy='.cache/personal_llm/venv/bin/python'
    def cached(repo):
        root=Path.home()/'.cache/huggingface/hub'/('models--'+repo.replace('/','--'))
        path=root/'snapshots'/(root/'refs/main').read_text().strip()
        if not (path/'config.json').exists():raise ValueError('Missing cached model')
        return str(path)
    base8=cached('mlx-community/Qwen3-8B-4bit');base05=cached('Qwen/Qwen2.5-0.5B')
    def call(args,gpu=False):
        cmd=([py,'-m','phase0.tools.gpulock','--'] if gpu else [])+args
        subprocess.run(cmd,env=env,check=True)
    for sid,mode,name in names():
        receipt=OUT/f'{name}_done.json'
        if receipt.exists():continue
        call([mpy,'-m','phase0.analysis.llmdec_mlx','--set',name,'--llm','8b','--base',base8,'--ctx-prompts','--prompts','p7','--gks','ens'],True)
        call([py,'-m','phase0.analysis.llmdec_fuzzy',name])
        from phase0.analysis import wordbeam_v4 as W
        wb=W.run_set(name,configs=('small',),procs=2)
        f=L/'fuzzy'/f'{name}.json';r=json.loads(f.read_text())
        for ident,groups in r['items'].items():
            for g in groups:groups[g]=list(dict.fromkeys(groups[g]+wb[ident]['small']))
        r['wordbeam']=wb;f.write_text(json.dumps(r))
        call([mpy,'-m','phase0.analysis.llmdec_mlx','--set',name,'--llm','8b','--base',base8,'--lm-only','--pool-from','8b'],True)
        call([mpy,'-m','phase0.analysis.llmdec_mlx','--set',name,'--llm','p05b','--base',base05,'--adapter','.cache/personal_llm/adapters/q25_05b_it1200','--lm-only','--pool-from','8b'],True)
        receipt.write_text(json.dumps(dict(set=name,status='complete')))
        print('FULL COMPLETE',name,flush=True)


def evaluate():
    from phase0.analysis import llmdec as LD,v4_suggest as VS,decipher as D,swipe_common as C
    os.environ['LLMDEC_FUZZY_TAG']='';LD._FZ.clear();cfg=json.loads(Path('results/llmdec/frozen_config_v4.json').read_text());run=cfg['run'];results={}
    for sid in V.load_data(Path('.cache/ctc_v4/desk_data.json')):
        lines=[w[2] for w in V.load_data(Path('.cache/ctc_v4/desk_data.json'))[sid]['windows']];results[sid]={}
        variants={}
        for mode,name in [(m,'s_motionfull_'+m+'_'+sid) for m in ['baseline','geometry','flow']]:
            _,pools,_=LD.build_pools(name,('8b','p05b'));hyps=[];variants[mode]=(pools,np.load(L/'sets'/f'{name}_lp.npz'))
            for p in pools:
                hyps.append(select_pool(p['P'][run['groups']],run))
            s=D.score_variant(lines,hyps);results[sid][mode]=dict(errors=int(s['wed'].sum()),words=int(s['wn'].sum()),char_errors=int(s['ced'].sum()),chars=int(s['cn'].sum()),hyps=hyps,by_line=s['hyp_by_line'])
        for mode in ['geometry','flow']:
            for pool_mode,camera_mode in [('baseline',mode),(mode,'baseline')]:
                hyps=[]
                for p in variants[pool_mode][0]:
                    P=dict(p['P']['ens']);z=variants[camera_mode][1]
                    P['ctc']=C.ctc_ens([z[f'{p["id"]}__{g}'] for g in P['groups']],P['texts'])
                    hyps.append(select_pool(P,run))
                s=D.score_variant(lines,hyps);results[sid][f'pool_{pool_mode}__camera_{camera_mode}']=dict(errors=int(s['wed'].sum()),words=int(s['wn'].sum()),hyps=hyps,by_line=s['hyp_by_line'])
    summary={m:dict(errors=sum(r[m]['errors'] for r in results.values()),words=sum(r[m]['words'] for r in results.values())) for m in next(iter(results.values()))}
    Path('results/nextgen/motion_full.json').write_text(json.dumps(dict(results=results,summary=summary,note='Frozen decoder; full candidate regeneration; development only; baseline held out entire camera session; not the earlier 25-error mixed-CV benchmark.'),indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','eval']);a=p.parse_args();{'prepare':prepare,'run':run,'eval':evaluate}[a.stage]()
