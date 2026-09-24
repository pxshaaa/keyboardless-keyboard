"""Summarize the resource acquisition and local foundation experiments."""
import json
from pathlib import Path


def main():
    root=Path('results/nextgen')
    pixels=[]
    for seed in [1,2,3]:
        result=json.loads((root/f'typing_pixels_s{seed}.json').read_text())
        row={'seed':seed}
        for mode in ['geometry','pixels']:
            records=[r for session in result['results'].values() for r in session[mode]]
            for metric in ['ce','baseline_ce','shuffled_ce','static_ce']:
                row[f'{mode}_{metric}']=sum(r[metric] for r in records)/sum(r['chars'] for r in records)
        pixels.append(row)
    means={k:sum(r[k] for r in pixels)/len(pixels) for k in pixels[0] if k!='seed'}
    streams=[]
    for look in [0,30,120]:
        result=json.loads((root/f'stream_20260913-125733-desk_l{look}.json').read_text())
        rows=result['rows']
        streams.append(dict(lookahead_frames=look,cer=sum(r['ce'] for r in rows)/sum(r['chars'] for r in rows),
            compute_seconds=result['compute_seconds'],video_seconds=result['video_seconds']))
    offline=json.loads((root/'stream_offline_reference.json').read_text())
    students=[]
    for name in ['causal_student','causal_student_balanced']:
        data=json.loads((root/f'{name}.json').read_text())
        rows=[r for s in data['results'].values() for r in s['rows']]
        students.append(dict(name=name,cer=sum(r['ce'] for r in rows)/sum(r['chars'] for r in rows),
            inference_seconds=sum(s['inference_seconds'] for s in data['results'].values())))
    report=dict(pixels=pixels,pixel_mean=means,streaming=streams,offline_reference=offline,
        causal_students=students,
        deployment_changed=False,evaluation='development recordings, not a fresh blind test',
        public_sample_trained=False)
    (root/'foundation_summary.json').write_text(json.dumps(report,indent=2))
    pixel_rows='\n'.join(f'| {label} | {means[key]:.3%} |' for label,key in [
        ('Unchanged three-seed base','pixels_baseline_ce'),('Geometry residual','geometry_ce'),
        ('Trainable pixels','pixels_ce'),('Pixels shuffled in time','pixels_shuffled_ce'),('Static average image','pixels_static_ce')])
    stream_rows='\n'.join(f'| {r["lookahead_frames"]/60:.1f} s | {r["cer"]:.3%} | {r["compute_seconds"]:.2f} s |' for r in streams)
    student_text='; '.join(f'{r["name"]}: {r["cer"]:.2%} CER' for r in students)
    text=f'''# Foundation work — 14 September 2026

Implemented and tested locally. No SSH, no commits, and no production decoder or model replacement.

## Trainable visual model

`typing_pixels.py` learns a two-layer image encoder from masked RGB hand crops plus frame differences, with a temporal residual over the existing camera probabilities and geometry. It pretrains on 81 keyboard windows from two recordings, then trains using the other two desk sessions and keyboard windows for each whole-session holdout. All 31 desk phrases are evaluated. This is a small 24-pixel-per-hand pilot, not a full-resolution fingertip model.

An initial unregularized six-run pilot was harmful; its results remain in `typing_pixels_s0.json`. The revised recipe uses image feature normalization and a KL penalty against damaging the base predictions, with 250 keyboard-pretraining and 250 adaptation steps. Seeds 1, 2, and 3 are separate replications of that recipe: 18 model fits, plus the initial six. The revision was made after inspecting development results. These are not new blind tests.

| Condition (three-seed mean, greedy character error) | CER |
|---|---|
{pixel_rows}

Static and shuffled-image controls perform similarly. The experiment does not demonstrate recovery of useful additional finger motion, so it is not promoted. Character errors here are before language-model decoding and must not be compared directly with previous word accuracy. Crop caches have source hashes and a recipe identifier. The first exploratory checkpoint lacks the later normalization layer; use its original report, not the current class, for historical inspection.

## Bounded-delay replay

`streaming_replay.py` uses a six-second history, half-second output blocks, and adjustable lookahead. Normalization sees only the available window, including the declared lookahead. It uses one existing zero-shot model. Output is continuous and does not use phrase boundaries. Phrase boundaries are used only for the following retrospective score on the last desk recording.

| Lookahead at 60 fps | Greedy CER | Total inference time |
|---|---|---|
{stream_rows}

The same model's offline reference is {offline['cer']:.3%} CER. Timings exclude landmark extraction and camera capture. Algorithmic delay includes up to another half-second for output batching. This establishes a working replay path and quantifies the loss from truncating future context; it does not establish live typing accuracy. A prefix-invariance test verifies that unavailable future landmarks cannot change committed output.

`causal_student.py` additionally trains a genuinely causal GRU from the teacher's frame probabilities, with calibration fixed at the first valid two-hand observation. Two loss variants were evaluated on three whole-session holdouts: {student_text}. Both are exploratory, use only 400 optimization steps per holdout, and fail to match the teacher. The second variant downweights blank frames after the first mostly collapsed to blanks. Neither is deployed. Separate tests verify prefix-invariant features and equality between stateful chunked and full-sequence inference.

## Task context and corrections

`task_context.py` builds vocabulary from explicitly named UTF-8 working files (maximum 1 MB each), records hashes, and reports unsupported symbols without silently changing them. `context_decode.py` inserts camera-supported alternatives and ranks them with CTC. It keeps the original prediction and does not automatically accept corrections. The full command path was exercised on an existing posterior window using the project README as context; this is a plumbing check, not an accuracy result. Neither screen access nor broad personal-data collection is enabled.

`phase0/tools/transcript_review.html` opens directly in a browser, imports replay JSON, shows partial text, accepts a manual correction, and exports original and corrected text separately. Browser tests cover import, edit, download, invalid input, and mobile layout. It is a recorded replay interface, not a live camera frontend. Edits do not enter training automatically.

The previously prepared calibration curriculum remains in `.cache/nextgen/calibration/part1.txt` through `part4.txt`. Public data cannot replace new recordings of this user's hands and camera placement. The ingest review gate still requires confirmed labels.

## Online resources actually checked

- [How We Type, author-hosted dataset](https://zenodo.org/records/4034268): verified archive metadata and downloaded the README. Implemented exact HTTP-range access to the ZIP64 central directory, selected the smallest real video, and fetched only that member (219.5 MB decompressed) instead of the 29.3 GB archive. ZIP CRC and SHA-256 provenance are recorded in `.cache/nextgen/public_video/`. The sample shows hands and keyboard with motion-capture markers; its encoded rate is 29.97 fps. The source describes 120 fps capture. Its part-two video/keylog offset is not established, so it has **not** been used as labelled training data. The source specifies non-commercial use.
- [TypingHands26 / TypeNet paper](https://openaccess.thecvf.com/content/WACV2022/html/Maman_Typenet_Towards_Camera_Enabled_Touch_Typing_on_Flat_Surfaces_Through_WACV_2022_paper.html): directly relevant camera and flat-surface work. No working public dataset download was found in the checked author/publication pages; do not confuse it with the unrelated biometric TypeNet repository. No claim that the dataset is unavailable everywhere.
- [Keystroke Typing Videos](https://huggingface.co/datasets/andrewt28/keystroke-typing-videos): retrieved the dataset card, which describes Reuters sentences with keylogs and lists AFL-3.0. The project already tested transfer from this dataset, so it was not treated as a new discovery or downloaded again.
- [Touchless Typing](https://sites.google.com/iiitd.ac.in/touchless-typing/): excluded after checking the source; it records head movements, not finger typing.
- [CTC word spotting](https://arxiv.org/abs/2406.07096): supports the context-candidate direction. The local code is a prototype, not a reproduction of the published speech benchmark.
- [Emformer streaming CTC](https://arxiv.org/abs/2203.15613): supports a future causal/chunk-trained model. The current replay wraps the existing model and does not implement Emformer or claim its speech results.

## Reproduce

Run from the repository root:

```sh
.venv/bin/python -m phase0.analysis.fetch_typing_sample
.venv/bin/python -m phase0.tools.gpulock --wait 1 -- .venv/bin/python -m phase0.analysis.typing_pixels --seed 1 --steps 250 --pretrain 250
.venv/bin/python -m phase0.analysis.streaming_replay 20260913-125733-desk --lookahead 30
.venv/bin/python -m phase0.analysis.task_context README.md --out /tmp/task-context-new.json
.venv/bin/python -m pytest phase0/tests/test_nextgen.py phase0/tests/test_nextgen_workflows.py -q
```

Context and suggestion exports refuse overwrites. Training commands overwrite their own seed outputs, so choose another seed/output directory to preserve a run. To use the correction page, load a `results/nextgen/stream_*.json` file through its picker.

## Remaining limits

No 100% accuracy has been achieved. The production pipeline remains the prior validated version. Fresh calibration and a truly untouched test require a new recording; external video still needs trustworthy timestamp alignment. A larger motion encoder, an accurate causal student, automatic live capture integration, and measured task-context accuracy remain further work. Twenty focused Python tests passed (the original eight plus twelve workflow checks), along with the browser smoke test.
'''
    Path('docs/FOUNDATION_WORK_2026-09-14.md').write_text(text)


if __name__=='__main__':
    main()
