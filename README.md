# keyboardless-keyboard

Type on a bare desk and let one phone camera read it. No keyboard, no headset, nothing on your hands.

This is a research project, written up here: **[I'm building a keyboardless keyboard](https://alidadi.me/notes/keyboardless-keyboard/)**.

## Where it stands

- An iPhone above the desk streams its ultra-wide camera to a Mac.
- A small sequence model reads the hand movements and outputs character probabilities.
- A decoder with a local language model turns those into words.

On sentences it had never seen, typed on a bare desk, it got **73% and 82% of words right** in two blind tests (6 and 5 sentences, so treat those as rough). It has been trained and tested on one person so far.

Read the write-up for the full story. The short version of what the experiments found:

- Reading the whole hand movement as a sequence (CTC) works far better than detecting individual taps.
- Most of the remaining errors are words the camera never proposes, so a bigger decoder can't fix them.
- A straight-down camera is the worst angle for seeing whether a finger touched the desk, and almost every published system films hands at about 45 degrees. But tilting the camera doesn't fix it either: on 30 people's real typing, re-projected to 45 degrees, fewer fingers get hidden and the text gets no better ([results/angle45](results/angle45/SUMMARY.md)).
- The one geometry that looks promising in simulation is a small mirror on the desk, which gives the same top-down camera a second, low view and lets it triangulate fingertip height ([results/mirror](results/mirror/SUMMARY.md)). It is untested on real footage.
- Several promising ideas (a pretrained contact model, a better hand tracker, image crops) made no difference once tested against proper controls.

## How it works

```
iPhone (WideCam app) --Wi-Fi MJPEG, 1280x720 @ 60 fps--> Mac
  -> MediaPipe hand landmarks (21 points per hand, per frame)
  -> CTC model: convolutions + small Transformer, ~1.2M parameters, ensemble of 6
  -> candidate sentences (dictionary beam search + optional local LLM)
  -> each candidate re-scored against the hand-motion evidence
  -> text
```

Training uses a real keyboard as the teacher: you type normally while a keylogger records every key, which labels the video for free. Then the model is applied to the same movements on a bare desk. Pretraining on [How We Type](https://zenodo.org/records/4034268), a public motion-capture dataset of 30 people typing, was one of the biggest gains.

## Repository layout

| Path | What's in it |
|---|---|
| `widecam/` | WideCam, the iPhone app that streams the ultra-wide camera over Wi-Fi, plus its [wire protocol](widecam/PROTOCOL.md) |
| `phase0/capture/` | Recorder: video + frame timestamps + keylogger, and the network video source |
| `phase0/analysis/` | Landmark extraction, the CTC model, decoders, the live demo, and every experiment |
| `phase0/tests/` | Tests |
| `results/*/SUMMARY.md` | One write-up per experiment: numbers and verdict first, then method and caveats |
| `docs/` | Design reviews, the decoding spec, and research surveys |
| `REPORT.md` | The running lab notebook from the first weeks |
| `mat/` | A printable A4 keyboard mat (an experiment, not required) |

## What is not included

- **Recordings.** The videos and keystroke logs stay private, because they contain real typing.
- **Trained weights.** They were trained on my recordings and on How We Type, whose licence is non-commercial.
- **The personal vocabulary.** The decoder can use a vocabulary built from your own writing; that data never leaves your machine and is gitignored here.

So cloning this gives you the full pipeline, but you'll need to record your own training data.

## Getting started

Requirements: macOS on Apple Silicon, an iPhone with an ultra-wide camera, Python 3.12, and Xcode with [XcodeGen](https://github.com/yonaskolb/XcodeGen) to build the app.

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Build and install WideCam (copy local.env.example to local.env and fill in your team id)
widecam/ios/build.sh && widecam/ios/install.sh

# 2. Record: phone above your hands, WideCam open. Type normally on a real keyboard.
#    The terminal needs Input Monitoring permission for the keylogger.
.venv/bin/python -m phase0.capture.recorder --condition kbd --backend net --camera auto \
  --width 1280 --height 720 --fps 60 --seconds 600

# 3. Extract hand landmarks
PYTHONPATH=. .venv/bin/python -m phase0.analysis.extract_landmarks data/sessions/<session-id>

# 4. Once you have trained models: the live demo (raw letters as you type, a phrase after each pause)
PYTHONPATH=. .venv/bin/python -m phase0.analysis.live_demo
```

Training and full decoding are described in `REPORT.md`, `results/ctc_v3/SUMMARY.md` and `results/v4/SUMMARY.md`. They are research scripts, not a polished CLI, and some expect a second machine for heavy compute.

## Open problems

This is one person's project, and the honest state is "promising, not solved". If you work on camera typing, gesture input or hand tracking, these are the places where help would matter most:

1. **Seeing contact from one camera.** From above, a press and a hover look almost the same. The mirror idea in `results/mirror/` is the best lead I have and needs a real recording.
2. **More than one typist.** Everything here is trained and tested on my hands. Nobody knows how it transfers to someone else, and that is the question that decides whether this can be a product.
3. **Honest evaluation.** The blind tests are 11 sentences in total. A shared, held-out set of desk typing recorded by several people would be worth more than any model change.
4. **Live decoding.** The accurate decoder takes about half a minute per phrase; the fast one loses about 20 points of word accuracy. `results/beam/` shows how to drop the slowest stage.
5. **Literal input.** Capitals, punctuation, digits and code identifiers are not handled yet.

Every experiment, including the ones that failed, has a write-up under `results/*/SUMMARY.md` with numbers first and caveats at the end. Please read the relevant one before repeating something: most dead ends here are documented.

Issues, pull requests and comparisons from your own setup are welcome. If you record sessions with WideCam, an issue with your numbers and camera placement is useful even without code.

## Security

Reading keystrokes from video is a known attack, and this is a narrow version of it. It needs a camera directly above the hands and a model trained on that person's labelled typing. Most of its accuracy comes from the dictionary and language model, so it does poorly on random strings such as passwords. Everything runs locally.

## Third-party data

Some experiments use public datasets under non-commercial licences: [How We Type](https://zenodo.org/records/4034268) and [EgoPressure](https://github.com/eth-siplab/EgoPressure) (CC BY-NC-SA). They are not redistributed here. Nothing trained on them should be used commercially.

## License

[MIT](LICENSE). The licence covers the code; it does not extend to the third-party datasets above.
