# computer-vision-test

Webcam hand-gesture control for macOS: bind custom hand gestures to actions on the computer.

**Status:** research / scaffolding. Nothing implemented yet.

## Goal

1. Track hands from the built-in webcam in real time.
2. Recognize a small set of *custom*, user-defined gestures (static poses first, dynamic ones later).
3. Bind each gesture to a concrete action on macOS — keystroke, media control, app launch, window management, arbitrary script.

## Layout (planned)

```
capture/     camera + frame pipeline
tracking/    hand landmark extraction
gestures/    gesture definition, recording, classification
actions/     macOS action dispatch (CGEvent / Shortcuts / scripts)
config/      gesture -> action bindings
```

## Research

Full landscape + safety research lives in Notion: "Research" page.

## License

Private / unlicensed for now.
