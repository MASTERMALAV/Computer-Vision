# ARGUS

**A**daptive **R**ecognition & **G**esture **U**nderstanding **S**ystem — a real-time,
CPU-first multimodal perception stack.

ARGUS answers three questions about the person in front of the camera, many times a second:

| Question | Signal | Status |
|---|---|---|
| **Who is this?** | Face detection → alignment → embedding → gallery match | Phase 2 |
| **What are they doing?** | 21-point hand landmarks → static + dynamic gesture recognition | Phase 3 |
| **What did they say?** | VAD → streaming ASR → (optional) speaker verification | Phase 4 |

The perception core is deliberately separate from anything that acts on the machine.
ARGUS decides *what happened*; acting on it is opt-in, gated behind operator identity,
and confirmable — see [Safety model](#safety-model).

---

## Why it is built this way

The target machine is an **Intel i5-1035G1 (4 cores / 8 threads, 1.0 GHz base), 8 GB RAM**,
with a 2 GB discrete GPU on an old driver (CUDA 11.4) that modern ONNX Runtime and PyTorch
CUDA builds no longer support. That is not a footnote — it is the central design constraint:

- **Every model is chosen to hit real-time on CPU.** No model is used because it tops a
  leaderboard; it is used because it tops a leaderboard *per millisecond on four cores*.
- **Work is done as rarely as it can be.** Detection runs every *N*th frame and tracking
  fills the gaps. Face embeddings are computed only every few frames per track, since an
  identity does not change between frames.
- **Latency is measured, not assumed.** Every stage is timed, and the age of a frame at
  draw time is on screen, because that is the lag you actually feel when you wave a hand.

If you run this on a stronger machine, `--profile accurate` moves every quality knob at once.

---

## Install

Requires Python 3.10+ (developed on 3.12).

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.venv\Scripts\python.exe -m pip install pygrabber        # Windows: camera names
```

---

## Quick start

**1. See what cameras you have, and which one ARGUS will use:**

```bash
python -m argus cameras
```

```
  Cameras available (3 found)

     [0] Integrated Camera
  -> [1] Brio 100
     [2] Integrated IR Camera   (infrared - not usable for face recognition)
```

The `->` marks the current selection. Auto-selection prefers an external colour camera
over the built-in one, and never picks a Windows Hello infrared camera.

**2. Pick a camera explicitly** — by name substring, by index, or leave it on `auto`:

```bash
python -m argus preview --camera brio
python -m argus preview --camera 0
```

To make it permanent, set `capture.camera.device` in `configs/default.yaml`.

**3. Verify capture is actually real-time:**

```bash
python -m argus bench capture --seconds 15
```

This is an acceptance gate, not a demo: it fails if sustained delivery falls more than
20 % below the requested frame rate.

### Preview controls

| Key | Action |
|---|---|
| `q` / `Esc` | quit |
| `n` | switch to the next camera, live |
| `s` | save a snapshot to `captures/` |
| `m` | toggle mirroring |
| `h` | hide the help panel |

---

## Configuration

One file, `configs/default.yaml`, mirrors the typed schema in `argus/config.py`.
Three layers, each overriding the last:

1. dataclass defaults — always complete and valid
2. the YAML file
3. `--set key=value` on the command line

```bash
python -m argus preview --set capture.camera.width=1920 --set capture.camera.fps=60
python -m argus config           # print the fully resolved result
```

**Unknown keys are a hard error.** A typo like `widht: 1920` fails at startup with the
list of valid keys, instead of silently running with defaults.

Profiles (`--profile fast|balanced|accurate`) move several coupled knobs at once — detector
size, model choice, detection stride, hand model complexity. Anything you set explicitly
still wins over the profile.

---

## Architecture

```
argus/
  config.py          typed, layered, strict configuration
  metrics.py         allocation-free rolling timers and FPS
  logsetup.py        one logging setup for every entry point
  capture/
    devices.py       camera discovery, naming, scoring, selection
    camera.py        threaded reader with latest-frame semantics
    preview.py       live preview + headless capture benchmark
  face/              Phase 2 - SCRFD detection, ArcFace embeddings, gallery
  hands/             Phase 3 - MediaPipe landmarks, gesture recognition
  audio/             Phase 4 - VAD, ASR, speaker verification
  fusion/            Phase 5 - event bus, perception state, action dispatch
  ui/overlay.py      shared drawing primitives
```

### Capture: why a thread

Reading frames on the main loop couples inference speed to capture speed. If a frame takes
40 ms to process, the next `cap.read()` returns an image the driver queued 40 ms ago, and
the lag compounds until the preview is visibly behind your hand.

The reader thread keeps **only the newest frame**. Frames produced while a consumer was busy
are dropped on purpose and *counted*, so the drop rate is visible on the HUD rather than
silently degrading interactivity.

---

## Phases

Each phase ends with a verification gate that has to pass before the next one starts.

- [x] **Phase 0 — Foundation.** Config, logging, metrics, project layout, tests.
- [x] **Phase 1 — Capture.** Camera discovery and selection by name, threaded latest-frame
      reader, negotiated-format reporting, live preview, capture benchmark.
- [ ] **Phase 2 — Face.** SCRFD detection, 5-point alignment, ArcFace embeddings, enrolment,
      gallery matching with hysteresis, identity tracking.
- [ ] **Phase 3 — Hands.** MediaPipe landmarks, pose-invariant features, static gesture
      classification, dynamic gestures (pinch, swipe), temporal smoothing.
- [ ] **Phase 4 — Voice.** Silero VAD, streaming ASR, optional speaker verification.
- [ ] **Phase 5 — Fusion.** Event bus, multimodal state, gesture → action dispatch.

---

## Safety model

Gestures that trigger real system actions are handled with deliberate friction, because a
false positive on "shut down" is expensive and a false positive on "scroll" is not:

- **Identity gating** — an action only fires for a recognised, enrolled operator.
- **Dry-run by default** — the dispatcher logs what it *would* do until explicitly armed.
- **Per-action confirmation** — destructive actions require a second, distinct signal.
- **Debounce and cooldown** — one gesture cannot fire twice in quick succession.

---

## Privacy

Face embeddings are biometric data. `data/` and `models/` are git-ignored: **enrolled
identities never leave this machine and are never committed.** Enrolment is explicit and
per-person; there is no background collection.

---

## Testing

```bash
.venv\Scripts\python.exe -m pytest                     # everything that needs no hardware
.venv\Scripts\python.exe -m pytest -m "not camera"     # skip hardware-dependent tests
```

---

## Model credits

- **SCRFD** / **ArcFace** — [InsightFace](https://github.com/deepinsight/insightface)
  (Guo et al., *Sample and Computation Redistribution for Efficient Face Detection*;
  Deng et al., *ArcFace: Additive Angular Margin Loss*)
- **MediaPipe Hands** — Zhang et al., *MediaPipe Hands: On-device Real-time Hand Tracking*
- **Silero VAD**, **faster-whisper** (CTranslate2) — Phase 4

## License

MIT
