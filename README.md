# ARGUS

**A**daptive **R**ecognition & **G**esture **U**nderstanding **S**ystem — a hands-free
cursor driven by a webcam. CPU-only, fully local, no cloud inference.

Point with an extended index finger to move the cursor. Pinch thumb-to-index to click,
thumb-to-middle to right-click, hold the pinch to drag. Relax your hand and the cursor
parks — then move your hand back to a comfortable spot and point again, exactly like
lifting a mouse off a pad.

---

## The interaction model, and why it is not what you would first guess

**The clutch is a pose, not a pinch.** A common design is "pinch to engage the cursor,
release to disengage" — but that collides head-on with "pinch to click": one signal
cannot mean two things. Here, *index extended* means the cursor is live. Relaxing your
hand parks it. That frees pinch to mean exactly one thing, and holding a finger out is
far less tiring than holding a pinch.

**Motion is relative, not absolute.** The cursor moves by the *displacement* of your
hand, like a mouse — not to a point your finger aims at. Ray-casting from a fingertip
needs real 3D; a single webcam's depth estimate is weak and the camera is not at your
eye. Relative motion is also what makes clutching meaningful: you can re-centre your hand
whenever you like, so the reachable screen area is unbounded and **your hand can stay low
near the desk**. No gorilla arm.

**The palm drives the cursor, not the fingertip.** When you pinch, your fingertip moves —
so a fingertip-driven cursor slides off the target at the exact moment you commit. That
click-induced drift is the largest source of error in mid-air pointing. The palm centroid
barely moves when fingers flex, so the problem is removed at the source rather than
filtered away afterwards. A cursor freeze on pinch *onset* catches the remainder.

**Every distance is measured in hand-widths.** Thresholds are divided by your own
wrist-to-knuckle span, so a pinch reads the same whether your hand is 30 cm or 60 cm from
the camera. One set of thresholds, any distance.

**Gain rises with speed.** A single fixed gain cannot both hit a 16 px close button and
cross a 4480 px dual-monitor desktop. Slow movements get precision; fast ones cover
ground, blended with a smoothstep so you never feel the change.

---

## Install

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.venv\Scripts\python.exe -m argus models pull
```

## Use it

```bash
python -m argus mouse
```

It starts **disarmed**: the full pipeline runs and the HUD shows exactly what it *would*
do, but nothing touches your cursor until you press **F9**.

| Key | Action |
|---|---|
| **F9** | arm / disarm cursor control |
| **Esc** (hold) | emergency disarm — works from any window |
| **F10** | re-centre the cursor on the primary display |
| `q` | quit |

Fit the gesture thresholds to your own hand — worth doing once, since thumb proportions
vary enough to matter:

```bash
python -m argus calibrate --write
python -m argus mouse -c configs/calibrated.yaml
```

Other commands: `cameras`, `preview`, `hands`, `screens`, `models`, `config`,
`bench capture`, `bench hands`.

---

## Safety

Driving the real cursor from a perception pipeline is inherently risky, so the safeguards
are structural rather than advisory:

- **Disarmed by default.** Nothing reaches the OS until you explicitly arm it.
- **The panic key is read from the hardware**, via `GetAsyncKeyState`, not from the
  preview window. This matters: the moment the system clicks something, the preview loses
  focus and a window-level key handler would stop working — precisely when you most need
  to stop it.
- **Buttons are always released** on every exit path, so quitting mid-drag can never leave
  the desktop with a stuck mouse button. The same applies if identity is lost mid-drag.
- **Input-side plausibility limits.** A hand cannot move a third of its own width in 33 ms;
  anything that claims otherwise is a landmark glitch and is rejected *before* gain
  amplifies it.
- **Identity gating** (optional, `security.require_identity`) runs on an authenticated
  *session*, not a per-frame face match — recognition at 30 fps is unaffordable on this
  CPU, and gating motion frame-by-frame would freeze the cursor every time you glanced at
  your keyboard. Destructive actions additionally require a *fresh* match.

---

## Measured on the target machine

Intel i5-1035G1 (4C/8T, 1.0 GHz), 8 GB RAM, **CPU-only** — the 2 GB GPU runs a driver
(CUDA 11.4) too old for current ONNX Runtime and Torch CUDA builds. Logitech Brio 100 at
1280×720.

| Stage | Mean | Budget |
|---|---|---|
| capture latency | 1.95 ms | — |
| preprocess | 0.30 ms | — |
| hand landmarks | **12.06 ms** | 33.3 |
| gestures | 0.16 ms | — |
| pointer | 0.03 ms | — |
| dispatch | 0.00 ms | — |
| HUD render | 4.85 ms | — |
| **total** | **17.4 ms** | 33.3 → **48 % headroom** |

Sustained **28.2 fps** end to end, 0 % dropped frames.

Three findings drove that, all measured rather than assumed:

1. **Backend choice was worth 6× throughput.** OpenCV's DirectShow backend negotiates
   uncompressed YUY2 at 720p, which this camera caps at 5 fps;
   `set(CAP_PROP_FOURCC, MJPG)` returns `True` and changes nothing. MSMF refuses the
   property outright yet delivers 30 fps. No static rule predicts this, so ARGUS measures
   each backend and keeps whichever actually delivers, caching the verdict.
2. **`OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` cut camera open time from 24.6 s to
   0.6 s** with identical throughput.
3. **Tracking one hand instead of two halved landmark cost** (28.4 → 12.1 ms). Each extra
   hand is a full model pass.

---

## Architecture

```
argus/
  config.py          typed, layered, strict configuration
  metrics.py         allocation-free rolling timers
  capture/           camera discovery, backend negotiation, threaded reader
  hands/             MediaPipe Tasks landmarks, hand geometry, calibration
  gestures/fsm.py    Schmitt triggers, debouncing, click/drag discrimination
  control/
    filters.py       One Euro filter, velocity gain curve
    screens.py       virtual desktop, per-monitor DPI, monitor enumeration
    injector.py      SendInput via ctypes
    pointer.py       clutch, relative mapping, gain, click freeze
    dispatcher.py    the only code allowed to affect the machine
    hotkeys.py       global key state
  app.py             the integrated runtime
```

### Two Windows details that silently break naive implementations

Both are real on the development machine, not hypothetical:

**The virtual desktop origin is not (0, 0).** The primary display's top-left is the
origin, so a monitor placed to its *left* occupies **negative X**. Here the external
2560×1440 sits at `x = -2560`. Code assuming `0 <= x < screen_width` cannot reach it at
all. Absolute coordinates are computed over the whole virtual desktop, with `width - 1` in
the denominator — using `width` makes the rightmost column unreachable, so the maximise
button can never be clicked.

**DPI awareness must be declared before any metric is read.** The laptop panel here runs
at 150 % scaling; without `PER_MONITOR_AWARE_V2` every coordinate is wrong by 1.5× and the
error grows with distance from the origin.

---

## Testing

```bash
.venv\Scripts\python.exe -m pytest        # 132 tests, no hardware required
```

Gestures and cursor maths are tested against **synthetic hands** with exactly specified
geometry, so a test can assert something precise — "the same pinch at four times the
apparent size must produce an identical event sequence" — with no ambiguity. Coverage
includes hysteresis (a signal oscillating inside the deadband must produce zero flips),
click-versus-drag discrimination, depth invariance, negative-coordinate mapping,
click-drift, stuck-button prevention, and the full landmarks→gestures→pointer→dispatch
chain.

---

## Status

- [x] **Phase 0–1** Foundation, camera discovery and selection, threaded capture
- [x] **Phase 2** Hand landmark engine
- [x] **Phase 3** Gesture state machine
- [x] **Phase 4** Cursor core: filtering, gain, clutch, injection
- [x] **Phase 5** Click, right-click, double-click, drag
- [ ] **Phase 6** Face recognition and identity gating *(interfaces in place, gate
      currently permissive)*
- [ ] **Phase 7** Scroll, two-hand zoom, voice

Face gating is wired end-to-end (`ActionDispatcher` consumes an `IdentityStatus`, and
losing identity mid-drag releases the button) but the recogniser itself is not built yet,
so `security.require_identity` defaults to `false`. SCRFD and ArcFace weights are already
fetched and checksum-pinned.

## Privacy

Face embeddings are biometric data. `data/` and `models/` are git-ignored: enrolled
identities never leave this machine and are never committed.

## Credits

MediaPipe Hands (Zhang et al.) · SCRFD and ArcFace, InsightFace (Guo et al., Deng et al.) ·
One Euro filter (Casiez, Roussel & Vogel, CHI 2012)

## License

MIT
