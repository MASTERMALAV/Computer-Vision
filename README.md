# ARGUS

**A**daptive **R**ecognition & **G**esture **U**nderstanding **S**ystem - a hands-free
cursor driven by a webcam. CPU-only, fully local, no cloud inference.

Point with an extended index finger to move the cursor. Then one rule covers everything
else: **tap a fingertip with your thumb for a click, hold it to do the sustained version.**
Index tap is a left click and index hold is a drag; middle tap is a right click and middle
hold is a scroll. Relax your hand and the cursor parks - then move it back to a comfortable
spot and point again, exactly like lifting a mouse off a pad.

Actions can be gated on your face, so the cursor only answers to you.

---

## The interaction model, and why it is not what you would first guess

**The clutch is a pose, not a pinch.** A common design is "pinch to engage the cursor,
release to disengage" - but that collides head-on with "pinch to click": one signal cannot
mean two things. Here, *index extended* means the cursor is live. Relaxing your hand parks
it. That frees pinch to mean exactly one thing, and holding a finger out is far less tiring
than holding a pinch.

**Motion is relative, not absolute.** The cursor moves by the *displacement* of your hand,
like a mouse - not to a point your finger aims at. Ray-casting from a fingertip needs real
3D; a single webcam's depth estimate is weak and the camera is not at your eye. Relative
motion is also what makes clutching meaningful: you can re-centre your hand whenever you
like, so the reachable screen area is unbounded and **your hand can stay low near the
desk**. No gorilla arm.

**The palm drives the cursor, not the fingertip.** When you pinch, your fingertip moves -
so a fingertip-driven cursor slides off the target at the exact moment you commit. That
click-induced drift is the largest source of error in mid-air pointing. The palm centroid
barely moves when fingers flex, so the problem is removed at the source rather than
filtered away afterwards. A cursor freeze on pinch *onset* catches the remainder.

**Every distance is measured in hand-widths.** Thresholds are divided by your own
wrist-to-knuckle span, so a pinch reads the same whether your hand is 30 cm or 60 cm from
the camera. One set of thresholds, any distance.

**Tap versus hold, on each of two fingers.** Scrolling was first built as a second hand
pose - two fingers extended - and it was the one gesture that felt wrong. It demanded
finger precision and a pose change at the same instant as a movement. Reusing the pinch
instead gives one rule covering four actions, with no extra hand shape at all: tap for the
discrete action, hold for the sustained one. Holding thumb-to-middle and sweeping is the
same motion as grabbing a page and pulling it.

**Scrolling has momentum.** Hand travel is finite - perhaps 15 cm of comfortable vertical
range, which is a few hundred pixels of document, against a wheel that has no limit. So
movement is boosted with speed and a flick coasts after release, decoupling distance
scrolled from distance moved. A slow release stops dead, because that is when you are
positioning carefully.

**Gain rises with speed.** A single fixed gain cannot both hit a 16 px close button and
cross a 4480 px dual-monitor desktop. Slow movements get precision; fast ones cover ground,
blended with a smoothstep so you never feel the change.

---

## Install

```bash
py -3.12 -m venv .venv
```
```bash
.venv\Scripts\python.exe -m pip install -r requirements-core.txt
```
```bash
.venv\Scripts\python.exe -m argus models pull
```

## Use it

Run everything through `argus.cmd` in the project root. It invokes the project's virtualenv
directly, so it works regardless of which Python is first on `PATH` - a plain
`python -m argus` fails with `ModuleNotFoundError: No module named 'yaml'` if it picks up a
system interpreter instead of `.venv`.

**Note the leading `.\`.** PowerShell does not run programs from the current directory
without it, so `argus mouse` gives *"The term 'argus' is not recognized"* while
`.\argus mouse` works. The `.\` form is also correct in `cmd.exe`, so every example
below uses it.

```bash
.\argus mouse
```

It starts **disarmed**: the full pipeline runs and the HUD shows exactly what it *would*
do, but nothing touches your cursor until you press **F9**.

| Key | Action |
|---|---|
| **F9** | arm / disarm cursor control |
| **Esc** (hold) | emergency disarm - works from any window |
| **F10** | re-centre the cursor on the primary display |
| `q` | quit |

If you would rather not type the `.\` each time, add the project folder to your
`PATH`, or call the interpreter directly:
`.venv\Scripts\python.exe -m argus mouse`.

### Fit it to your hand

Worth doing once - thumb proportions vary enough between people that a pinch reading 0.28
on one hand reads 0.40 on another, and a threshold set between those either fires
constantly or never fires at all.

```bash
.\argus calibrate --write
```
```bash
.\argus mouse -c configs/calibrated.yaml
```

### Identity gating

Enrol your face, then let only you drive the cursor:

```bash
.\argus enroll "Your Name"
```

Enrolment walks through five head poses and only accepts frames where the face is large
enough, square enough to the camera and not cut off - a gallery built from bad crops fails
in ways that look like a broken threshold and are miserable to debug. Afterwards it reports
how well your samples separate from anyone else enrolled, and suggests a threshold measured
from *your* data rather than taken from a paper.

```bash
.\argus faces --analyse
```
```bash
.\argus mouse --set security.require_identity=true --set security.operator="Your Name"
```

`argus face` shows live detection, recognition and session state.

### Choosing a camera

```bash
.\argus cameras --pick
```

Shows each working camera live and saves your choice. Press **c** during `argus mouse` to
switch cameras without restarting.

Cameras are identified as `backend:index`, e.g. `msmf:0`, because **an index alone does not
identify a camera**: Windows' two capture backends enumerate devices independently and can
disagree. On the development machine DirectShow lists `[Integrated, Brio]` while Media
Foundation's index 0 *is* the Brio - exactly reversed. `argus cameras --scan` opens every
combination and reports what each actually delivers.

Other commands: `cameras`, `preview`, `hands`, `screens`, `models`, `config`,
`bench capture`, `bench hands`, `bench face`.

---

## Safety

Driving the real cursor from a perception pipeline is inherently risky, so the safeguards
are structural rather than advisory:

- **Disarmed by default.** Nothing reaches the OS until you explicitly arm it.
- **The panic key is read from the hardware**, via `GetAsyncKeyState`, not from the preview
  window. This matters: the moment the system clicks something, the preview loses focus and
  a window-level key handler would stop working - precisely when you most need to stop it.
- **Buttons are always released** on every exit path, so quitting mid-drag can never leave
  the desktop with a stuck mouse button. The same applies if identity is lost mid-drag.
- **Input-side plausibility limits.** A hand cannot move a third of its own width in 33 ms;
  anything claiming otherwise is a landmark glitch and is rejected *before* gain amplifies
  it.
- **Identity gating** runs on an authenticated *session*, not a per-frame face match -
  recognition at 30 fps is unaffordable here, and gating motion frame-by-frame would freeze
  the cursor every time you glanced at your keyboard. Destructive actions additionally
  require a match within the last few seconds, which is a stricter question than "is the
  session open".
- **Cancellable confirmation.** Destructive actions run a visible countdown first, and
  cancelling is deliberately easier than confirming: there is no confirm gesture to
  perform, and the panic key or *any* gesture stops it. A confirmation requiring a second
  gesture would be vulnerable to the same false positive that triggered the first.

**Known limitation:** there is no presentation-attack detection. A photograph or a video of
an enrolled face on a phone screen will pass recognition. This is stated plainly rather
than papered over with a weak liveness heuristic that would give false confidence. The gate
is a convenience control that stops the cursor answering to the wrong person in a shared
room - not a security boundary against someone deliberately attacking it.

---

## Measured on the target machine

Intel i5-1035G1 (4C/8T, 1.0 GHz), 8 GB RAM, **CPU-only** - the 2 GB GPU runs a driver
(CUDA 11.4) too old for current ONNX Runtime and Torch CUDA builds. Logitech Brio 100 at
1280x720.

Full pipeline, everything enabled:

| Stage | Mean | p95 |
|---|---|---|
| preprocess | 0.30 ms | 0.46 |
| hand landmarks | **11.24 ms** | 12.53 |
| face (duty-cycled) | **0.01 ms** | 0.01 |
| gestures | 0.16 ms | 0.20 |
| pointer | 0.00 ms | 0.01 |
| dispatch | 0.00 ms | 0.00 |
| HUD render | 3.86 ms | 4.62 |
| display | 2.99 ms | 3.42 |
| **total** | **18.6 ms** | of a 33.3 ms budget - **44 % headroom** |

Sustained **29.9 fps**, 0 % dropped frames, capture latency 1.95 ms.

Face recognition costs **39.4 ms** for a full cycle (detect 25.3, align 0.7, embed 13.5,
match 0.01) at **100 % detection rate** - but it shows as 0.01 ms per frame above, because
it only runs when its answer could change something. Once a session is open that is once
every 15 s: **0.26 % steady-state load**. With gating off and nobody enrolled it does not
run at all, since no answer it could produce would matter.

### Findings that drove the design, all measured rather than assumed

1. **Backend choice was worth 6x throughput.** OpenCV's DirectShow backend negotiates
   uncompressed YUY2 at 720p, which this camera caps at 5 fps;
   `set(CAP_PROP_FOURCC, MJPG)` returns `True` and changes nothing. MSMF refuses the
   property outright yet delivers 30 fps. No static rule predicts this, so ARGUS measures
   each backend and keeps whichever actually delivers, caching the verdict.
2. **`OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` cut camera open time from 24.6 s to
   0.6 s** with identical throughput.
3. **Tracking one hand instead of two halved landmark cost** (28.4 -> 12.1 ms). Each extra
   hand is a full model pass.
4. **An output jump clamp was overriding the gain curve.** It fired on essentially every
   move (mean 401 px against a 420 px limit), flattening the ballistics into a constant.
   Clamping hand displacement at the *input* instead is physically meaningful and leaves
   the curve intact.
5. **Face recognition with nothing to decide cost 10 fps.** With gating off and an empty
   gallery it was still hunting for a face every fifth frame, competing with hand tracking
   for the same cores. Work that cannot change a decision is not cheap work.
6. **An index does not identify a camera.** DirectShow and Media Foundation enumerate
   devices independently and, on this machine, in *reverse* order. Resolving a name
   through one and capturing through the other opened the wrong camera while reporting
   the right name - silently, with no error. Cameras are now identified by
   `backend:index`, and `auto` chooses from a scan in which every pair was really opened
   and measured.
7. **A wedged capture backend must not be permanent.** Media Foundation can keep reporting
   a device as openable while returning no frames, after a process holding it exits
   uncleanly. Since the negotiated backend is cached, that would fail every future run
   identically - so a persistent no-frames failure now falls back to another backend and
   discards the cached verdict.

---

## Architecture

```
argus/
  config.py          typed, layered, strict configuration
  metrics.py         allocation-free rolling timers
  capture/           camera discovery, backend negotiation, threaded reader
  hands/             MediaPipe Tasks landmarks, hand geometry, calibration
  gestures/fsm.py    Schmitt triggers, debouncing, click/drag discrimination
  face/
    detector.py      SCRFD, decoded straight from the ONNX graph
    align.py         Umeyama similarity transform to the ArcFace frame
    embedder.py      ArcFace 512-d embeddings
    gallery.py       enrolled identities, matching, threshold analysis
    session.py       authenticated operator sessions
    pipeline.py      duty-cycled scheduling
  control/
    filters.py       One Euro filter, velocity gain curve
    screens.py       virtual desktop, per-monitor DPI, monitor enumeration
    injector.py      SendInput via ctypes
    pointer.py       clutch, relative mapping, gain, click freeze, scroll
    dispatcher.py    the only code allowed to affect the machine
    confirm.py       cancellable countdown for destructive actions
    hotkeys.py       global key state
  app.py             the integrated runtime
```

### Two Windows details that silently break naive implementations

Both are real on the development machine, not hypothetical:

**The virtual desktop origin is not (0, 0).** The primary display's top-left is the origin,
so a monitor placed to its *left* occupies **negative X**. Here the external 2560x1440 sits
at `x = -2560`. Code assuming `0 <= x < screen_width` cannot reach it at all. Absolute
coordinates are computed over the whole virtual desktop, with `width - 1` in the
denominator - using `width` makes the rightmost column unreachable, so the maximise button
can never be clicked.

**DPI awareness must be declared before any metric is read.** The laptop panel here runs at
150 % scaling; without `PER_MONITOR_AWARE_V2` every coordinate is wrong by 1.5x and the
error grows with distance from the origin.

---

## Testing

```bash
.venv\Scripts\python.exe -m pytest
```

212 tests, no hardware required.

Gestures and cursor maths are tested against **synthetic hands** with exactly specified
geometry, so a test can assert something precise - "the same pinch at four times the
apparent size must produce an identical event sequence" - with no ambiguity.

The fixture deliberately refuses to build impossible hands. A hand has fewer degrees of
freedom than a test might ask for: with the index extended and the middle curled, the thumb
cannot be 0.15 hand-widths from the middle fingertip *and* 0.95 from the index fingertip,
because those tips are two hand-widths apart. The thumb is placed relative to one named
target and the rest follows. An earlier version allowed the contradiction and silently
produced hands whose middle finger was implausibly extended - which only surfaced when
scroll mode started reading them as two-finger poses.

Coverage includes hysteresis (a signal oscillating inside the deadband must produce zero
flips), click-versus-drag discrimination, depth invariance of both pointing and scrolling,
negative-coordinate mapping, click-drift, stuck-button prevention, SCRFD anchor decoding,
Umeyama alignment against a known transform, gallery matching and margin rejection,
model-mismatch refusal, session expiry and freshness, confirmation cancellation, and the
full landmarks -> gestures -> pointer -> dispatch chain.

---

## Status

- [x] **Phase 0-1** Foundation, camera discovery and selection, threaded capture
- [x] **Phase 2** Hand landmark engine
- [x] **Phase 3** Gesture state machine
- [x] **Phase 4** Cursor core: filtering, gain, clutch, injection
- [x] **Phase 5** Click, right-click, double-click, drag
- [x] **Phase 6** Face recognition, enrolment, authenticated sessions, identity gating
- [x] **Phase 7a** Two-finger scroll, cancellable confirmation countdown
- [ ] **Phase 7b** Two-hand zoom, voice

`security.require_identity` defaults to `false` so the system is usable before anyone
enrols; turn it on after `argus enroll`.

## Privacy

Face embeddings are biometric data. `data/` and `models/` are git-ignored: enrolled
identities never leave this machine and are never committed.

## Credits

MediaPipe Hands (Zhang et al.) - SCRFD and ArcFace, InsightFace (Guo et al., Deng et al.) -
One Euro filter (Casiez, Roussel & Vogel, CHI 2012) - Umeyama similarity transform
(IEEE PAMI 1991)

## License

MIT
