# ARGUS - how to drive it

Everything in plain language. Keep this open the first few times.

Run commands from the project folder. **The leading `.\` matters** - PowerShell will not
run a program from the current directory without it.

---

## The one idea behind it all

Think of your hand as a mouse you can pick up.

- **Index finger straight** = the mouse is *on the mat*. Moving your hand moves the cursor.
- **Fingers relaxed / loose fist** = the mouse is *lifted off the mat*. The cursor freezes
  where it is, and you can move your hand anywhere without the cursor following.

That second part is the important one. It means you never have to stretch. Move your hand
a little, relax it, bring your hand back to a comfortable spot, point again, keep moving.
Exactly like lifting a real mouse and re-placing it. **Keep your hand low, near the desk.**

The cursor follows your **palm**, not your fingertip. So when you pinch to click, the
cursor does not slide off the thing you were aiming at.

---

## The controls

| What you do | What happens |
|---|---|
| **Index finger straight**, move your hand | Cursor moves |
| **Fingers relaxed** (loose fist) | Cursor parks - reposition your hand freely |
| **Thumb taps index fingertip** | Left click |
| **Thumb taps middle fingertip** | Right click |
| **Thumb taps index twice quickly** | Double click |
| **Thumb holds index** (~0.35 s) then move | Drag - release the pinch to drop |
| **Thumb holds middle** (~0.30 s) then move up/down | Scroll - release to stop |
| **Open your whole hand**, move it up / down | **Volume** up / down |
| **Two fingers up (a V)**, move up / down | **Brightness** up / down |
| **Thumbs up**, held ~0.9 s | **Launch WhatsApp** |

### Which fingers, exactly

```
        index    middle   ring   pinky
          |        |        |      |
   thumb  |        |        |      |
      \   |        |        |      |
       [ your hand, palm toward the camera ]
```

- **Thumb** - the button. It touches the index tip, or the middle tip.
- **Index** - the on/off switch. Straight = cursor live. Curled = cursor parked.
- **Middle** - the thumb's second target: tap it to right-click, hold it to scroll.
- **Ring and pinky** - not used. Do whatever is comfortable.

**There is really only one rule to remember**, and it is the same for both fingers:

| | quick tap | hold, then move |
|---|---|---|
| thumb + **index** | left click | **drag** |
| thumb + **middle** | right click | **scroll** |

Tap = a click. Hold = a sustained action. That is the whole system.

### Things worth knowing

- **Moving your hand fast?** Gestures are ignored while your hand is moving quickly,
  because the tracking is least reliable then. Slow down slightly before you click.
- **Scrolling has momentum.** A quick flick keeps scrolling and coasts to a stop, like a
  phone. A slow, deliberate movement stops the instant you release, so you can position
  precisely. You do not have to sweep your arm down the whole page.
- **Clicking is switched off while you scroll**, so a stray index pinch cannot click
  whatever the page just scrolled under the cursor.
- **You cannot drag and scroll at once** - you only have one thumb, so letting go of one
  grip is what frees it for the other.
- **Move slowly for precision, quickly to cover distance.** The cursor speeds up when your
  hand does, like a normal mouse pointer.
- **While you type, your hand is ignored.** An index finger resting over the keyboard
  looks exactly like the pointing pose, so gestures are suppressed for about half a second
  after each keystroke - the same thing a laptop touchpad does. Holding Ctrl or Shift does
  *not* count as typing, so Ctrl-click and Shift-click still work, and a drag or scroll
  already in progress is never interrupted.
- **Your distance from the camera does not matter.** Every measurement is scaled by the
  size of your own hand, so a pinch means the same thing near or far.

---

## Keys

| Key | What it does |
|---|---|
| **F9** | Arm / disarm. **Nothing touches your cursor until you press this.** |
| **Esc** (hold ~0.4 s) | Emergency stop. Works from *any* window, even if ARGUS lost focus. |
| **F10** | Jump the cursor to your other monitor |
| **c** | Switch to the next camera |
| **F11** | Cycle the display: full preview -> small pill -> nothing |
| **Ctrl+Alt+Q** | Quit (works in any mode) |
| **q** | Quit (full preview window only) |
| **h** | Hide the help panel |

### The three display modes

The big camera window is a *debugging* tool. Once you trust the gestures you do not
need it, and it is not free - it costs about a third of the per-frame budget.

| Mode | What you see | Cost per frame |
|---|---|---|
| **full** | camera feed, hand skeleton, full HUD | 9.14 ms |
| **pill** | a small bar in the corner: armed state, what your hand is doing, pinch meter | **0.65 ms** |
| **none** | nothing | 0 ms |

Press **F11** to cycle, or start in a mode directly:

```bash
.\argus mouse --hud pill
```

The pill is **click-through** - it can never swallow a click, which matters because the
system is driving your cursor - and it never takes keyboard focus. Because of that, all
its controls (F9, F11, Esc, Ctrl+Alt+Q) are read straight from the keyboard and work no
matter which window is active.

It always starts **disarmed**. The window shows exactly what it *would* do, in a dry run,
so you can watch the pinch numbers and get a feel for it before anything is live.

---

## Calibration - what you will be asked to do

Run it once. It measures *your* hand, because thumb length relative to palm size varies a
lot between people.

```bash
.\argus calibrate --write
```

You get **8 seconds per pose** to read the instruction and get into position, then about
1.5 seconds of holding still. The panel tells you **READY** or **ADJUST** live, with the
number it is measuring - so you can fix your hand *before* anything is recorded.

| Key | During calibration |
|---|---|
| **SPACE** | Start recording now (once it says READY) |
| **R** | Redo this pose |
| **Q** | Quit without saving |

### The five poses, and the check each one must pass

Measurements are in "hand-widths" - your wrist-to-knuckle distance is 1.0.

| # | Pose | What to do | Must measure |
|---|---|---|---|
| 1 | **OPEN HAND** | Index straight, thumb held **well away** from it. Like holding an invisible cup. | 0.55 - 2.20 |
| 2 | **PINCH** | Thumb and index fingertips **actually touching**, pad to pad. | 0.05 - 0.60 |
| 3 | **MIDDLE PINCH** | Thumb and **middle** fingertips touching. Index can stay out. | 0.05 - 0.60 |
| 4 | **POINTING** | Index finger **completely straight**, as if pointing at the screen. | 1.00 - 1.90 |
| 5 | **RELAXED** | Fingers curled into a **loose fist**. This is the cursor-off pose. | 0.10 - 0.95 |

### The two mistakes that ruin it

**1. Not actually touching during PINCH.** If your fingertips hover a centimetre apart, it
measures around 1.3 instead of 0.3 - and the resulting settings would click continuously.
This is the single most common failure.

**2. Not fully straightening during POINTING.** A half-bent index measures around 0.9
instead of 1.3, and the cursor on/off switch becomes unreliable.

Calibration now **refuses to save** if either happens, and tells you which pose to redo.
The defaults work fine, so a refused calibration leaves you no worse off.

### Also

- Keep your **whole hand inside the frame** - if it is cut off at the edge, landmarks go
  wrong.
- Sit **reasonably close**. If the panel says "TOO FAR", your hand is too small in frame.
- Decent, even lighting. Avoid a bright window directly behind you.

---

## Choosing which camera to use

If ARGUS is using the wrong camera, pick one by eye:

```bash
.\argus cameras --pick
```

It shows each working camera live, one at a time. **N** = next, **ENTER** = choose this
one, **Q** = cancel. Your choice is saved to `configs/default.yaml`, so every command uses
it from then on.

You can also switch cameras **while the mouse is running** - press **c**. The HUD shows
which camera is live.

To see what was found without choosing:

```bash
.\argus cameras --scan
```

### Why a camera is named like `msmf:0`

Windows has two ways of talking to cameras - DirectShow and Media Foundation - and **they
number the cameras differently**. On this machine:

| | index 0 | index 1 |
|---|---|---|
| DirectShow | Integrated Camera | Brio 100 |
| Media Foundation | **Brio 100** | Integrated Camera |

They are reversed. So "camera 1" is meaningless on its own - it is a different physical
camera depending on who is asking. A camera is therefore identified by **both**: `msmf:0`
means "Media Foundation, index 0", which on this machine is the Brio.

This is exactly what was wrong before: ARGUS looked the Brio up in the DirectShow list
(index 1) and then opened Media Foundation index 1 - the laptop camera. It reported "Brio
100" the whole time, because the name lookup and the capture were using different
numbering.

`argus cameras --scan` now opens every combination and reports what each one actually
delivers, and identifies unnamed ones by comparing what they see against the named ones.

### Pinning a camera by hand

```bash
.\argus mouse --camera msmf:0
```

or in `configs/default.yaml`:

```yaml
capture:
  camera:
    device: msmf:0
```

`auto` also works and now picks the fastest real camera, preferring an external one.

### If a camera shows black frames

`--scan` reports `(black frames)` when a camera opens fine but produces nothing but black.
That is almost always a **physical privacy shutter** on the laptop camera. Slide it open,
or just choose the other camera.

---

---

## Volume, brightness and launching an app

Three whole-hand poses. Your thumb and two fingers already carry four actions between
them, so these use the shape of the hand instead.

### Volume - open your whole hand, then move it up or down

1. **Spread your whole hand open**, fingers apart, palm toward the camera.
2. The panel shows **[open_palm/volume]**. The cursor stops moving - that is how you know
   it registered.
3. **Move your hand up to raise the volume, down to lower it.** Like a slider.

That is all. You are not turning anything - just moving your open hand up and down.

### Brightness - two fingers up, then move up or down

1. **Index and middle fingers up, ring and pinky folded down.** A peace sign.
2. The panel shows **[v_sign/brightness]**.
3. **Move your hand up to brighten, down to dim.**

**Brightness only works on the laptop screen.** External monitors need a protocol called
DDC/CI which most implement badly, so ARGUS says it is unavailable rather than pretending.
Check with `.\argus doctor --live`.

### Launch WhatsApp - thumbs up, and hold

1. **Thumbs up**: fingers folded, thumb clearly out.
2. The panel counts up - **launching 40%... 70%...** - so you can see it registering.
3. At 100% it launches. Then **drop the pose** before doing it again; holding longer does
   nothing, by design, so one gesture opens one copy.

If the counter does not appear, the pose is not being seen. Keep your whole hand in frame
and make the thumb clearly separate from your fingers.

### If a pose will not register

Open the full panel and watch the pose readout:

```bash
.\argus mouse --hud full
```

The panel shows the pose it currently sees, like `[open_palm/volume]` or `[unknown/point]`.
If it says `unknown`, the hand shape is between poses - spread your fingers further, or
fold the spare ones further down.

### Preferring a dial to a slider

If you would rather turn your wrist like a volume knob than slide your hand:

```bash
.\argus mouse --set actions.rotation.mode=rotate
```

Then, in either pose, **turn your wrist like a steering wheel** - clockwise raises,
anticlockwise lowers. A turn never runs out of room, which is its one real advantage. The
slider is the default because volume and brightness are bounded at 0 and 100, and a slider
is what people reach for unprompted.

Other settings worth knowing:

```bash
.\argus mouse --set actions.rotation.units_per_step=0.05   # finer steps
.\argus mouse --set actions.rotation.invert=true           # flip the direction
.\argus mouse --set actions.brightness_percent=3           # gentler brightness
.\argus mouse --set actions.launch_hold_s=1.5              # longer hold to launch
```

### The safety rules are the same as for clicking

Changing the volume is still an effect on your machine, so it obeys the same three rules:
the system must be **armed** (F9), the operator must be **recognised** if identity gating
is on, and **nothing fires while you are typing**.

---

## Two commands worth knowing

```bash
.\argus doctor --live
```

Checks everything and tells you what to fix. Run it first whenever something feels off.

```bash
.\argus practice
```

Target practice that measures how good your pointing actually is, in bits per second, and
suggests whether the cursor is too fast or too slow for you. Works with a normal mouse too,
so you can compare.

### Jumping between monitors

Your desktop is 4480 pixels wide. Crossing it by hand takes about one and a half
hand-widths even at full speed - more than one clutch cycle. Press **F10** to jump to the
other monitor instead. It keeps your relative position, so from the top-left of one screen
you arrive at the top-left of the other.

---

## Face recognition (optional)

Only needed if you want the cursor to respond to you and nobody else.

```bash
.\argus enroll "Your Name"
```

Five head poses: straight on, slightly left, slightly right, slightly down, and leaning in.
It only accepts frames where your face is big enough, square enough to the camera, and not
cut off - so a sample is sometimes skipped, which is normal.

Then:

```bash
.\argus mouse --set security.require_identity=true --set security.operator="Your Name"
```

**How the gating behaves:** it recognises you once and then keeps a *session* open. You can
look down at your keyboard, turn away, rub your eye - the cursor keeps working. The session
only ends after about two minutes with no sight of you.

**One honest limitation:** a photo of your face on a phone will fool it. It is a
convenience control for a shared room, not real security.

---

## Suggested first session

1. `argus calibrate --write` - do the five poses carefully.
2. `argus mouse -c configs/calibrated.yaml` - it opens **disarmed**.
3. Point with your index finger and watch the HUD. The `clutch` line should say
   **ENGAGED**. Relax your hand; it should say **released**.
4. Pinch a few times and watch the `pinch i` number drop. Confirm it goes clearly below
   the `close<` value shown next to it.
5. Only when all of that looks right: press **F9** and drive.
6. If anything feels wrong, **hold Esc**.

If the cursor moves too fast or too slow for you, that is one number:
`control.gain.pixels_per_unit` in your config. Higher = faster.
