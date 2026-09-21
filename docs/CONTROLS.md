# ARGUS - how to drive it

Everything in plain language. Keep this open the first few times.

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
| **Index + middle both straight**, move hand up/down | Scroll |

### Which fingers, exactly

```
        index    middle   ring   pinky
          |        |        |      |
   thumb  |        |        |      |
      \   |        |        |      |
       [ your hand, palm toward the camera ]
```

- **Thumb** - only ever used for clicking. It taps the index tip (left click) or the
  middle tip (right click).
- **Index** - the on/off switch. Straight = cursor live. Curled = cursor parked.
- **Middle** - raise it alongside the index to switch into scrolling.
- **Ring and pinky** - not used. Do whatever is comfortable.

### Things worth knowing

- **Moving your hand fast?** Gestures are ignored while your hand is moving quickly,
  because the tracking is least reliable then. Slow down slightly before you click.
- **Clicking is switched off while scrolling.** The two-finger pose naturally brings your
  thumb near your middle finger, which would otherwise fire a right click constantly.
- **Move slowly for precision, quickly to cover distance.** The cursor speeds up when your
  hand does, like a normal mouse pointer.
- **Your distance from the camera does not matter.** Every measurement is scaled by the
  size of your own hand, so a pinch means the same thing near or far.

---

## Keys

| Key | What it does |
|---|---|
| **F9** | Arm / disarm. **Nothing touches your cursor until you press this.** |
| **Esc** (hold ~0.4 s) | Emergency stop. Works from *any* window, even if ARGUS lost focus. |
| **F10** | Snap the cursor back to the middle of your laptop screen |
| **c** | Switch to the next camera |
| **q** | Quit |
| **h** | Hide the help panel |

It always starts **disarmed**. The window shows exactly what it *would* do, in a dry run,
so you can watch the pinch numbers and get a feel for it before anything is live.

---

## Calibration - what you will be asked to do

Run it once. It measures *your* hand, because thumb length relative to palm size varies a
lot between people.

```bash
argus calibrate --write
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
argus cameras --pick
```

It shows each working camera live, one at a time. **N** = next, **ENTER** = choose this
one, **Q** = cancel. Your choice is saved to `configs/default.yaml`, so every command uses
it from then on.

You can also switch cameras **while the mouse is running** - press **c**. The HUD shows
which camera is live.

To see what was found without choosing:

```bash
argus cameras --scan
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
argus mouse --camera msmf:0
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

## Face recognition (optional)

Only needed if you want the cursor to respond to you and nobody else.

```bash
argus enroll "Your Name"
```

Five head poses: straight on, slightly left, slightly right, slightly down, and leaning in.
It only accepts frames where your face is big enough, square enough to the camera, and not
cut off - so a sample is sometimes skipped, which is normal.

Then:

```bash
argus mouse --set security.require_identity=true --set security.operator="Your Name"
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
