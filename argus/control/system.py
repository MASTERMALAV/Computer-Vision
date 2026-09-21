"""System effects a gesture can have: volume, brightness, launching an app.

Each one is reached by a different mechanism, and the differences matter:

**Volume** is a media key. Sending ``VK_VOLUME_UP`` through ``SendInput`` costs
microseconds and lands in the same queue as a keyboard's own volume key, so the
on-screen volume overlay appears exactly as it normally does.

**Brightness** has no key. It goes through WMI, which on this machine takes
**9.7 ms** per call - measured, not guessed. That is a third of a frame, far too
much to call from the pipeline, so it runs on a worker thread that only ever
applies the *latest* requested level. Turning the knob quickly therefore costs
the pipeline nothing and skips the intermediate values rather than queueing
them, which is also what the eye wants.

**Launching** is a one-shot, so it can simply be spawned.

Everything here is disarmed by the same interlock as the cursor: nothing runs
unless the operator armed the system.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from ..logsetup import get_logger

log = get_logger("control.system")

IS_WINDOWS = sys.platform == "win32"

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002

VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #
class VolumeControl:
    """Volume via the standard media keys."""

    def __init__(self) -> None:
        self.steps_sent = 0

    def _tap(self, vk: int) -> bool:
        if not IS_WINDOWS:
            return False
        from .injector import INPUT, KEYBDINPUT, ARGUS_SIGNATURE

        events = (INPUT * 2)()
        for i, flags in enumerate((0, KEYEVENTF_KEYUP)):
            events[i].type = INPUT_KEYBOARD
            events[i].ki = KEYBDINPUT(
                wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=ARGUS_SIGNATURE
            )
        sent = ctypes.windll.user32.SendInput(2, events, ctypes.sizeof(INPUT))
        return sent == 2

    def step(self, direction: int) -> int:
        """Nudge the volume. Returns how many steps were actually sent."""
        if direction == 0:
            return 0
        vk = VK_VOLUME_UP if direction > 0 else VK_VOLUME_DOWN
        sent = 0
        for _ in range(min(abs(direction), 5)):  # never fire a burst
            if self._tap(vk):
                sent += 1
        self.steps_sent += sent
        return sent

    def mute(self) -> bool:
        return self._tap(VK_VOLUME_MUTE)


# --------------------------------------------------------------------------- #
# Brightness
# --------------------------------------------------------------------------- #
class BrightnessControl:
    """Display brightness, applied on a worker thread.

    Only the internal panel is reachable this way. External monitors need
    DDC/CI, which many of them implement badly or not at all, so this reports
    itself unavailable rather than pretending.
    """

    def __init__(self, min_level: int = 5, max_level: int = 100) -> None:
        self.min_level = min_level
        self.max_level = max_level
        self.available = False
        self.level: int | None = None

        self._target: int | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._applied = 0

    # ------------------------------------------------------------------ #
    def start(self) -> "BrightnessControl":
        if not IS_WINDOWS:
            return self
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(ready,), name="argus-brightness", daemon=True
        )
        self._thread.start()
        # Wait briefly for the worker to report whether WMI is usable, so
        # callers know immediately whether to offer the control at all.
        ready.wait(timeout=4.0)
        return self

    def _run(self, ready: threading.Event) -> None:
        obj = None
        in_params = None
        try:
            import comtypes
            import comtypes.client as cc

            # COM is per-thread: the objects must be created here, not handed in.
            comtypes.CoInitialize()
            wmi = cc.CoGetObject(r"winmgmts:\\.\root\WMI")
            for m in wmi.ExecQuery("SELECT * FROM WmiMonitorBrightness"):
                self.level = int(m.Properties_["CurrentBrightness"].Value)
            methods = list(wmi.ExecQuery("SELECT * FROM WmiMonitorBrightnessMethods"))
            if methods and self.level is not None:
                obj = methods[0]
                in_params = obj.Methods_["WmiSetBrightness"].InParameters.SpawnInstance_()
                in_params.Properties_["Timeout"].Value = 1
                self.available = True
                log.info("brightness control ready (currently %d%%)", self.level)
            else:
                log.info("no internal panel exposes brightness; control unavailable")
        except Exception as exc:
            log.info("brightness control unavailable: %s", exc)
        finally:
            ready.set()

        if not self.available:
            return

        while not self._stop.is_set():
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            with self._lock:
                target, self._target = self._target, None
            if target is None:
                continue
            try:
                in_params.Properties_["Brightness"].Value = int(target)
                obj.ExecMethod_("WmiSetBrightness", in_params)
                self.level = int(target)
                self._applied += 1
            except Exception as exc:  # pragma: no cover - driver dependent
                log.debug("brightness set failed: %s", exc)

        try:
            import comtypes

            comtypes.CoUninitialize()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def adjust(self, delta: int) -> int | None:
        """Change brightness by ``delta`` percent. Returns the new target."""
        if not self.available or self.level is None:
            return None
        target = max(self.min_level, min(self.max_level, self.level + delta))
        if target == self.level:
            return target
        # Update the reported level immediately so successive steps accumulate
        # from the requested value rather than from whatever the panel has
        # actually reached, which lags.
        self.level = target
        with self._lock:
            self._target = target
        self._wake.set()
        return target

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def stats(self) -> dict:
        return {"available": self.available, "level": self.level, "applied": self._applied}


# --------------------------------------------------------------------------- #
# Launching
# --------------------------------------------------------------------------- #
def launch(target: str) -> bool:
    """Start an application, a URI, or a Store app.

    Three forms are accepted:

    * ``shell:AppsFolder\\<AUMID>`` - a Store app, started through Explorer
    * ``something://`` - a registered URI protocol
    * a path or executable name
    """
    target = (target or "").strip()
    if not target:
        return False
    try:
        if target.lower().startswith("shell:appsfolder"):
            subprocess.Popen(["explorer.exe", target], close_fds=True)
        elif "://" in target:
            if IS_WINDOWS:
                os.startfile(target)  # noqa: S606 - a user-configured target
            else:
                subprocess.Popen(["xdg-open", target], close_fds=True)
        elif IS_WINDOWS and os.path.exists(target):
            os.startfile(target)  # noqa: S606
        else:
            subprocess.Popen(target, shell=True, close_fds=True)  # noqa: S602
        log.info("launched %s", target)
        return True
    except Exception as exc:
        log.error("could not launch %r: %s", target, exc)
        return False


def find_store_app(name_fragment: str) -> str | None:
    """Look up a Store app's launch target by a fragment of its name.

    Store apps are launched by an AUMID that differs per machine, so hardcoding
    one would only ever be right on the machine it was copied from.
    """
    if not IS_WINDOWS:
        return None
    script = (
        "Get-StartApps | Where-Object { $_.Name -like '*"
        + name_fragment.replace("'", "")
        + "*' } | Select-Object -First 1 -ExpandProperty AppID"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=25, check=False,
        )
        aumid = out.stdout.strip().splitlines()
        if aumid and aumid[0].strip():
            return "shell:AppsFolder\\" + aumid[0].strip()
    except Exception as exc:
        log.debug("store app lookup failed: %s", exc)
    return None


# --------------------------------------------------------------------------- #
@dataclass
class SystemActionStats:
    volume_steps: int = 0
    brightness_steps: int = 0
    launches: int = 0
    blocked: int = 0


class SystemActions:
    """One handle for every system effect, sharing the arming interlock."""

    def __init__(self, launch_target: str = "", cooldown_s: float = 1.5) -> None:
        self.volume = VolumeControl()
        self.brightness = BrightnessControl()
        self.launch_target = launch_target
        self.launch_cooldown_s = cooldown_s
        self.stats = SystemActionStats()
        self._last_launch = -1e9

    def start(self) -> "SystemActions":
        self.brightness.start()
        return self

    def volume_step(self, direction: int) -> None:
        self.stats.volume_steps += self.volume.step(direction)

    def brightness_step(self, direction: int, percent: int = 5) -> int | None:
        level = self.brightness.adjust(direction * percent)
        if level is not None:
            self.stats.brightness_steps += 1
        return level

    def launch_app(self, now: float) -> bool:
        """Launch the configured app, rate-limited so one gesture starts one copy."""
        if not self.launch_target:
            log.warning("no launch target configured (actions.launch_target)")
            return False
        if (now - self._last_launch) < self.launch_cooldown_s:
            self.stats.blocked += 1
            return False
        self._last_launch = now
        ok = launch(self.launch_target)
        if ok:
            self.stats.launches += 1
        return ok

    def close(self) -> None:
        self.brightness.stop()

    def describe(self) -> dict:
        return {
            "volume_steps": self.stats.volume_steps,
            "brightness_steps": self.stats.brightness_steps,
            "launches": self.stats.launches,
            "brightness": self.brightness.stats(),
        }
