"""Interactive enrolment, and the tools to check that it worked.

Enrolment quality decides whether recognition works, so this is deliberately
picky. Samples are rejected when the face is too small, too turned, tilted, or
cut off by the frame edge, and the reason is shown live rather than silently
accepted - a gallery built from bad crops fails in ways that look like a broken
threshold and are very hard to debug later.

Samples are also spread over several head poses, because the gallery matches on
the closest stored sample. Enrolling twenty near-identical frontal frames gives
twenty copies of one pose and still fails the moment the operator turns their
head.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..capture.camera import CameraStream
from ..capture.picker import select_camera
from ..config import ArgusConfig
from ..logsetup import get_logger
from ..ui.overlay import COLORS, draw_bar, draw_box, draw_panel, draw_text
from .align import align_face, face_quality, quality_problems
from .detector import FaceDetector
from .embedder import FaceEmbedder
from .gallery import Gallery

log = get_logger("face.enroll")


@dataclass
class EnrolStep:
    key: str
    title: str
    instruction: str
    samples: int = 6
    collected: list[np.ndarray] = field(default_factory=list)


def default_steps(per_pose: int = 6) -> list[EnrolStep]:
    """Poses chosen to span the variation that normal desk work produces."""
    return [
        EnrolStep("front", "LOOK AT THE CAMERA", "Face the camera straight on.", per_pose),
        EnrolStep("left", "TURN SLIGHTLY LEFT", "Turn your head a little to the left.", per_pose),
        EnrolStep("right", "TURN SLIGHTLY RIGHT", "Turn your head a little to the right.", per_pose),
        EnrolStep("down", "LOOK SLIGHTLY DOWN", "Tilt your head down, as if at a keyboard.", per_pose),
        EnrolStep("close", "LEAN IN", "Move a little closer to the camera.", per_pose),
    ]


def run_enrolment(
    cfg: ArgusConfig,
    name: str,
    per_pose: int = 6,
    reset: bool = False,
) -> int:
    """Capture face samples for one person and add them to the gallery."""
    import cv2

    name = name.strip()
    if not name:
        print("  A name is required:  argus enroll <name>")
        return 2

    detector = FaceDetector(cfg.face.detector, cfg.runtime).start()
    embedder = FaceEmbedder(cfg.face.recognizer, cfg.runtime).start()

    try:
        gallery = Gallery.load(
            cfg.face.gallery_path, model=cfg.face.recognizer.model, strict_model=not reset
        )
    except Exception as exc:
        print(f"\n  {exc}\n")
        return 2
    if reset:
        gallery = Gallery(model=cfg.face.recognizer.model, path=cfg.face.gallery_path)
        print("  --reset: starting a fresh gallery")
    gallery.model = cfg.face.recognizer.model

    device = select_camera(cfg)
    cam = CameraStream(cfg.capture.camera, device=device).start()

    window = cfg.ui.window_name + " - enrolment"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, cfg.ui.preview_width, int(cfg.ui.preview_width * 9 / 16))

    steps = default_steps(per_pose)
    total_wanted = sum(s.samples for s in steps)

    print()
    print(f"  Enrolling: {name}")
    print(f"  {len(steps)} poses, {per_pose} samples each ({total_wanted} total).")
    print("  Samples are only taken when the face is clear enough. Press q to abort.")
    print()

    aborted = False
    last_capture = 0.0
    CAPTURE_INTERVAL = 0.22  # avoid capturing near-identical consecutive frames

    try:
        for step in steps:
            print(f"  -> {step.title}: {step.instruction}")
            while len(step.collected) < step.samples:
                frame = cam.read(timeout=2.0)
                if frame is None:
                    if not cam.is_running:
                        aborted = True
                        break
                    continue

                faces = detector.detect(frame.image)
                canvas = frame.image.copy()
                problems: list[str] = []
                box = None

                if faces:
                    box = max(faces, key=lambda f: f.area)
                    quality = face_quality(box, frame.image.shape)
                    problems = quality_problems(quality)
                    colour = COLORS["error"] if problems else COLORS["ok"]
                    draw_box(canvas, tuple(box.bbox), colour,
                             label=f"{box.score:.2f}", corner_style=True)
                    for kx, ky in box.keypoints:
                        cv2.circle(canvas, (int(kx), int(ky)), 2, colour, -1, cv2.LINE_AA)

                    now = time.perf_counter()
                    if not problems and (now - last_capture) >= CAPTURE_INTERVAL:
                        try:
                            aligned = align_face(frame.image, box.keypoints)
                            step.collected.append(embedder.embed(aligned))
                            last_capture = now
                        except ValueError as exc:
                            problems = [str(exc)]
                else:
                    problems = ["no face detected"]

                lines = [step.instruction, f"captured {len(step.collected)}/{step.samples}"]
                lines += [f"! {p}" for p in problems]
                draw_panel(canvas, lines, origin=(12, 12),
                           title=f"{step.title}  ({steps.index(step) + 1}/{len(steps)})",
                           min_width=430)
                draw_bar(canvas, (22, 104), 400, len(step.collected) / step.samples,
                         COLORS["error"] if problems else COLORS["ok"], height=10)
                done = sum(len(s.collected) for s in steps)
                draw_text(canvas, f"total {done}/{total_wanted}", (22, 140), 0.55,
                          COLORS["accent"], 1)

                cv2.imshow(window, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    aborted = True
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    aborted = True
                    break
            if aborted:
                break
            print(f"     captured {len(step.collected)}")
    finally:
        detector.close()
        embedder.close()
        cam.stop()
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    collected = [e for step in steps for e in step.collected]
    if aborted and len(collected) < total_wanted // 2:
        print("\n  Aborted - the gallery was not changed.\n")
        return 1
    if not collected:
        print("\n  No usable samples captured. Check lighting and camera framing.\n")
        return 1

    if reset or name not in gallery.names:
        pass
    else:
        print(f"  Adding to the {gallery.count(name)} sample(s) already stored for {name}.")

    gallery.add(name, np.stack(collected))
    path = gallery.save(cfg.face.gallery_path)

    print(f"\n  Enrolled {name}: {len(collected)} samples -> {path}")
    print(f"  Gallery now holds {len(gallery.names)} identities, "
          f"{gallery.total_samples()} samples.")

    _print_analysis(gallery, cfg)
    print("  Turn identity gating on with:")
    print("    --set security.require_identity=true --set security.operator=" + name)
    print()
    return 0


def _print_analysis(gallery: Gallery, cfg: ArgusConfig) -> None:
    report = gallery.analyse()
    genuine, impostor = report.get("genuine"), report.get("impostor")
    if not genuine:
        return

    print()
    print("  Similarity analysis (cosine, higher = more alike)")
    print(f"    same person   : min {genuine['min']:.3f}  p05 {genuine['p05']:.3f}  "
          f"mean {genuine['mean']:.3f}  ({genuine['n']} pairs)")
    if impostor:
        print(f"    different     : mean {impostor['mean']:.3f}  p95 {impostor['p95']:.3f}  "
              f"max {impostor['max']:.3f}  ({impostor['n']} pairs)")
        print(f"    separation    : {report['separation']:+.3f}")
        if report.get("overlap"):
            print("    ! The two distributions OVERLAP. No threshold can separate these")
            print("      identities reliably - re-enrol with better lighting and more poses.")
    print(f"    current threshold : {cfg.face.recognizer.match_threshold:.2f}")
    print(f"    suggested         : {report['suggested_threshold']:.2f}")
    if report.get("note"):
        print(f"    note: {report['note']}")


def show_gallery(cfg: ArgusConfig, analyse: bool = False) -> int:
    """Print who is enrolled."""
    try:
        gallery = Gallery.load(cfg.face.gallery_path, model=cfg.face.recognizer.model)
    except Exception as exc:
        print(f"\n  {exc}\n")
        return 2

    print()
    if gallery.is_empty:
        print("  No identities enrolled.")
        print("  Enrol with:  argus enroll <your name>")
        print()
        return 0

    print(f"  Enrolled identities ({len(gallery.names)})")
    for row in gallery.summary():
        when = (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(row["enrolled_at"]))
            if row["enrolled_at"]
            else "unknown"
        )
        print(f"    {row['name']:<20} {row['samples']:>3} samples   enrolled {when}")
    print(f"  Model: {gallery.model or 'unknown'}")
    if analyse:
        _print_analysis(gallery, cfg)
    print()
    return 0


def forget(cfg: ArgusConfig, name: str) -> int:
    """Remove one identity from the gallery."""
    gallery = Gallery.load(cfg.face.gallery_path, model=cfg.face.recognizer.model)
    if not gallery.remove(name):
        print(f"\n  '{name}' is not enrolled. Known: {', '.join(gallery.names) or 'nobody'}\n")
        return 1
    gallery.save(cfg.face.gallery_path)
    print(f"\n  Removed {name}. {len(gallery.names)} identities remain.\n")
    return 0
