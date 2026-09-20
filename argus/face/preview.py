"""Phase 6 verification: live face preview and a headless recognition benchmark.

The benchmark answers the question the scheduling design rests on: face
recognition is too expensive to run every frame, so how expensive is it exactly,
and does the duty-cycled schedule keep it out of the hand-tracking budget?
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..capture.camera import CameraStream
from ..capture.devices import resolve_device
from ..config import ROOT, ArgusConfig
from ..logsetup import get_logger
from ..metrics import Metrics
from ..ui.overlay import COLORS, draw_box, draw_panel, draw_text, fps_color
from .align import face_quality, quality_problems
from .detector import FaceDetector
from .embedder import FaceEmbedder
from .gallery import Gallery
from .pipeline import FacePipeline

log = get_logger("face.preview")

WARMUP = 10


def _label_for(match) -> tuple[str, tuple[int, int, int]]:
    if match is None:
        return "no match run", COLORS["muted"]
    if match.is_known:
        return f"{match.name}  {match.score:.2f}", COLORS["known"]
    return f"unknown  {match.score:.2f}", COLORS["unknown"]


def run_face_preview(cfg: ArgusConfig, seconds: float = 0.0) -> int:
    """Live face detection, recognition and session state."""
    import cv2

    pipeline = FacePipeline(cfg)
    try:
        pipeline.start()
    except Exception as exc:
        print(f"\n  {exc}\n")
        return 2

    device = resolve_device(
        cfg.capture.camera.device,
        backend=cfg.capture.camera.backend,
        exclude_ir=cfg.capture.camera.exclude_ir,
    )
    cam = CameraStream(cfg.capture.camera, device=device).start()
    metrics = Metrics(window=cfg.runtime.metrics_window)

    window = cfg.ui.window_name + " - faces"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, cfg.ui.preview_width, int(cfg.ui.preview_width * 9 / 16))
    deadline = time.perf_counter() + seconds if seconds > 0 else None

    # In the preview the schedule is bypassed so every frame is measured -
    # otherwise the display would freeze on a stale result between runs.
    force_every_frame = True

    try:
        while True:
            frame = cam.read(timeout=2.0)
            if frame is None:
                if not cam.is_running:
                    break
                continue
            now = time.perf_counter()
            metrics.mark_frame()

            if force_every_frame:
                pipeline.session.tick(now)
                start = time.perf_counter()
                faces = pipeline.detector.detect(frame.image)
                detect_ms = (time.perf_counter() - start) * 1000.0
                metrics.record("face.detect", detect_ms)
                observation = None
                if faces:
                    from .align import align_face

                    box = max(faces, key=lambda f: f.area)
                    start = time.perf_counter()
                    try:
                        embedding = pipeline.embedder.embed(align_face(frame.image, box.keypoints))
                        metrics.record("face.embed", (time.perf_counter() - start) * 1000.0)
                        match = pipeline.gallery.match(
                            embedding,
                            cfg.face.recognizer.match_threshold,
                            cfg.face.recognizer.margin_threshold,
                        )
                    except ValueError:
                        match = None
                    observation = (box, match)
            else:
                obs = pipeline.update(frame.image, frame.index, now)
                observation = (obs.box, obs.match) if obs.box else None

            canvas = frame.image.copy()
            lines = []
            if observation:
                box, match = observation
                label, colour = _label_for(match)
                quality = face_quality(box, frame.image.shape)
                problems = quality_problems(quality)
                draw_box(canvas, tuple(box.bbox), colour, label=label)
                for kx, ky in box.keypoints:
                    cv2.circle(canvas, (int(kx), int(ky)), 2, colour, -1, cv2.LINE_AA)
                lines = [
                    f"face      {box.width:.0f}x{box.height:.0f}px  score {box.score:.2f}",
                    f"roll      {quality['roll_deg']:.0f} deg   yaw {quality['yaw_ratio']:+.2f}",
                ]
                if match is not None:
                    lines.append(f"match     {match.describe()}")
                    lines.append(f"margin    {match.margin:+.3f}")
                if problems:
                    lines.append("! " + "; ".join(problems))
            else:
                lines = ["no face detected"]

            lines.append(
                f"detect    {metrics.mean_ms('face.detect'):5.1f} ms"
                f"   embed {metrics.mean_ms('face.embed'):5.1f} ms"
            )
            lines.append(f"enrolled  {', '.join(pipeline.gallery.names) or 'nobody'}")
            lines.append(f"session   {pipeline.session.describe(now)}")

            draw_panel(canvas, lines, origin=(12, 12), title="ARGUS  -  faces", min_width=420)
            draw_text(canvas, f"{metrics.fps:4.1f} FPS", (canvas.shape[1] - 150, 40),
                      0.9, fps_color(metrics.fps, 12.0), 2)

            cv2.imshow(window, canvas)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if deadline is not None and now > deadline:
                break
    finally:
        pipeline.close()
        cam.stop()
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    print()
    print(metrics.report(header="Face recognition"))
    return 0


def bench_face(cfg: ArgusConfig, seconds: float = 20.0, out: str | None = None) -> int:
    """Headless face benchmark - the Phase 6 acceptance gate."""
    detector = FaceDetector(cfg.face.detector, cfg.runtime).start()
    embedder = FaceEmbedder(cfg.face.recognizer, cfg.runtime).start()
    gallery = Gallery.load(cfg.face.gallery_path, model=cfg.face.recognizer.model,
                           strict_model=False)

    device = resolve_device(
        cfg.capture.camera.device,
        backend=cfg.capture.camera.backend,
        exclude_ir=cfg.capture.camera.exclude_ir,
    )
    cam = CameraStream(cfg.capture.camera, device=device).start()

    print(
        f"Benchmarking face recognition for {seconds:.0f} s\n"
        f"  detector  {cfg.face.detector.model} at {cfg.face.detector.input_size}px\n"
        f"  embedder  {cfg.face.recognizer.model}\n"
        f"  gallery   {len(gallery.names)} identities\n"
        "  Sit in front of the camera for a representative measurement."
    )

    for _ in range(WARMUP):
        frame = cam.read(timeout=2.0)
        if frame is not None:
            detector.detect(frame.image)

    metrics = Metrics(window=int(max(120, seconds * 40)))
    frames = detected = recognised = 0
    scores: list[float] = []
    end = time.perf_counter() + seconds
    try:
        while time.perf_counter() < end:
            frame = cam.read(timeout=2.0)
            if frame is None:
                continue
            frames += 1
            metrics.mark_frame()

            with metrics.timer("face.detect"):
                faces = detector.detect(frame.image)
            if not faces:
                continue
            detected += 1
            box = max(faces, key=lambda f: f.area)
            try:
                from .align import align_face

                with metrics.timer("face.align"):
                    aligned = align_face(frame.image, box.keypoints)
                with metrics.timer("face.embed"):
                    embedding = embedder.embed(aligned)
            except ValueError:
                continue
            with metrics.timer("face.match"):
                match = gallery.match(
                    embedding,
                    cfg.face.recognizer.match_threshold,
                    cfg.face.recognizer.margin_threshold,
                )
            scores.append(match.score)
            if match.is_known:
                recognised += 1
    finally:
        detector.close()
        embedder.close()
        cam.stop()

    detect_stat = metrics.get("face.detect")
    embed_stat = metrics.get("face.embed")
    total_ms = metrics.mean_ms("face.detect") + metrics.mean_ms("face.align") + \
        metrics.mean_ms("face.embed") + metrics.mean_ms("face.match")

    print()
    print(metrics.report(header="Face benchmark"))
    print(f"  frames {frames}  with a face {detected} ({100.0*detected/max(frames,1):.0f}%)")
    if gallery.is_empty:
        print("  no gallery, so recognition rate is not meaningful "
              "(run: argus enroll <name>)")
    else:
        print(f"  recognised {recognised}/{max(detected,1)} "
              f"({100.0*recognised/max(detected,1):.0f}%)")
    if scores:
        import numpy as np

        a = np.asarray(scores)
        print(f"  similarity  mean {a.mean():.3f}  min {a.min():.3f}  max {a.max():.3f}")

    # The whole point of duty-cycling is that this work stays out of the
    # per-frame budget. It only has to be affordable at the rate it runs.
    reverify = max(cfg.security.reverify_interval_s, 0.001)
    steady_load_pct = 100.0 * (total_ms / 1000.0) / reverify

    checks = [
        ("detect mean", detect_stat.mean <= 120.0, f"{detect_stat.mean:6.1f} ms  (need <= 120)"),
        ("embed mean", embed_stat.mean <= 60.0, f"{embed_stat.mean:6.1f} ms  (need <= 60)"),
        ("full cycle", total_ms <= 200.0, f"{total_ms:6.1f} ms  (need <= 200)"),
        ("steady-state load", steady_load_pct <= 3.0,
         f"{steady_load_pct:6.2f} %   (one pass every {reverify:.0f}s, need <= 3)"),
        ("detection rate", detected >= frames * 0.5 or frames == 0,
         f"{100.0*detected/max(frames,1):6.1f} %   (need >= 50, requires a face in view)"),
    ]
    print()
    for label, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {label:<18} {detail}")
    ok = all(p for _, p, _ in checks)
    print(f"\n  Phase 6 face gate: {'PASS' if ok else 'FAIL'}\n")

    if out:
        path = Path(out)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "detector": cfg.face.detector.model,
                    "embedder": cfg.face.recognizer.model,
                    "input_size": cfg.face.detector.input_size,
                    "frames": frames,
                    "detected": detected,
                    "recognised": recognised,
                    "identities": len(gallery.names),
                    "full_cycle_ms": round(total_ms, 2),
                    "steady_load_pct": round(steady_load_pct, 3),
                    "passed": ok,
                    "metrics": metrics.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  results written to {path}")
    return 0 if ok else 1
