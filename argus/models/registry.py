"""Model weights: download, verify, extract, locate.

Weights are never committed. They are fetched on demand into ``models/`` and
verified against a pinned SHA-256, so a corrupted download or a silently
re-uploaded release asset fails loudly instead of producing a model that loads
but behaves subtly differently.

Checksums below were computed from the actual downloads used to develop and
benchmark this system. Verify with:

    python -m argus models verify
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..config import MODELS_DIR
from ..logsetup import get_logger

log = get_logger("models")

CACHE_DIR = MODELS_DIR / "cache"
CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class ModelSpec:
    """One model file, and how to obtain it."""

    name: str
    url: str
    sha256: str  # of the DOWNLOADED artefact (archive or bare file)
    size: int  # bytes of the downloaded artefact
    filename: str  # what it is called once installed under models/
    description: str
    credit: str
    member: str | None = None  # path inside the archive, if the download is a zip
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_archive(self) -> bool:
        return self.member is not None

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename

    @property
    def download_path(self) -> Path:
        return CACHE_DIR / self.url.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
REGISTRY: dict[str, ModelSpec] = {
    # ---------------- hands ---------------- #
    "hand_landmarker": ModelSpec(
        name="hand_landmarker",
        url=(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        ),
        sha256="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        size=7819105,
        filename="hand_landmarker.task",
        description="MediaPipe hand landmarks: BlazePalm detector + 21-point regressor (float16)",
        credit="Google MediaPipe (Zhang et al., MediaPipe Hands)",
        tags=("hands", "required"),
    ),
    # ---------------- face: compact pack (default) ---------------- #
    # buffalo_sc = SCRFD-500MF detector + MobileFaceNet ArcFace embedder.
    # 15 MB total, chosen because it is the accuracy/latency sweet spot on a
    # 4-core CPU. The million-distractor benchmarks that favour the large pack
    # are not representative of a gallery of a handful of enrolled people.
    "scrfd_500m": ModelSpec(
        name="scrfd_500m",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        sha256="57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72",
        size=14969382,
        filename="scrfd_500m.onnx",
        member="det_500m.onnx",
        description="SCRFD-500MF face detector (bbox + 5 keypoints)",
        credit="InsightFace (Guo et al., Sample and Computation Redistribution)",
        tags=("face", "detector"),
    ),
    "arcface_mbf": ModelSpec(
        name="arcface_mbf",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        sha256="57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72",
        size=14969382,
        filename="arcface_mbf.onnx",
        member="w600k_mbf.onnx",
        description="ArcFace MobileFaceNet, 512-d embeddings, trained on WebFace600K",
        credit="InsightFace (Deng et al., ArcFace)",
        tags=("face", "recognizer"),
    ),
    # ---------------- face: accurate pack ---------------- #
    "scrfd_10g": ModelSpec(
        name="scrfd_10g",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
        size=288621354,
        filename="scrfd_10g.onnx",
        member="det_10g.onnx",
        description="SCRFD-10GF face detector - higher recall on small/oblique faces",
        credit="InsightFace (Guo et al.)",
        tags=("face", "detector", "accurate"),
    ),
    "arcface_r50": ModelSpec(
        name="arcface_r50",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
        size=288621354,
        filename="arcface_r50.onnx",
        member="w600k_r50.onnx",
        description="ArcFace ResNet50, 512-d embeddings, WebFace600K - highest accuracy",
        credit="InsightFace (Deng et al., ArcFace)",
        tags=("face", "recognizer", "accurate"),
    ),
}


class ModelError(RuntimeError):
    """Raised when a model cannot be fetched or fails verification."""


# --------------------------------------------------------------------------- #
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(spec: ModelSpec, progress: bool = True) -> Path:
    """Fetch the artefact into the cache, resuming nothing, verifying always."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = spec.download_path

    if target.exists():
        actual = sha256_of(target)
        if actual == spec.sha256:
            log.debug("%s: cached download already verified", spec.name)
            return target
        log.warning("%s: cached download failed checksum, re-fetching", spec.name)
        target.unlink()

    tmp = target.with_suffix(target.suffix + ".part")
    log.info("downloading %s (%.1f MB) ...", spec.name, spec.size / 1e6)
    try:
        with urllib.request.urlopen(spec.url, timeout=60) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length") or spec.size)
            done = 0
            last_pct = -10
            with tmp.open("wb") as out:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if progress and total:
                        pct = int(100 * done / total)
                        if pct >= last_pct + 10:
                            log.info("  %s: %d%% (%.1f/%.1f MB)", spec.name, pct, done / 1e6, total / 1e6)
                            last_pct = pct
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise ModelError(
            f"Could not download {spec.name} from {spec.url}\n  {exc}\n"
            "Check your network connection, or download the file manually to "
            f"{target}"
        ) from exc

    actual = sha256_of(tmp)
    if actual != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise ModelError(
            f"Checksum mismatch for {spec.name}.\n"
            f"  expected {spec.sha256}\n"
            f"  actual   {actual}\n"
            "The download was corrupted, or the upstream asset changed. Refusing to use it."
        )
    tmp.replace(target)
    log.info("%s: downloaded and verified", spec.name)
    return target


def _install(spec: ModelSpec, archive: Path) -> Path:
    """Place the usable model file at ``spec.path``."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not spec.is_archive:
        shutil.copyfile(archive, spec.path)
        return spec.path

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        # Release archives are sometimes rooted in a folder and sometimes not.
        match = next((n for n in names if n == spec.member or n.endswith("/" + spec.member)), None)
        if match is None:
            raise ModelError(
                f"{spec.member!r} not found inside {archive.name}. Archive contains: {names}"
            )
        with zf.open(match) as src, spec.path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    log.info("%s: installed to %s", spec.name, spec.path.name)
    return spec.path


def ensure(name: str, progress: bool = True) -> Path:
    """Return a local path to the named model, fetching it if necessary."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise ModelError(f"Unknown model {name!r}. Known: {', '.join(sorted(REGISTRY))}")
    if spec.path.exists() and spec.path.stat().st_size > 0:
        return spec.path
    archive = _download(spec, progress=progress)
    return _install(spec, archive)


def ensure_many(names: list[str], progress: bool = True) -> dict[str, Path]:
    return {n: ensure(n, progress=progress) for n in names}


def is_installed(name: str) -> bool:
    spec = REGISTRY.get(name)
    return bool(spec and spec.path.exists() and spec.path.stat().st_size > 0)


def status() -> list[dict]:
    """Installation state of every registered model."""
    rows = []
    for spec in REGISTRY.values():
        installed = is_installed(spec.name)
        rows.append(
            {
                "name": spec.name,
                "installed": installed,
                "size_mb": round(spec.path.stat().st_size / 1e6, 2) if installed else None,
                "path": str(spec.path),
                "description": spec.description,
                "credit": spec.credit,
                "tags": list(spec.tags),
            }
        )
    return rows


def verify_installed() -> list[tuple[str, bool, str]]:
    """Re-verify cached downloads against their pinned checksums."""
    results: list[tuple[str, bool, str]] = []
    for spec in REGISTRY.values():
        archive = spec.download_path
        if not archive.exists():
            results.append((spec.name, False, "download not cached"))
            continue
        actual = sha256_of(archive)
        ok = actual == spec.sha256
        results.append((spec.name, ok, "verified" if ok else f"MISMATCH {actual[:16]}..."))
    return results


def models_for_profile(profile: str) -> list[str]:
    """Which models a given runtime profile needs."""
    if profile == "accurate":
        return ["hand_landmarker", "scrfd_10g", "arcface_r50"]
    return ["hand_landmarker", "scrfd_500m", "arcface_mbf"]
