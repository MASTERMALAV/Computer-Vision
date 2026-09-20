"""The enrolled-identity gallery: storage, matching, and threshold analysis.

Matching uses the *maximum* similarity against any stored sample of a person
rather than the similarity to their mean embedding. Averaging a set of poses
produces a vector that resembles none of them, so a person enrolled across a
range of head angles matches their own average worse than any individual
sample. Taking the maximum keeps each pose usable.

Acceptance needs two conditions, not one:

* the best similarity must clear ``match_threshold`` - is this anyone we know?
* it must beat the runner-up by ``margin_threshold`` - is it *clearly* this
  person rather than a coin flip between two lookalikes?

Embeddings from different recognition models are not comparable, so the model
name is stored with the gallery and a mismatch is refused rather than silently
producing garbage similarities.

**Privacy:** these vectors are biometric data. The file lives under ``data/``,
which is git-ignored, and nothing here ever transmits it anywhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ROOT
from ..logsetup import get_logger
from .embedder import EMBEDDING_DIM, normalize, similarity_matrix

log = get_logger("face.gallery")

GALLERY_VERSION = 2


@dataclass
class MatchResult:
    """Outcome of comparing one probe embedding against the gallery."""

    name: str  # matched identity, or "" when rejected
    score: float  # best cosine similarity found
    margin: float  # how far the best beat the runner-up
    accepted: bool
    runner_up: str = ""
    reason: str = ""

    @property
    def is_known(self) -> bool:
        return self.accepted and bool(self.name)

    def describe(self) -> str:
        if self.accepted:
            return f"{self.name} ({self.score:.3f})"
        if self.name:
            return f"{self.name}? rejected: {self.reason} ({self.score:.3f})"
        return f"unknown ({self.score:.3f})"


class GalleryError(RuntimeError):
    pass


class Gallery:
    """Enrolled people and their face embeddings."""

    def __init__(self, model: str = "", path: str | Path | None = None) -> None:
        self.model = model
        self.path = Path(path) if path else None
        self._people: dict[str, np.ndarray] = {}
        self._enrolled_at: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    @property
    def names(self) -> list[str]:
        return sorted(self._people)

    @property
    def is_empty(self) -> bool:
        return not self._people

    def count(self, name: str) -> int:
        return int(self._people[name].shape[0]) if name in self._people else 0

    def embeddings(self, name: str) -> np.ndarray:
        return self._people.get(name, np.zeros((0, EMBEDDING_DIM), dtype=np.float32))

    def total_samples(self) -> int:
        return sum(int(v.shape[0]) for v in self._people.values())

    # ------------------------------------------------------------------ #
    def add(self, name: str, embeddings: np.ndarray) -> int:
        """Add samples for a person. Returns the new sample count."""
        name = name.strip()
        if not name:
            raise GalleryError("an identity needs a non-empty name")
        arr = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if arr.shape[1] != EMBEDDING_DIM:
            raise GalleryError(f"expected {EMBEDDING_DIM}-d embeddings, got {arr.shape[1]}")
        arr = np.stack([normalize(v) for v in arr])

        if name in self._people:
            self._people[name] = np.vstack([self._people[name], arr])
        else:
            self._people[name] = arr
            self._enrolled_at[name] = time.time()
        return self.count(name)

    def remove(self, name: str) -> bool:
        existed = name in self._people
        self._people.pop(name, None)
        self._enrolled_at.pop(name, None)
        return existed

    # ------------------------------------------------------------------ #
    def match(
        self,
        embedding: np.ndarray,
        threshold: float,
        margin: float = 0.0,
    ) -> MatchResult:
        """Compare one probe against every enrolled person."""
        if self.is_empty:
            return MatchResult("", 0.0, 0.0, False, reason="gallery is empty")

        probe = normalize(embedding).reshape(1, -1)
        scores: list[tuple[str, float]] = []
        for name, samples in self._people.items():
            sims = similarity_matrix(probe, samples)[0]
            scores.append((name, float(sims.max())))
        scores.sort(key=lambda item: item[1], reverse=True)

        best_name, best_score = scores[0]
        runner_name, runner_score = scores[1] if len(scores) > 1 else ("", -1.0)
        gap = best_score - runner_score if runner_score > -1.0 else 1.0

        if best_score < threshold:
            return MatchResult(
                "", best_score, gap, False,
                runner_up=best_name,
                reason=f"below threshold {threshold:.2f}",
            )
        if gap < margin:
            return MatchResult(
                "", best_score, gap, False,
                runner_up=runner_name,
                reason=f"too close to {runner_name} (gap {gap:.3f} < {margin:.2f})",
            )
        return MatchResult(best_name, best_score, gap, True, runner_up=runner_name)

    # ------------------------------------------------------------------ #
    def analyse(self) -> dict:
        """Genuine vs impostor similarity distributions, for threshold choice.

        Rather than adopting a threshold from a paper evaluated on a different
        model and population, this measures the separation actually achieved by
        *this* model on *these* people - which is the number that governs
        whether the gate works.
        """
        genuine: list[float] = []
        impostor: list[float] = []

        for name, samples in self._people.items():
            if samples.shape[0] > 1:
                sims = similarity_matrix(samples, samples)
                # Off-diagonal only: a sample matched against itself is 1.0.
                iu = np.triu_indices(sims.shape[0], k=1)
                genuine.extend(sims[iu].tolist())
            for other, other_samples in self._people.items():
                if other <= name:
                    continue
                impostor.extend(similarity_matrix(samples, other_samples).reshape(-1).tolist())

        def summarise(values: list[float]) -> dict | None:
            if not values:
                return None
            a = np.asarray(values, dtype=np.float64)
            return {
                "n": int(a.size),
                "min": float(a.min()),
                "p01": float(np.percentile(a, 1)),
                "p05": float(np.percentile(a, 5)),
                "mean": float(a.mean()),
                "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)),
                "max": float(a.max()),
            }

        g, i = summarise(genuine), summarise(impostor)
        report: dict = {"genuine": g, "impostor": i, "people": len(self._people)}

        if g and i:
            # Put the threshold midway between the worst genuine pair and the
            # best impostor pair, and say plainly whether they overlap at all.
            report["separation"] = g["p05"] - i["p95"]
            report["suggested_threshold"] = round((g["p05"] + i["p95"]) / 2.0, 3)
            report["overlap"] = g["p05"] <= i["p95"]
        elif g:
            # One person enrolled: no impostor data, so the only defensible
            # suggestion is one that admits their own worst-case variation.
            report["suggested_threshold"] = round(max(g["p05"] - 0.05, 0.2), 3)
            report["overlap"] = False
            report["note"] = (
                "only one identity enrolled, so there is no impostor data. "
                "The suggested threshold reflects your own pose variation only "
                "and is not evidence that someone else would be rejected."
            )
        return report

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise GalleryError("no gallery path given")
        if not target.is_absolute():
            target = ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, np.ndarray] = {
            "__version__": np.array([GALLERY_VERSION]),
            "__model__": np.array([self.model]),
            "__names__": np.array(self.names, dtype=object),
        }
        for name in self.names:
            payload[f"emb::{name}"] = self._people[name]
            payload[f"ts::{name}"] = np.array([self._enrolled_at.get(name, 0.0)])

        np.savez_compressed(target, **payload)
        self.path = target
        log.info(
            "gallery saved: %d identities, %d samples -> %s",
            len(self._people), self.total_samples(), target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path, model: str = "", strict_model: bool = True) -> "Gallery":
        target = Path(path)
        if not target.is_absolute():
            target = ROOT / target
        if not target.exists():
            return cls(model=model, path=target)

        with np.load(target, allow_pickle=True) as data:
            stored_model = str(data["__model__"][0]) if "__model__" in data else ""
            gallery = cls(model=stored_model or model, path=target)
            names = [str(n) for n in data["__names__"]] if "__names__" in data else []
            for name in names:
                key = f"emb::{name}"
                if key in data:
                    gallery._people[name] = data[key].astype(np.float32)
                ts = f"ts::{name}"
                gallery._enrolled_at[name] = float(data[ts][0]) if ts in data else 0.0

        if strict_model and model and stored_model and stored_model != model:
            raise GalleryError(
                f"This gallery was enrolled with '{stored_model}' but the current "
                f"profile uses '{model}'. Embeddings from different recognition "
                "models are not comparable, so matching would be meaningless.\n"
                f"  Either switch back to the matching profile, or re-enrol with:\n"
                f"    argus enroll <name> --reset"
            )
        log.info(
            "gallery loaded: %d identities, %d samples (model %s)",
            len(gallery._people), gallery.total_samples(), stored_model or "unknown",
        )
        return gallery

    def summary(self) -> list[dict]:
        return [
            {
                "name": name,
                "samples": self.count(name),
                "enrolled_at": self._enrolled_at.get(name, 0.0),
            }
            for name in self.names
        ]
