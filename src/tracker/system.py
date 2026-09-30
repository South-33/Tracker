"""Self-contained causal person-tracker system and bundle format."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from ultralytics.utils import IterableSimpleNamespace

from tracker.causal import CausalPersonTracker
from tracker.model import TrackingYOLO
from tracker.runtime import CausalTrackerRuntime


BUNDLE_FORMAT = "causal-person-tracker-v1"


@dataclass(frozen=True)
class TrackerConfig:
    """Validated runtime configuration for the current tracking candidate."""

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.45
    track_buffer: int = 30
    match_thresh: float = 0.80
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    gmc_max_corners: int = 100
    proximity_thresh: float = 0.50
    appearance_thresh: float = 0.80
    min_detections_for_reid: int = 10
    detector_confidence: float = 0.05
    owner_alpha: float = 0.20
    owner_average_base_cost_budget: float = 0.00025
    coast_frames: int = 1
    coast_min_active_tracks: int = 10

    def association_args(self, device: torch.device) -> IterableSimpleNamespace:
        return IterableSimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            fuse_score=self.fuse_score,
            gmc_method=self.gmc_method,
            proximity_thresh=self.proximity_thresh,
            appearance_thresh=self.appearance_thresh,
            with_reid=True,
            model="auto",
            device=str(device),
        )

    def metadata(self) -> dict:
        return asdict(self)


def _resolve_detector(identity_path: Path, detector_reference: str | Path) -> Path:
    detector = Path(detector_reference)
    if detector.is_absolute():
        return detector
    candidates = [
        Path.cwd() / detector,
        identity_path.parents[2] / detector
        if len(identity_path.parents) >= 3
        else identity_path.parent / detector,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(detector_reference)


def build_tracker_bundle(
    identity_checkpoint: str | Path,
    owner_checkpoint: str | Path,
    output: str | Path,
    *,
    config: TrackerConfig | None = None,
) -> dict:
    """Package perception, identity, association, and config into one artifact."""
    identity_path = Path(identity_checkpoint).resolve()
    owner_path = Path(owner_checkpoint).resolve()
    identity = torch.load(
        identity_path,
        map_location="cpu",
        weights_only=True,
    )
    owner = torch.load(
        owner_path,
        map_location="cpu",
        weights_only=False,
    )
    detector_path = _resolve_detector(identity_path, identity["detector"])
    model = TrackingYOLO(
        detector_path,
        identity["embedding_dim"],
    ).cpu().eval()

    bundle = {
        "format": BUNDLE_FORMAT,
        "config": (config or TrackerConfig()).metadata(),
        "detector": model.detector,
        "embedding_dim": int(identity["embedding_dim"]),
        "embedding": identity["embedding"],
        "owner": owner,
        "source": {
            "identity_checkpoint": identity_path.name,
            "owner_checkpoint": owner_path.name,
            "detector_reference": str(identity["detector"]),
            "identity_objective": identity.get("objective"),
            "owner_objective": owner.get("objective"),
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    return bundle


class PersonTracker:
    """One causal tracker object: frame in, active anonymous tracks out."""

    def __init__(
        self,
        bundle: str | Path,
        *,
        device: str | torch.device = "cuda",
        config: TrackerConfig | None = None,
    ):
        self.device = torch.device(device)
        self.bundle_path = Path(bundle)
        checkpoint = torch.load(
            self.bundle_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("format") != BUNDLE_FORMAT:
            raise ValueError(
                f"unsupported tracker bundle format: {checkpoint.get('format')!r}"
            )
        config_values = dict(checkpoint["config"])
        # Historical v1 bundles predate the explicit detector floor and
        # crowded one-frame coast. Preserve their original runtime behavior.
        config_values.setdefault("detector_confidence", 0.10)
        config_values.setdefault("coast_frames", 0)
        config_values.setdefault("coast_min_active_tracks", 0)
        bundled_config = TrackerConfig(**config_values)
        self.config = config or bundled_config

        self.model = TrackingYOLO(
            checkpoint["detector"],
            checkpoint["embedding_dim"],
        ).to(self.device).eval()
        self.model.embedding.load_state_dict(checkpoint["embedding"])

        self.association = CausalPersonTracker(
            self.config.association_args(self.device),
            checkpoint["owner"],
            owner_alpha=self.config.owner_alpha,
            average_base_cost_budget=self.config.owner_average_base_cost_budget,
            gmc_max_corners=self.config.gmc_max_corners,
            coast_frames=self.config.coast_frames,
            coast_min_active_tracks=self.config.coast_min_active_tracks,
        )
        self.runtime = CausalTrackerRuntime(
            self.model,
            self.association,
            device=self.device,
            with_reid=True,
            feature_mode="trained",
            min_detections_for_reid=self.config.min_detections_for_reid,
            detector_confidence=self.config.detector_confidence,
        )

    @torch.inference_mode()
    def step(self, frame):
        return self.runtime.step(frame)

    def reset(self) -> None:
        self.association.reset()
        self.runtime.frames = 0
        self.runtime.reid_frames = 0
        self.runtime.reid_detections = 0
        self.runtime.compute_seconds = 0.0

    def stats(self) -> dict:
        return self.runtime.stats()
