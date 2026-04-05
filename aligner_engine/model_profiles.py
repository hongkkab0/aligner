from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    train_config_relpath: str
    pretrained_filename: str
    deploy_config_relpath: str
    description: str = ""

    def train_config_path(self, root_path: str) -> str:
        return str(Path(root_path) / self.train_config_relpath)

    def deploy_config_path(self, root_path: str) -> str:
        return str(Path(root_path) / self.deploy_config_relpath)

    def default_pretrained_path(self, root_path: str) -> str:
        return str(Path(root_path) / "aligner_pretrained" / self.pretrained_filename)


_PROFILES: Dict[str, ModelProfile] = {
    "rotated_rtmdet_tiny": ModelProfile(
        id="rotated_rtmdet_tiny",
        label="RTMDet Tiny",
        train_config_relpath="aligner_engine/mm_rotate_det/dice/configs/rotated_rtmdet/rotated_rtmdet_tiny.py",
        pretrained_filename="rotated_rtmdet_tiny-3x-dota-6f6f2ff9.pth",
        deploy_config_relpath="aligner_engine/mm_rotate_det/dice/configs/deploy/mmrotate/rotated-detection-dice-openvino-dynamic-512x512.py",
        description="Fastest startup and lowest resource usage.",
    ),
    "rotated_rtmdet_s": ModelProfile(
        id="rotated_rtmdet_s",
        label="RTMDet Small",
        train_config_relpath="aligner_engine/mm_rotate_det/dice/configs/rotated_rtmdet/rotated_rtmdet_s.py",
        pretrained_filename="rotated_rtmdet_s-3x-dota-11f6ccf5.pth",
        deploy_config_relpath="aligner_engine/mm_rotate_det/dice/configs/deploy/mmrotate/rotated-detection-dice-openvino-dynamic-512x512.py",
        description="Balanced default profile.",
    ),
    "rotated_rtmdet_m": ModelProfile(
        id="rotated_rtmdet_m",
        label="RTMDet Medium",
        train_config_relpath="aligner_engine/mm_rotate_det/dice/configs/rotated_rtmdet/rotated_rtmdet_m.py",
        pretrained_filename="rotated_rtmdet_m-3x-dota-beeadda6.pth",
        deploy_config_relpath="aligner_engine/mm_rotate_det/dice/configs/deploy/mmrotate/rotated-detection-dice-openvino-dynamic-512x512.py",
        description="Higher accuracy with more memory usage.",
    ),
    "rotated_rtmdet_l": ModelProfile(
        id="rotated_rtmdet_l",
        label="RTMDet Large",
        train_config_relpath="aligner_engine/mm_rotate_det/dice/configs/rotated_rtmdet/rotated_rtmdet_l.py",
        pretrained_filename="rotated_rtmdet_l-3x-dota-be17e453.pth",
        deploy_config_relpath="aligner_engine/mm_rotate_det/dice/configs/deploy/mmrotate/rotated-detection-dice-openvino-dynamic-512x512.py",
        description="Best accuracy, slowest and heaviest profile.",
    ),
}

_DEFAULT_PROFILE_ID = "rotated_rtmdet_s"


def get_default_model_profile_id() -> str:
    return _DEFAULT_PROFILE_ID


def iter_model_profiles() -> Iterable[ModelProfile]:
    return _PROFILES.values()


def list_model_profiles() -> List[ModelProfile]:
    return list(iter_model_profiles())


def get_model_profile(profile_id: str | None) -> ModelProfile:
    if not profile_id:
        return _PROFILES[_DEFAULT_PROFILE_ID]
    return _PROFILES.get(profile_id, _PROFILES[_DEFAULT_PROFILE_ID])


def resolve_pretrained_checkpoint(root_path: str, profile_id: str | None, explicit_path: str = "") -> str:
    explicit = explicit_path.strip()
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return str(candidate)

    profile = get_model_profile(profile_id)
    default_path = Path(profile.default_pretrained_path(root_path))
    if default_path.exists():
        return str(default_path)

    return ""
