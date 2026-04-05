from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ProjectSettings:
    config_format_version: str = "1.1.0"
    model_profile: str = ""
    model_pretrained_checkpoint: str = ""
    aug_flip_horizontal_use: bool = False
    aug_flip_vertical_use: bool = False
    no_rotation: bool = False
    include_empty: bool = False
    resize: int = 512
    max_epochs: int = 80
    batch_size: int = 4
    inference_enable_openvino: bool = True

    @classmethod
    def from_dict(cls, data: dict | None, default_model_profile: str = "") -> "ProjectSettings":
        payload = data or {}
        settings = cls()
        settings.model_profile = payload.get("model.profile", default_model_profile or settings.model_profile)
        settings.model_pretrained_checkpoint = payload.get(
            "model.pretrained_checkpoint",
            settings.model_pretrained_checkpoint,
        )
        settings.aug_flip_horizontal_use = bool(payload.get("aug.flip.horizontal.use", settings.aug_flip_horizontal_use))
        settings.aug_flip_vertical_use = bool(payload.get("aug.flip.vertical.use", settings.aug_flip_vertical_use))
        settings.no_rotation = bool(payload.get("no_rotation", settings.no_rotation))
        settings.include_empty = bool(payload.get("include_empty", settings.include_empty))
        settings.resize = int(payload.get("resize", settings.resize))
        settings.max_epochs = int(payload.get("max_epochs", settings.max_epochs))
        settings.batch_size = int(payload.get("batch_size", settings.batch_size))
        settings.inference_enable_openvino = bool(
            payload.get("inference.enable_openvino", settings.inference_enable_openvino)
        )
        settings.config_format_version = str(payload.get("config_format_version", settings.config_format_version))
        return settings

    def to_dict(self) -> dict:
        return {
            "config_format_version": self.config_format_version,
            "model.profile": self.model_profile,
            "model.pretrained_checkpoint": self.model_pretrained_checkpoint,
            "aug.flip.horizontal.use": self.aug_flip_horizontal_use,
            "aug.flip.vertical.use": self.aug_flip_vertical_use,
            "no_rotation": self.no_rotation,
            "include_empty": self.include_empty,
            "resize": self.resize,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "inference.enable_openvino": self.inference_enable_openvino,
        }
