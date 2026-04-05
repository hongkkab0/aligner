from aligner_engine.model_profiles import get_default_model_profile_id
from aligner_engine.project_settings import ProjectSettings


CONFIG_FORMAT_VERSION = "1.1.0"


def get_default_project_config():
    settings = ProjectSettings.from_dict({}, default_model_profile=get_default_model_profile_id())
    settings.config_format_version = CONFIG_FORMAT_VERSION
    return settings.to_dict()
