from mmengine import HOOKS, join_path, HistoryBuffer

from mmengine.runner import Runner

from mmengine.hooks import RuntimeInfoHook
from mmengine.version import __version__


# This is a version not including get_git_hash() function
@HOOKS.register_module()
class DiceRuntimeInfoHook(RuntimeInfoHook):

    def before_run(self, runner) -> None:
        """Update metainfo.

        Args:
            runner (Runner): The runner of the training process.
        """
        metainfo = dict(
            cfg=runner.cfg.pretty_text,
            seed=runner.seed,
            experiment_name=runner.experiment_name,
            mmengine_version=__version__)
        runner.message_hub.update_info_dict(metainfo)

        self.last_loop_stage = None

