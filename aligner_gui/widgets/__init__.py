# Compatibility shim — aligner_gui.widgets is superseded by aligner_gui.shared.
from aligner_gui.shared import (  # noqa: F401
    graph_widget,
    image_panel,
    log_widget,
    progress_general_dialog,
    progress_list_dialog,
)
