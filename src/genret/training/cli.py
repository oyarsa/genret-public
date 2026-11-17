# pyright: basic
"""CLI commands for generative retrieval training and evaluation.

The top-level commands are defined in themed sibling modules and attached
via their register hooks:

- ``cli_train``    — train, evaluate, generate.
- ``cli_codebook`` — build-codebooks, build-leiden-codebooks,
  build-codebook-data, train-codebook, evaluate-codebook, smoke-codebook.

Shared helpers (``_load_checkpoint``, ``_validate_data_files``) live in
``_cli_helpers`` and are re-exported here for existing importers. The
``glen`` and ``classifier`` sub-apps are mounted at the bottom.

All CUDA-dependent imports are deferred to command bodies so that the CLI
module can be loaded without ``torch`` installed (e.g. during linting).
"""

import logging

from genret.training._cli_helpers import _load_checkpoint, _validate_data_files
from genret.training.cli_codebook import register_codebook_commands
from genret.training.cli_train import register_train_commands
from genret.utils import make_app

logger = logging.getLogger(__name__)

app = make_app("Train, evaluate, and generate with generative retrieval")
register_train_commands(app)
register_codebook_commands(app)


# ---------------------------------------------------------------------------
# GLEN sub-app
# ---------------------------------------------------------------------------

from genret.training.cli_glen import glen_app  # noqa: E402

app.add_typer(glen_app, name="glen")

# ---------------------------------------------------------------------------
# Classifier sub-app
# ---------------------------------------------------------------------------

from genret.training.cli_classifier import classifier_app  # noqa: E402

app.add_typer(classifier_app, name="classifier")

# ---------------------------------------------------------------------------
# MINDER-baseline sub-app
# ---------------------------------------------------------------------------

from genret.training.cli_minder import minder_app  # noqa: E402

app.add_typer(minder_app, name="minder")

__all__ = ["_load_checkpoint", "_validate_data_files", "app"]
