# pyright: basic
"""CLI commands for GLEN-style generative retrieval.

Registered under ``training glen`` in the main CLI. The commands are
defined in themed sibling modules and attached here via their
``register()`` hooks:

- ``cli_glen_index``  — build-docids, smooth-bullets, build-title-index.
- ``cli_glen_train``  — train, train-distill, train-split-loss.
- ``cli_glen_bullets``— train-bullets, train-marginal, train-contrastive.
- ``cli_glen_paper``  — train-paper-title, train-paper-marginal, train-titles.
- ``cli_glen_eval``   — evaluate, smoke.
- ``cli_glen_c2t`` / ``cli_glen_usage`` / ``glen_paper_eval`` — extra commands.
"""

from __future__ import annotations

import logging

from genret.training.cli_glen_bullets import register as _register_bullets
from genret.training.cli_glen_c2t import register as _register_c2t
from genret.training.cli_glen_eval import register as _register_eval
from genret.training.cli_glen_index import register as _register_index
from genret.training.cli_glen_paper import register as _register_paper
from genret.training.cli_glen_train import register as _register_train
from genret.training.cli_glen_usage import register as _register_usage
from genret.training.glen_paper_eval import register as _register_paper_eval
from genret.utils import make_app

logger = logging.getLogger(__name__)

glen_app = make_app("GLEN-style generative retrieval commands")
_register_index(glen_app)
_register_train(glen_app)
_register_bullets(glen_app)
_register_paper(glen_app)
_register_eval(glen_app)
_register_paper_eval(glen_app)
_register_c2t(glen_app)
_register_usage(glen_app)

__all__ = ["glen_app"]
