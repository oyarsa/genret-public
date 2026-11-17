"""Top-level CLI for genret, composing all subcommands."""

from dotenv import load_dotenv

from genret.docid import app as docid_app
from genret.download.cli import app as download_app
from genret.explorer.cli import app as explorer_app
from genret.log_config import setup_logging
from genret.pipeline.cli import app as pipeline_app
from genret.plot import app as plot_app
from genret.processing.cli import app as paper_processing_app
from genret.processing.judge_cli import app as judge_app
from genret.processing.term_extraction import app as term_extraction_app
from genret.s2.cli import app as paper_counts_app
from genret.utils import make_app

app = make_app("Genret - S2ORC paper processing toolkit")

app.add_typer(docid_app, name="docid")
app.add_typer(download_app, name="download")
app.add_typer(explorer_app, name="explorer")
app.add_typer(judge_app, name="judge")
app.add_typer(paper_counts_app, name="paper-counts")
app.add_typer(paper_processing_app, name="paper-processing")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(plot_app, name="plot")
app.add_typer(term_extraction_app, name="term-extraction")

try:
    from genret.training.cli import app as training_app

    app.add_typer(training_app, name="training")
except ImportError:
    pass


def main() -> None:
    """Entry point for the genret CLI."""

    load_dotenv()
    setup_logging()
    app()


if __name__ == "__main__":
    main()
