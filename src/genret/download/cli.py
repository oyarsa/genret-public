"""CLI for downloading datasets, metadata, and PDFs."""

from genret.download.datasets import app as datasets_app
from genret.download.metadata import app as metadata_app
from genret.download.pdfs import app as pdfs_app
from genret.download.supervisor import app as supervisor_app
from genret.utils import make_app

app = make_app("Download datasets, metadata, and PDFs")

app.add_typer(datasets_app, name="datasets")
app.add_typer(metadata_app, name="metadata")
app.add_typer(pdfs_app, name="pdfs")
app.add_typer(supervisor_app, name="supervisor")
