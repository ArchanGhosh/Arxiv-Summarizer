import nox

SOURCE_FILES = (
    "setup.py",
    "noxfile.py",
    "src/arxiv_summarizer/",
    "src/arxiv_summarizer.examples/",
)


@nox.session()
def format(session):
    session.install("black", "isort")

    session.run("isort", "--profile=black", *SOURCE_FILES)
    session.run("black", "--target-version=py39", *SOURCE_FILES)
