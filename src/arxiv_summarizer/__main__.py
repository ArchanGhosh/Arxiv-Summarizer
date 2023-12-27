from logging import CRITICAL, DEBUG, ERROR, FATAL, INFO, WARN, WARNING

import click
from rich.console import Console
from transformers.utils import logging

from arxiv_summarizer.fetch_paper import ArxivPaper
from arxiv_summarizer.summarizer import ArxivSummarizer, SummarizationModel

logging.set_verbosity(ERROR)
logger = logging.get_logger(__name__)

console = Console()


@click.command()
@click.option(
    "--model",
    default="Falconsai/text_summarization",
    help="Huggingface id for the model. Defaults to `Falconsai/text_summarization`",
)
@click.option(
    "--tokenizer",
    default="Falconsai/text_summarization",
    help="Huggingface id for the tokenizer.  Defaults to `Falconsai/text_summarization`",
)
@click.option(
    "--token_length",
    default=512,
    help="Maximum token length to use while summarization. Defaults to 512",
)
@click.argument("arxiv_id")
def main(
    arxiv_id: str,
    model: str = "Falconsai/text_summarization",
    tokenizer: str = "Falconsai/text_summarization",
    token_length: int = 512,
):
    summarizer = ArxivSummarizer(
        summarizer=SummarizationModel(
            model=model,
            tokenizer=tokenizer,
            max_length=token_length,
            min_length=token_length // 3,
            do_sample=False,
            truncation=True,
            num_beams=3,
            early_stopping=True,
            as_cli=True,
        )
    )
    paper, data = summarizer(arxiv_id)

    console.print(paper.name, justify="center")
    console.rule(f"[ {paper.published} ]")
    console.print(" ".join([s[0]["summary_text"] for s in data]), justify="full")


if __name__ == "__main__":
    main()
