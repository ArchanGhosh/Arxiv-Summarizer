from typing import List

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from arxiv_summarizer.fetch_paper import ArxivPaper, fetch_paper

if __name__ == "__main__":
    papers = fetch_paper("Yoshua Bengio", max_docs=15)

    results: List[ArxivPaper] = [paper for paper in papers]

    print(f"{len(results)} Papers Found !!!")

    with Progress() as progress:
        task = progress.add_task("[cyan] Downloading content...", total=len(results))

        for index, paper in enumerate(results):
            progress.update(
                task,
                advance=1,
                description=f"Downloading content for paper {paper.arxiv_id}",
            )
            c = results[index].content  # This will download the content automatically.

    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim")
    table.add_column("Title", style="dim")
    table.add_column("Authors", style="dim")
    table.add_column("Content Size", style="dim")

    for entry in results:
        entry: ArxivPaper
        table.add_row(
            entry.arxiv_id,
            entry.name,
            ", ".join(entry.authors),
            str(len(entry.content)),
        )

    console.print(table)
