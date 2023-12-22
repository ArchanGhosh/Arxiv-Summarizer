from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Iterator
from urllib.request import urlretrieve
from tempfile import NamedTemporaryFile
import warnings

from arxiv_summarizer.arxiv_wrapper import ArxivAPIWrapper
from langchain.document_loaders import ArxivLoader 
from arxiv.arxiv import Result

@dataclass(slots=True)
class ArxivPaper:
    arxiv_id: str
    name: str
    authors: List[str]
    summary: str
    published: datetime
    primary_category: str
    categories: List[str]
    pdf_url: str
    links: List[Tuple[str, str, str, str]]
    journal_ref: str
    doi: str

    # content: str
    _content: int = field(repr=False, init=False, default = None)


    @property
    def content(self, ) -> str:
        if self._content:
            return self._content

        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF package not found, please install it with "
                "`pip install pymupdf`"
            )

        with NamedTemporaryFile(mode = "r", delete = True, prefix="arxiv_", suffix=".pdf") as arxiv_pdf:
            try:
                doc_file_name, msg = urlretrieve(self.pdf_url, arxiv_pdf.name)
                with fitz.open(doc_file_name) as doc_file:
                    self._content: str = "".join(page.get_text() for page in doc_file)
            except:
                warnings.warn(f"Cannot fetch arxiv:{self.arxiv_id}. Possibly withdrawn.")
                self._content = None
                return None

        return self._content
    
    @staticmethod
    def from_result(result: Result):
        paper = ArxivPaper(
            arxiv_id = result.entry_id.split("/")[-1],
            name = result.title,
            authors = [
                author.name for author in result.authors
            ],
            summary = result.summary,
            published = result.published,
            primary_category = result.primary_category,
            categories = result.categories,
            pdf_url = result.pdf_url,
            links = [
                (link.title, link.href, link.rel, link.content_type) for link in result.links
            ],
            journal_ref = result.journal_ref,
            doi = result.doi
        )
        return paper

def fetch_paper(query, max_docs = 1) -> Iterator[ArxivPaper]:
    """Fetches ArXiv papers based on the given query.

    Args:
        query (str): The search query for ArXiv papers.
        max_docs (int, optional): The maximum number of documents to fetch. Defaults to 1.

    Returns:
        List[ArxivPaper]: A list of ArXivPaper objects representing the fetched papers.

    Yields:
        Iterator[ArxivPaper]: A list of ArXivPaper objects representing the fetched papers.

    Note:
        The function uses the ArxivAPIWrapper to fetch paper summaries and metadata,
        and then converts the results into ArxivPaper objects.

    Raises:
        ImportError: If the PyMuPDF package (required for extracting content from PDFs)
        is not installed.

    Warnings:
        UserWarning: If the specified `max_docs` exceeds 50, it defaults to 50 documents.
        UserWarning: If a paper cannot be fetched (possibly withdrawn), a warning is issued.

    Example:
        .. code-block:: python

            from arxiv_summarizer.fetch_paper import fetch_paper

            papers = fetch_paper("machine learning", max_docs=5)
            for paper in papers:
                print(paper.name, paper.authors, paper.published)
        
    """    
    query = query.lower()

    # TODO: This check won't be necessary anymore, as the "content" 
    # is now being fetched, only when needed.
    if (max_docs > 50):
        warnings.warn("Max docs exceeded 50, defaulting to 50 documents only")

    max_docs = min(50, max_docs)

    client = ArxivAPIWrapper(
        doc_content_chars_max = None, 
        top_k_results = max_docs, 
        load_max_docs = max_docs,
        load_all_available_meta = True
    )

    results = client.get_summaries_as_docs(query = query)

    for res in results:
        yield ArxivPaper.from_result(res)



        

