from typing import List, Tuple, AnyStr
import torch
from transformers import pipeline

from arxiv_summarizer.utils import strip, chunk, is_arxiv_identifier
from arxiv_summarizer.fetch_paper import fetch_paper, ArxivPaper
from arxiv_summarizer.exceptions import FalseArxivId, ArxivDocumentWithdrawn, ArxivQueryReturnedEmpty 


class SummarizationModel:
    """Summarization model for generating summaries.
    """    
    def __init__(self, model, tokenizer, as_cli = False, **kwargs):
        """Initialize a summarization model.

        Args:
            model (str): The pre-trained model name or path.
            tokenizer (str): The pre-trained tokenizer name or path.
            batch_size (int, optional): Batch size for summarization. Defaults to None.
            num_workers (int, optional): Number of workers for summarization. Defaults to None.
            max_length (int, optional): Maximum length of the generated summary. Defaults to None.
            min_length (int, optional): Minimum length of the generated summary. Defaults to None.
            do_sample (bool, optional): Whether to use sampling for summary generation. Defaults to False.
            truncation (bool, optional): Whether to truncate the summary. Defaults to None.
            no_repeat_ngram_size (int, optional): Size of n-grams to avoid repetition. Defaults to None.
            num_beams (int, optional): Number of beams for beam search. Defaults to 1.
            early_stopping (bool, optional): Whether to stop generation when the stopping criterion is met. Defaults to False.
        """        
        batch_size = kwargs.pop("batch_size", None)
        num_workers = kwargs.pop("num_workers", None)
        piped_model = pipeline("summarization", model=model, tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)

        max_length = kwargs.pop("max_length", 10240)
        min_length = kwargs.pop("min_length", 1024)
        do_sample  = kwargs.pop("do_sample", False)

        truncation = kwargs.pop("truncation", None)
        no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", None)
        num_beams = kwargs.pop("num_beams", 1)
        early_stopping = kwargs.pop("early_stopping", False)
        self.kwargs = {
                # "max_length": min(max_length, len(x)),            # At most, the size of the input text
                # "min_length": min(min_length, (len(x) * 2) // 3), # At most 66.6% of the size of the input text
                "max_length": max_length,
                "min_length": min_length,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "num_beams": num_beams,
                "do_sample": do_sample, 
                "truncation": truncation,
                "early_stopping": early_stopping,
        }
        self.as_cli = as_cli

        self.model = [
            lambda x_and_kwargs: piped_model(
                x_and_kwargs[0], 
                **x_and_kwargs[1]
            )
        ]

    def add(self, layer):
        """Add a layer to the summarization model.

        Args:
            layer (callable): A callable layer to be added to the model.
        """        
        self.model.append(layer)

    def __call__(self, text):
        """Generate a summary for the input text using the summarization model.

        Args:
            text (str): The input text to be summarized.

        Returns:
            str: The generated summary.
        """

        result = []
        if self.as_cli:
            from tqdm.rich import tqdm
        else:
            from tqdm import tqdm

        for data in tqdm(text, unit = "sentence"):
            __data__ = data
            for layer in self.model:
                    __data__ = layer(
                            (
                                __data__,
                                {
                                    "max_length": min(self.kwargs["max_length"], len(data)),            # At most, the size of the input text
                                    "min_length": min(self.kwargs["min_length"], (len(data) * 2) // 3), # At most 66.6% of the size of the input text
                                    "do_sample": self.kwargs["do_sample"], 
                                    "truncation": self.kwargs["truncation"],
                                    "no_repeat_ngram_size": self.kwargs["no_repeat_ngram_size"],
                                    "num_beams": self.kwargs["num_beams"],
                                    "early_stopping": self.kwargs["early_stopping"],
                                }
                            )
                    )
            result.append(
                __data__
            )

        return result

class ArxivSummarizer:
    """ArXiv document summarizer.

    Returns:
        str: The generated summary for the ArXiv document.

    Example:
        The `ArxivSummarizer` class can be used with different configurations:

        1. With an existing `SummarizationModel`:

        ```python
        summarizer_model = SummarizationModel(model="Falconsai/text_summarization", tokenizer="Falconsai/text_summarization", max_length=256)
        arxiv_summarizer = ArxivSummarizer(summarizer=summarizer_model)
        summary = arxiv_summarizer(arxiv_id="1234.6789")
        ```

        2. Using a pre-trained model by name:

        ```python
        arxiv_summarizer = ArxivSummarizer(model="facebook/bart-large-cnn", max_length=256)
        summary = arxiv_summarizer(arxiv_id="1234.6789")
        ```

        3. Using default models based on GPU availability:

        ```python
        arxiv_summarizer = ArxivSummarizer()
        summary = arxiv_summarizer(arxiv_id="1234.6789")
        ```
        
    Args:
        summarizer (SummarizationModel): The pre-initialized summarization model.
        model (str, optional): The pre-trained model name or path. Defaults to `Falconsai/text_summarization`_ for CPU and `facebook/bart-large-cnn`_ when CUDA is available.
        max_length (int, optional): Maximum length of the generated summary. Defaults to 256.
        min_length (int, optional): Minimum length of the generated summary. Defaults to 64.
        do_sample (bool, optional): Whether to use sampling for summary generation. Defaults to False.

    Todo:
        * For module TODOs
        * You have to also use ``sphinx.ext.todo`` extension

    .. _Falconsai/text_summarization:
    https://huggingface.co/Falconsai/text_summarization

    .. _facebook/bart-large-cnn:
    https://huggingface.co/facebook/bart-large-cnn
    """

    def __init__(self, summarizer: SummarizationModel = None, model: str = None, max_length: int = 256, min_length: int = 64, do_sample: bool = False):

        if summarizer:
            self.model = summarizer
        else:
            if model is None:
                if torch.cuda.is_available():
                    model = "facebook/bart-large-cnn"
                else:
                    model = "Falconsai/text_summarization"

            self.model = SummarizationModel(
                model=model, 
                tokenizer=model, 
                
                max_length=max_length, 
                min_length=min_length, 
                do_sample=do_sample,
                truncation = True,
                num_beams=3,
                early_stopping=True,
            )

    def __call__(self, arxiv_id:str) -> Tuple[ArxivPaper, AnyStr]:
        """Generate a summary for the specified ArXiv document.

        Args:
            arxiv_id (str): The ArXiv identifier or search query for the document.

        Raises:
            FalseArxivId: Raised when the provided arxiv identifier `arxiv_id` is not found.
            ArxivQueryReturnedEmpty: Raised when the arxiv query `arxiv_id` returns no results.


        Returns:
            Tuple[ArxivPaper, AnyStr]: A tuple containing an ArxivPaper object representing the fetched document and the generated summary.
        """
        # metadata, page_content = doc_loader(arxiv_id)
        papers = list(fetch_paper(arxiv_id, max_docs=3))
        if len(papers) == 0:
            if is_arxiv_identifier(arxiv_id):
                raise FalseArxivId(f"Cannot find the arxiv id `{arxiv_id}`")
            else:
                raise ArxivQueryReturnedEmpty(f"Cannot find any arxiv entry containing `{arxiv_id}`")

        paper: ArxivPaper = papers[0]

        content = strip(paper.content)
        sentence = chunk(content)
        return (paper, self.model(sentence))
