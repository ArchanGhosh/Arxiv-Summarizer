from arxiv_summarizer import ArxivSummarizer

if __name__ == "__main__":
    # Initialize ArxivSummarizer with default models
    summarizer = ArxivSummarizer()

    # Generate a summary
    paper, summary = summarizer(arxiv_id="2106.11298v1")
    print(summary)
