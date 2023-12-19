from langchain.document_loaders import ArxivLoader 

from arxiv_summarizer.arxiv_wrapper import ArxivAPIWrapper
import warnings


def fetch_paper(query, max_docs=1, load_all_metadata=True):
    query=query.lower()
    if (max_docs>50):
        warnings.warn("Max docs exceeded 50, defaulting to 50 documents only")
    max_docs = min(50, max_docs)

    client = ArxivAPIWrapper(doc_content_chars_max = None, top_k_results=max_docs, load_all_available_meta = load_all_metadata)

    #docs = ArxivLoader(query=query, load_max_docs=max_docs, load_all_available_meta = load_all_metadata).get_summaries_as_docs()
    summaries = client.get_summaries_as_docs(query=query)

    for doc in summaries:
        yield (doc.metadata['Title'], doc.metadata['entry_id'].split("/")[-1])

def fetch_content(query):
    query = query.lower()
    docs = ArxivLoader(query=query, load_max_docs=1, load_all_available_meta = True).load()

    return docs[0].metadata, docs[0].page_content

print(list(fetch_paper(query="transformer", max_docs=20)))




        

