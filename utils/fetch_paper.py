from langchain.document_loaders import ArxivLoader 
def fetch_paper(search_query = "unet", load_max_docs = 1):
    search_query = search_query.lower()
    docs = ArxivLoader(query="unet", load_max_docs=load_max_docs, load_all_available_meta=True).load()
    title_id_pairs = {}
    for i in range(len(docs)):
        paper_id = docs[i].metadata["entry_id"].split("/")[-1]
        print(paper_id)
        title_id_pairs[docs[i].metadata["Title"]] = paper_id
    #paper_idx = [paper_id[-1] for i in docs: paper_id=i.metadata['entry_id'].split("/")]
    print(title_id_pairs)

fetch_paper(search_query = "Nerf", load_max_docs = 10)