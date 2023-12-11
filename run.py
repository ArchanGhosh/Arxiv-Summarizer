from utils import chunk, strip, clip, doc_loader, model_selector, summarize 

def run(search_query):
    ''' This function returns basic metadata of the paper along with the summarized contents
    '''
    metadata, page_content = doc_loader.doc_loader(search_query)
    content = strip.strip(page_content)
    sent = chunk.chunk(content)
    summarized = summarize.summarize(sent)
    

    print(f"Published : {metadata['Published']} ")
    print(f"Title : {metadata['Title']} ")
    print(f"Authors : {metadata['Authors']}")
    print(f"Summarised Text : {summarized}")

if __name__ == '__main__':

    #print("Working")
    print("Please Provide a Arxiv Index number or Paper Title as free text search to fetch paper")
    print("Warning: Free Text Search is not always accurate") 
    search_query = input("Enter here: ")
    # print(search_query)
    print("Paper fetched begining text summarisation")
    #search_query = "MTTN"
    run(search_query) 