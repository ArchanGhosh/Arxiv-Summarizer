'''
Implementation of the Arxiv Summarizer as an independent module
'''

from langchain.document_loaders import ArxivLoader 
from transformers import pipeline

def doc_loader(search_query):
    '''The purpose of this function is to load the documents from Arxiv'''
    '''We are using the Arxiv Loader under Langchain library, which intern invokes the Arxiv API'''

    try :
        docs = ArxivLoader(query=search_query, load_max_docs=1).load()
        return docs[0].metadata, docs[0].page_content
    except Exception as e:
        print(e)

def strip(content):
    ''' The purpose of this function is to strip the contents of the page of line breaks, thats helps us in better summarization, 
    as line breaks can cause the model to not function properly'''

    content = str(content)
    content = content.split("\n")
    content = " ".join(content)

    return content

def clip(content):
    ''' Using the Clip Function we are trying to clip all the contents that are above the Introduction mainly abstract, title and authors &
        all references as they are not necessary for summarization
    '''
    loc_intro = content.find("Introduction")
    loc_refer = content.rfind("References")
    if loc_intro !=-1:
        if loc_refer !=-1:
            content = content[loc_intro:loc_refer]
        else:
            content = content[loc_intro:]
            print("Warning: Paper Doesn't have a References Title, may lead to overlap of references in summary")
    else:
        print("Warning: Paper Doesn't Have an Introduction Title, these may lead to overlap of summarization")

    return content

def chunk(content):
    '''We are currently using a manual chunking method that combines 10 sentences at one time.
        Since the model context length is around 512 we are assume that 10 sentences would create a context length of that is between 600+-100
    '''
    content = clip(content)

    sent = []
    c= 0
    k = ""
    content = content.split(". ")
    for i in range(len(content)):
        k = k + content[i] + ". "
        c = c+1
        if c == 10:
            sent.append(k)
            c = 0
            k = ""
        elif i==len(content)-1:
            sent.append(k)

    return sent

def summarize(sent):
    ''' This is the main function that summarizes the contents of the paper. Currently we are using the following model: https://huggingface.co/Falconsai/text_summarization
        This model is both light and efficient interms of CPU configuration. As part of continuous improvement we are planning to have this available with a GPU efficient model 
        as well. It will be initialized automatically based on CUDA availability but at the same time it can be invoked manually as well.
    '''

    model_str = "Falconsai/text_summarization"
    tokenizer_str = "Falconsai/text_summarization"

    summarizer = pipeline("summarization", model=model_str, tokenizer = tokenizer_str)


    summarized = ""
    for i in sent:
        s = summarizer(i, max_length=256, min_length=64, do_sample=False)
        summarized = summarized + s[0]['summary_text'] +"\n"

    return summarized

def run(search_query):
    ''' This function returns basic metadata of the paper along with the summarized contents
    '''
    metadata, page_content = doc_loader(search_query)
    content = strip(page_content)
    sent = chunk(content)
    summarized = summarize(sent)
    

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