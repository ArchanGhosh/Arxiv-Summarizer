from arxiv_summarizer.exceptions import InvalidArxivId, FalseArxivId, ArxivDocumentWithdrawn

def chunk(content):
    '''We are currently using a manual chunking method that combines 10 sentences at one time.
        Since the model context length is around 512 we are assume that 10 sentences would create a context length of that is between 600+-100
    '''
    abst, content = clip(content)

    sentences = []
    c = 0
    k = ""
    content = content.split(". ")
    for i in range(len(content)):
        k = k + content[i] + ". "
        c = c+1
        if c == 10:
            sentences.append(k)
            c = 0
            k = ""
        elif i==len(content)-1:
            sentences.append(k)

    return sentences

def clip(content):
    ''' Using the Clip Function we are trying to clip all the contents that are above the Introduction mainly abstract, title and authors &
        all references as they are not necessary for summarization
    '''
    loc_abst = content.find("Abstract")
    loc_intro = content.find("Introduction")
    loc_refer = content.rfind("References")
    if loc_intro !=-1:
        if loc_refer !=-1:
            return content[loc_abst:loc_intro], content[loc_intro:loc_refer]
        else:
            return content[loc_abst:loc_intro], content[loc_intro:]
            print("Warning: Paper Doesn't have a References Title, may lead to overlap of references in summary")
    else:
        print("Warning: Paper Doesn't Have an Introduction Title, these may lead to overlap of summarization")

from langchain.document_loaders import ArxivLoader 

def doc_loader(search_query):
    '''The purpose of this function is to load the documents from Arxiv'''
    '''We are using the Arxiv Loader under Langchain library, which intern invokes the Arxiv API'''

    docs = ArxivLoader(query=search_query, load_max_docs=1).load()
    if len(docs) == 0:
        raise FalseArxivId(f"Cannot find the arxiv id `{search_query}`")
    return docs[0].metadata, docs[0].page_content

import torch

def model_selector():
    '''This function allows us to choose between 2 models, lightweight model to be used with CPU, and a heavyweight model to be used if GPU is available
    '''
    if torch.cuda.is_available():
        tokenizer_str = "facebook/bart-large-cnn"
        model_str = "facebook/bart-large-cnn"
        return tokenizer_str, model_str
    else:
        tokenizer_str = "Falconsai/text_summarization"
        model_str = "Falconsai/text_summarization"
        return tokenizer_str, model_str

def strip(content):
    ''' The purpose of this function is to strip the contents of the page of line breaks, thats helps us in better summarization, 
    as line breaks can cause the model to not function properly'''

    content = str(content)
    content = content.split("\n")
    content = " ".join(content)

    return content

from transformers import pipeline

def summarize(sent):
    ''' This is the main function that summarizes the contents of the paper. Model is initialized automatically based on CUDA availability.
    '''
    tokenizer_str, model_str = model_selector.model_selector()

    summarizer = pipeline("summarization", model=model_str, tokenizer = tokenizer_str)

    summarized = ""
    for i in sent:
        s = summarizer(i, max_length=256, min_length=64, do_sample=False)
        summarized = summarized + s[0]['summary_text'] +"\n"

    return summarized




