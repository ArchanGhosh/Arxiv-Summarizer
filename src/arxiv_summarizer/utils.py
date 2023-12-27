import re
from transformers import pipeline
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

def is_arxiv_identifier(query: str, raise_exception:bool = False) -> bool:
    """Check if a query is an arxiv identifier."""
    arxiv_identifier_pattern = r"\d{2}(0[1-9]|1[0-2])\.\d{4,5}(v\d+|)|\d{7}.*"
    for query_item in query.split():
        match_result = re.match(arxiv_identifier_pattern, query_item)
        if not match_result:
            if raise_exception:
                raise InvalidArxivId(f"Not a valid arxiv id `{query}`")
            return False
        assert match_result is not None
        if not match_result.group(0) == query_item:
            return False
    return True


