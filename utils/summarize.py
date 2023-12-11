from model_selector import model_selector
from transformers import pipeline

def summarize(sent):
    ''' This is the main function that summarizes the contents of the paper. Model is initialized automatically based on CUDA availability.
    '''
    tokenizer_str, model_str = model_selector()

    summarizer = pipeline("summarization", model=model_str, tokenizer = tokenizer_str)

    summarized = ""
    for i in sent:
        s = summarizer(i, max_length=256, min_length=64, do_sample=False)
        summarized = summarized + s[0]['summary_text'] +"\n"

    return summarized
