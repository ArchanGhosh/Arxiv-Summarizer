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