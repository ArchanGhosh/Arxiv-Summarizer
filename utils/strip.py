def strip(content):
    ''' The purpose of this function is to strip the contents of the page of line breaks, thats helps us in better summarization, 
    as line breaks can cause the model to not function properly'''

    content = str(content)
    content = content.split("\n")
    content = " ".join(content)

    return content