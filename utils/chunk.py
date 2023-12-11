def chunk(content):
    '''We are currently using a manual chunking method that combines 10 sentences at one time.
        Since the model context length is around 512 we are assume that 10 sentences would create a context length of that is between 600+-100
    '''
    content = clip(content)

    sent = []
    c = 0
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