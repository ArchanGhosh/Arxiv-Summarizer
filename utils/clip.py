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