import streamlit as st
from langchain.document_loaders import ArxivLoader 
from transformers import pipeline



def strip(content):
  content = str(content)
  #print(content)
  content = content.split("\n")
  content = " ".join(content)
  #print(content)

  return content

def clip(content):
  loc_intro = content.find("Introduction")
  loc_refer = content.rfind("Reference")
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

  print("-----Clipping content between Intro and References--------")

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
  model_str = "Falconsai/text_summarization"
  tokenizer_str = "Falconsai/text_summarization"

  summarizer = pipeline("summarization", model=model_str, tokenizer = tokenizer_str)


  summarized = ""
  for i in sent:
    s = summarizer(i, max_length=256, min_length=64, do_sample=False)
    summarized = summarized + s[0]['summary_text'] +"\n"

  return summarized

def doc_load(search_query="default", n_docs=1):
  if search_query == "default":
    return [" ",  " "], [" ",  " "], [" ",  " "]
  try :
    print("-------searching Paper----------")
    docs = ArxivLoader(query=search_query, load_max_docs=n_docs).load()
    titles = []
    n_pairs = {}
    for i in range(n_docs):
      title = docs[i].metadata['Title']
      titles.append(title)
      n_pairs[title] = i
    return titles, docs, n_pairs
  except Exception as e:
    print("--------ERROR while Trying to Search Paper-------------")
    print(e)
    

def run(choice, docs, n_pairs):
  ch = n_pairs[choice]
  st.text("Fetching Metadata")
  print("-----fetching metadata-------------")
  metadata = docs[ch].metadata
  content = docs[ch].page_content 

  print("----stripping new lines----------")
  content = strip(content)
  print("-----------chunking content--------------")
  sent = chunk(content)
  st.text("Chunking Text....")
  st.text("🤔 Shortening text...")
  print("----summarizing content---------")
  summarized = summarize(sent)


  out = "Date: "+ str(metadata['Published']) + "\n" + "\n Title: "+ metadata['Title'] + "\n" + "\n Authors: " + metadata['Authors'] + "\n" + "\n Summary: \n" + summarized
  return out

st.title("ArXiV Summarizer")
titles = []
with st.form(key="search_form"):
    col1, col2 = st.columns(2)
    with col1:
        search_query = st.text_input("Search Using Paper ID or Name*")
    with col2:
        n_docs = st.selectbox(label="Number of Documents to Load", options=(1, 2, 3, 4, 5, 6, 7, 8, 9 ,10))
    submit = st.form_submit_button(label="Search")
if submit:
  c = "Fetching Papers 🤔 "
  st.write(c)
try:
  titles, docs, n_pairs = doc_load(search_query=search_query, n_docs=n_docs)
except Exception as e:  
  print(e)
if titles:
  c = "Papers Fetched 🤩 "
  st.write(c)
else:
  c = "Error while Fetching Papers 😥 Please Check ID or Name"
  st.write(c)

label = "Papers for " + search_query
with st.form(key="paper_form"):
  paper_name = st.selectbox(label=label, options=titles)
  submit_paper = st.form_submit_button(label="Fetch Paper & Summarize")
  print(submit_paper)
if submit_paper:
  st.text("Reading Document.... 📄 ")
  output = run(paper_name, docs, n_pairs)
  st.text_area(label = "Summary", value=output, height = 650)

st.text('''* - Please Use Paper ID (Example : 2301.10172) as it will give accurate results. 
          Free text search can give errors sometimes''')
st.text("While using Paper ID no need to change Number of Documents to load")