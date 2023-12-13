from langchain.document_loaders import ArxivLoader
from transformers import pipeline
import gradio as gr

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

def fn_one(search_query, n_docs):
  docs = ArxivLoader(query=search_query, load_max_docs=n_docs).load()
  print(search_query, n_docs)
  titles = []
  n_pairs = {}
  for i in range(n_docs):
    title = docs[i].metadata['Title']
    titles.append(title)
    n_pairs[title] = i
  return gr.Dropdown(titles), docs, n_pairs

def fn_two(choice, docs, n_pairs):
  ch = n_pairs[str(choice)]
  metadata = docs[ch].metadata
  content = docs[ch].page_content

  content = strip(content)
  sent = chunk(content)
  summarized = summarize(sent)


  out = "Date: "+ str(metadata['Published']) + "\n" + "\n Title: "+ metadata['Title'] + "\n" + "\n Authors: " + metadata['Authors'] + "\n" + "\n Summary: \n" + summarized
  return out
  return 'one output to show in the result box'


with gr.Blocks() as demo:
  with gr.Row():
    paper_name = gr.Textbox(label="Enter Paper Name/ID*")
    n_docs = gr.Dropdown(label="Number of Docs to Load", choices = [1,2,3,4,5,6,7,8,9,10])
    docs = gr.State() #gr.Textbox(label="second", visible=False)
    n_pairs = gr.State() #gr.Textbox(label="third", visible=False)
  fetch_btn = gr.Button("Fetch")
  #with gr.Row():
  label = "Papers for " + str(paper_name)
  choice = gr.Dropdown(label = label, interactive=True)
  submit_btn = gr.Button('Fetch & Summarize')
  result = gr.Textbox(label="Summary", visible=True)
  gr.Textbox(label = "Disclaimer", value="* - Please Use Paper ID (Example : 2301.10172) as it will give accurate results. Free text search can give errors sometimes While using Paper ID no need to change Number of Documents to load",
             interactive=False)

  fetch_btn.click(fn=fn_one, inputs=[paper_name, n_docs],
                    outputs=[choice, docs, n_pairs],
                    api_name="fetch")
  submit_btn.click(fn=fn_two, inputs=[choice, docs, n_pairs],
                    outputs=[result],
                    api_name="submit")
if __name__ == "__main__":
    demo.launch()