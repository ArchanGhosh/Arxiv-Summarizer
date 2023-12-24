# Arxiv Summarizer

The `ArxivSummarizer` is a Python class designed for summarizing ArXiv documents using Hugging Face's Transformers library. It can be configured with a custom `SummarizationModel` or with pre-trained models based on user preferences.

## Table of Contents

- [Arxiv Summarizer](#arxiv-summarizer)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [As CLI](#as-cli)
    - [With Custom SummarizationModel](#with-custom-summarizationmodel)
    - [With Pre-trained Model by Name](#with-pre-trained-model-by-name)
    - [With Default Models](#with-default-models)
    - [Fetching a list of papers](#fetching-a-list-of-papers)
  - [Examples](#examples)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

Make sure you have Python 3.8 or later installed. Install the required dependencies using the following command:

```bash
pip install .
```

## Usage

### As CLI

Arxiv Summarizer can be easily used as a CLI tool to get papers summarized... 

```sh
$ python3 -m arxiv_summarizer 1234.56789v1
```

### With Custom SummarizationModel

If you have a custom `SummarizationModel` and `Tokenizer`, you can use them with `ArxivSummarizer` directly:

```python
from arxiv_summarizer import SummarizationModel, ArxivSummarizer

# Initialize your custom SummarizationModel
custom_model = SummarizationModel(
    model="your_custom_model", 
    tokenizer="your_custom_tokenizer", 
    max_length=512, do_sample=True
)

# Initialize ArxivSummarizer with your custom model
summarizer = ArxivSummarizer(summarizer=custom_model)

# Generate a summary
summary = summarizer(arxiv_id="1234.5678")
print(summary)
```

### With Pre-trained Model by Name

You can use a pre-trained model from Hugging Face's model hub by specifying its name:

```python
from arxiv_summarizer import ArxivSummarizer

# Initialize ArxivSummarizer with a pre-trained model by name
summarizer = ArxivSummarizer(model="facebook/bart-large-cnn")

# Generate a summary
summary = summarizer(arxiv_id="1234.5678")
print(summary)
```

### With Default Models

If you don't provide a specific model, `ArxivSummarizer` will use default models based on GPU availability:

```python
from arxiv_summarizer import ArxivSummarizer

# Initialize ArxivSummarizer with default models
summarizer = ArxivSummarizer()

# Generate a summary
summary = summarizer(arxiv_id="1234.5678")
print(summary)
```

### Fetching a list of papers

First we can search a list of papers directly using the `fetch_paper()` definition.
```python
from rich.progress import Progress
from rich.console import Console
from rich.table import Table

from typing import List
from arxiv_summarizer.fetch_paper import fetch_paper, ArxivPaper

# Get the list of papers
papers = fetch_paper("Yoshua Bengio", max_docs=15)
results : List[ArxivPaper] = [paper for paper in papers]

print(f"{len(results)} Papers Found !!!")
```

This will load the papers, their metadata and their summaries. Now we can download the content of the paper and show the progress using a progressbar from `rich`.
```python
# Download the papers
with Progress() as progress:
    task = progress.add_task("[cyan] Downloading content...", total = len(results))

    for index, paper in enumerate(results):
        progress.update(task, advance=1, description=f"Downloading content for paper {paper.arxiv_id}")
        _ = results[index].content # This will download the content automatically.
```

Once all the content has been downloaded, we can display the content in a tabular structure using a `rich` Table.
```python
# Print the data
console = Console()

table = Table(show_header=True, header_style="bold magenta")
table.add_column("ID", style="dim")
table.add_column("Title", style="dim")
table.add_column("Authors", style="dim")
table.add_column("Content Size", style="dim")

for entry in results:
    entry:ArxivPaper
    table.add_row(entry.arxiv_id, entry.name, ", ".join(entry.authors), str(len(entry.content)))

console.print(table)
```

```
15 Papers Found !!!
Downloading content for paper 1203.4416v1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ ID           ┃ Title                                      ┃ Authors                                    ┃ Content Size ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 1206.5533v2  │ Practical recommendations for              │ Yoshua Bengio                              │ 134815       │
│              │ gradient-based training of deep            │                                            │              │
│              │ architectures                              │                                            │              │
│ 1207.4404v1  │ Better Mixing via Deep Representations     │ Yoshua Bengio, Grégoire Mesnil, Yann       │ 31767        │
│              │                                            │ Dauphin, Salah Rifai                       │              │
│ 1305.0445v2  │ Deep Learning of Representations: Looking  │ Yoshua Bengio                              │ 121365       │
│              │ Forward                                    │                                            │              │
│ 1212.2686v1  │ Joint Training of Deep Boltzmann Machines  │ Ian Goodfellow, Aaron Courville, Yoshua    │ 13806        │
│              │                                            │ Bengio                                     │              │
│ 1703.07718v1 │ Independently Controllable Features        │ Emmanuel Bengio, Valentin Thomas, Joelle   │ 19385        │
│              │                                            │ Pineau, Doina Precup, Yoshua Bengio        │              │
│ 1211.5063v2  │ On the difficulty of training Recurrent    │ Razvan Pascanu, Tomas Mikolov, Yoshua      │ 50908        │
│              │ Neural Networks                            │ Bengio                                     │              │
│ 1206.5538v3  │ Representation Learning: A Review and New  │ Yoshua Bengio, Aaron Courville, Pascal     │ 194906       │
│              │ Perspectives                               │ Vincent                                    │              │
│ 1207.0057v1  │ Implicit Density Estimation by Local       │ Yoshua Bengio, Guillaume Alain, Salah      │ 35635        │
│              │ Moment Matching to Sample from             │ Rifai                                      │              │
│              │ Auto-Encoders                              │                                            │              │
│ 1305.6663v4  │ Generalized Denoising Auto-Encoders as     │ Yoshua Bengio, Li Yao, Guillaume Alain,    │ 33769        │
│              │ Generative Models                          │ Pascal Vincent                             │              │
│ 1311.6184v4  │ Bounding the Test Log-Likelihood of        │ Yoshua Bengio, Li Yao, Kyunghyun Cho       │ 23711        │
│              │ Generative Models                          │                                            │              │
│ 1510.02777v2 │ Early Inference in Energy-Based Models     │ Yoshua Bengio, Asja Fischer                │ 26477        │
│              │ Approximates Back-Propagation              │                                            │              │
│ 1509.05936v2 │ STDP as presynaptic activity times rate of │ Yoshua Bengio, Thomas Mesnard, Asja        │ 22030        │
│              │ change of postsynaptic activity            │ Fischer, Saizheng Zhang, Yuhuai Wu         │              │
│ 1103.2832v1  │ Autotagging music with conditional         │ Michael Mandel, Razvan Pascanu, Hugo       │ 47698        │
│              │ restricted Boltzmann machines              │ Larochelle, Yoshua Bengio                  │              │
│ 2007.15139v2 │ Deriving Differential Target Propagation   │ Yoshua Bengio                              │ 63661        │
│              │ from Iterating Approximate Inverses        │                                            │              │
│ 1203.4416v1  │ On Training Deep Boltzmann Machines        │ Guillaume Desjardins, Aaron Courville,     │ 20531        │
│              │                                            │ Yoshua Bengio                              │              │
└──────────────┴────────────────────────────────────────────┴────────────────────────────────────────────┴──────────────┘
```

## Examples

For more detailed examples, refer to the [Examples](examples/) directory.

## Contributing

Contributions are welcome! Please refer to the [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the [Apache](LICENSE).
