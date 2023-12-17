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

## Examples

For more detailed examples, refer to the [Examples](examples/) directory.

## Contributing

Contributions are welcome! Please refer to the [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the [Apache](LICENSE).
