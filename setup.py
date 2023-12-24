from setuptools import find_packages, setup

# from sphinx.setup_command import BuildDoc

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="arxiv-summarizer",
    version="0.1.0",
    description="A happy toolkit for arxiv paper summarization and understanding.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArchanGhosh/Arxiv-Summarizer",
    project_urls={
        "GitHub": "https://github.com/ArchanGhosh/Arxiv-Summarizer",
        "Homepage": "https://github.com/ArchanGhosh/Arxiv-Summarizer",
    },
    author="Archan Ghosh, Arnav Das",
    author_email="gharchan@gmail.com, arnav.das88@gmail.com",
    maintainer="Archan Ghosh, Arnav Das, Debgandhar Ghosh, Subhayu Bala",
    maintainer_email="gharchan@gmail.com, arnav.das88@gmail.com, debgandhar4000@gmail.com, balasubhayu99@gmail.com",
    keywords="arxiv sdk summarization summary",
    license="Apache",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    py_modules=['__main__'],
    entry_points={
        'console_scripts': [
            'arxiv_summarizer = arxiv_summarizer.__main__:cli',
        ],
    },
    # cmdclass={"build_sphinx": BuildDoc},
    # these are optional and override conf.py settings
    command_options={
        "build_sphinx": {
            "project": ("setup.py", "arxiv-summarizer"),
            "version": ("setup.py", "0.1.0a"),
            "release": ("setup.py", "0.1"),
            "source_dir": ("setup.py", "docs"),
        }
    },
    install_requires=[
        # Arxiv & PDF
        "langchain",    # For loading Arxiv PDFs
        "arxiv",        # `langchain` dependency
        "pymupdf",      # `langchain` dependency

        # ML
        "transformers", # For Transformer models
        "gradio",       # For Web Interface
        "torch==2.1.2", # For Machine Learning

        # CLI
        "click",        # For CLI arguments
        "rich",         # For Rich Text in CLI

        # Tests
        "pytest",       # For Unit Tests
    ],
    extras_require={
        "dev": [
            "nox",
            "pytest",
        ],
        "docs": [
            "sphinx",
            "sphinxemoji",
            "pydata-sphinx-theme",
            "numpydoc",
            "sphinx_panels",
            "matplotlib",
            "Ipython",
            "sphinx-hoverxref",
        ],
    },
    classifiers=[
        # License
        "License :: OSI Approved :: GNU Affero General Public License v3",
        # Project Maturity
        "Development Status :: 1 - Planning",
        # Topic
        "Topic :: Communications",
        # Intended Audience
        "Intended Audience :: Science/Research",
        # Compatibility
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        # Python Version
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)