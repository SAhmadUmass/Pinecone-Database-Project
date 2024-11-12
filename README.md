# Pinecone Vector Database Project

This project demonstrates how to create a vector database using Pinecone and utilize it for document retrieval and question-answering tasks. The main components of the project include:

* Indexing a PDF document (an essay by Peter Thiel) into a Pinecone vector database
* Generating embeddings using SentenceTransformers
* Implementing a question-answering system using OpenAI's GPT-3.5-turbo model
* Enhancing answer generation with citations using Pydantic and the Instructor library

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [How It Works](#how-it-works)
* [Results and Observations](#results-and-observations)
* [Potential Improvements](#potential-improvements)

## Overview

The primary goal of this project is to explore the capabilities of vector databases and language models in retrieving and generating information from documents. By indexing an essay by Peter Thiel into Pinecone and utilizing embeddings, we can query the database to find relevant sections and use a language model to generate answers based on those sections.

## Features

* **PDF Document Processing**: Convert a PDF document into text and split it into individual pages for indexing
* **Vector Embeddings**: Generate embeddings for each page using the all-MiniLM-L6-v2 model from SentenceTransformers
* **Pinecone Indexing**: Create and manage a Pinecone index to store and query the embeddings
* **Question Answering**: Use OpenAI's GPT-3.5-turbo model to generate answers to user queries based on the indexed content
* **Citation System**: Incorporate a citation mechanism to provide direct quotes from the source document, reducing the chance of hallucinations

## Usage

### Indexing the Document

Run the `index_documents.py` script to process the PDF and index it into Pinecone:

```bash
python index_documents.py
```

This script will:
* Read the PDF and split it into pages
* Generate embeddings for each page
* Upsert the embeddings into the Pinecone index

### Asking Questions

Run the `main.py` script to query the Pinecone index and generate answers:

```bash
python main.py
```

You can modify the `query` variable in `main.py` to ask different questions.

## Project Structure

* `index_documents.py`: Script to process the PDF and index it into Pinecone
* `main.py`: Main script to perform question-answering using the indexed data
* `requirements.txt`: List of required Python packages
* `.env`: Environment variables file (not included; you need to create this)
* `README.md`: Project documentation

## How It Works

### Document Processing
* **PDF to Text**: The PDF is read page by page using PyPDF2. Each page's text is extracted and stored with metadata such as page number and source.

### Embedding Generation
* **Sentence Embeddings**: SentenceTransformer's all-MiniLM-L6-v2 model generates embeddings for each page's content.

### Indexing
* **Pinecone Index**: Embeddings along with their metadata are upserted into a Pinecone index named `pinecone-database-test`.

### Querying
* **User Query**: The user's question is embedded using the same embedding model
* **Similarity Search**: The Pinecone index is queried to retrieve the top relevant documents based on cosine similarity

### Answer Generation
* **Contextual Answering**: The retrieved documents are combined into a context string
* **OpenAI GPT-3.5-turbo**: The context and user question are passed to the GPT-3.5-turbo model to generate an answer
* **Structured Output**: The Instructor library and Pydantic models enforce a structured response, ensuring each fact includes citations

### Citation System
* **Regex Matching**: Regular expressions are used to find exact matches of the quoted phrases in the context
* **Pydantic Validation**: The Fact and QuestionAnswer models validate the presence of citations, reducing the likelihood of hallucinations

## Results and Observations

* **Model Performance**: Smaller open-source models like Flan-T5-XL and LLaMA-3.2-3B struggled to generate coherent responses due to their size. GPT-3.5-turbo provided significantly better results with minimal adjustments.
* **Reducing Hallucinations**: By incorporating Pydantic and the Instructor library and enforcing a citation system, the model's tendency to hallucinate was reduced. The system requires direct quotes found with regex, ensuring the answers are grounded in the source material.

## Potential Improvements

* **Expand Dataset**: Index more documents to create a richer knowledge base
* **Advanced Chunking**: Implement smarter text chunking methods to improve context retrieval
* **Fine-tuning**: Experiment with fine-tuning models on specific datasets to improve answer quality
* **User Interface**: Develop a web or command-line interface for users to input queries more conveniently

## Acknowledgments

* The citation system implementation was inspired by and adapted from the Instructor library's exact citations example, which can be seen [here](https://python.useinstructor.com/examples/exact_citations/)
