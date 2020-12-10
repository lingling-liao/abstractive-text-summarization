# Abstractive Text Summarization
This tutorial shows how to build a Abstractive Text Summarizer for generating news article headlines.

## Official Docker images for TensorFlow

Docker pull command:

```
docker pull tensorflow/tensorflow:2.3.0-gpu-jupyter
```

Running containers:

```
docker run --gpus all -p 6006:6006 -p 8888:8888 -v [local]:/tf -itd tensorflow/tensorflow:2.3.0-gpu-jupyter
```

## Using Transformer

Install pandas & xlrd:

```
pip install pandas
pip install xlrd
```

Training a model:

```
python abstractive_text_summarization_using_transformers.py
```

## Using Hugging Face

Install packages

```
pip install transformers
pip install datasets

# OpenAI GPT original tokenization workflow
pip install spacy ftfy==4.4.3
python -m spacy download en
```
