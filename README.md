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

## Download sample data

Multi-News, consists of news articles and human-written summaries of these articles from the site newser.com. Each summary is professionally written by editors and includes links to the original articles cited.

https://drive.google.com/drive/folders/1A5S0LJhg5vPx13hmg67F1OsuLwRFn8ss?usp=sharing

## Using Transformer

Install pandas:

```
pip install pandas
```

Training a model:

```
python abstractive_text_summarization_using_transformers.py
```
