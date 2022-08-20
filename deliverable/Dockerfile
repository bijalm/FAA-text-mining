FROM python:3.10.4-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY binarizers ./binarizers
COPY label-maps ./label-maps
COPY models ./models
COPY results ./results
COPY vectorizers ./vectorizers
COPY inference.py ./inference.py
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
COPY ./req.txt /req.txt

RUN pip3 install --no-cache-dir -r /req.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader omw-1.4