# SimCKP

![architecture](/simckp.png)

Source code for our *Findings of EMNLP 2023* paper [SimCKP: Simple Contrastive Learning of Keyphrase Representations](https://github.com/brightjade/SimCKP).

## Requirements

- Python (tested on 3.8.18)
- CUDA (tested on 11.8)
- PyTorch (tested on 2.0.1)
- Transformers (tested on 4.34.0)
- nltk (tested on 3.7)
- fasttext (tested on 0.9.2)
- datasketch (tested on 1.6.4)
- wandb
- tqdm

## Setting Up POS Tagger

To use Stanford POS tagger,
1. Install OpenJDK 1.8 in server:
    ```bash
    >> sudo apt-get update
    >> sudo apt-get install openjdk-8-jdk
    ```
2. Install Stanford CoreNLP (may be a different version; v4.4.0 was used in paper):
    ```bash
    >> wget http://nlp.stanford.edu/software/stanford-corenlp-latest.zip
    >> unzip stanford-corenlp-latest.zip && cd stanford-corenlp-4.4.0
    ```
3. Run Java server in the Stanford CoreNLP directory:
    ```bash
    >> nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
        -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
        -status_port 9000 -port 9000 -timeout 15000 1> /dev/null 2>&1 &
    ```
- Ref: https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK
- Install OpenJDK without sudo: https://stackoverflow.com/questions/2549873/installing-jdk-without-sudo


## Preprocessing

Before running the preprocessing script, make sure the raw data is present as follows:

```
[working directory]
 |-- data/
 |    |-- raw_data/
 |    |    |-- inspec_test.json
 |    |    |-- kp20k_test.json
 |    |    |-- kp20k_train.json
 |    |    |-- kp20k_valid.json
 |    |    |-- krapivin_test.json
 |    |    |-- nus_test.json
 |    |    |-- semeval_test.json
 |    |-- stanford-corenlp-4.4.0/
 |    |-- lid.176.bin
```

Raw dataset can be downloaded [here](https://github.com/memray/OpenNMT-kpg-release). `lid.176.bin` is a pre-trained fasttext model for detecting languages in text and can be downloaded [here](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin).

To preprocess raw data, run

```bash
>> python cleanse.py
```

Then, annotate part-of-speech tags by running

```bash
>> python pos_tagging.py
```

To skip data cleansing and POS tagging, just download our [POS tagged data](https://drive.google.com/file/d/1Owh3_5EmZ0AAcBEzACwzDOfc_QvH7PXZ/view?usp=sharing).

## Training

Train the extractor (stage 1):

```bash
>> bash scripts/train_extractor.sh
```

Extract present keyphrases and generate absent candidates:

```bash
>> bash scripts/extract.sh
```

Train the reranker (stage 2):

```bash
>> bash scripts/train_ranker.sh
```

Rerank absent keyphrases:

```bash
>> bash scripts/rank.sh
```

## Evaluation

Evaluate both present and absent keyphrase predictions:

```bash
>> bash scripts/evaluate.sh
```

## Citation

If you make use of this code in your work, please kindly cite our paper:

```bibtex
@inproceedings{choi-etal-2023-simckp,
    title = "{S}im{CKP}: Simple Contrastive Learning of Keyphrase Representations",
    author = "Choi, Minseok  and
      Gwak, Chaeheon  and
      Kim, Seho  and
      Kim, Si  and
      Choo, Jaegul",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.199",
    doi = "10.18653/v1/2023.findings-emnlp.199",
    pages = "3003--3015",
    abstract = "Keyphrase generation (KG) aims to generate a set of summarizing words or phrases given a source document, while keyphrase extraction (KE) aims to identify them from the text. Because the search space is much smaller in KE, it is often combined with KG to predict keyphrases that may or may not exist in the corresponding document. However, current unified approaches adopt sequence labeling and maximization-based generation that primarily operate at a token level, falling short in observing and scoring keyphrases as a whole. In this work, we propose SimCKP, a simple contrastive learning framework that consists of two stages: 1) An extractor-generator that extracts keyphrases by learning context-aware phrase-level representations in a contrastive manner while also generating keyphrases that do not appear in the document; 2) A reranker that adapts scores for each generated phrase by likewise aligning their representations with the corresponding document. Experimental results on multiple benchmark datasets demonstrate the effectiveness of our proposed approach, which outperforms the state-of-the-art models by a significant margin.",
}
```
