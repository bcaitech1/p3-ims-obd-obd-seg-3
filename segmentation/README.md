# Segmentation Project for NAVER AI BoostCamp 



# VumbleBot - BaselineCode  <!-- omit in toc -->

- [DEMO](#demo)
  - [Reader](#reader)
  - [Retrieval](#retrieval)
- [TIPS](#tips)
- [Simple Use](#simple-use)
  - [Dependencies](#dependencies)
  - [Install packages](#install-packages)
  - [Predict](#predict)
  - [Reader train/validation](#reader-trainvalidation)
  - [Retriever train/validation](#retriever-trainvalidation)
  - [Reader, Retriever validation](#reader-retriever-validation)
  - [Make dataset](#make-dataset)
- [File Structure](#file-structure)
  - [Input](#input)
  - [Baseline_code](#baseline_code)
- [Json File Example](#json-file-example)
- [Usage](#usage)
  - [Usage: Train](#usage-train)
    - [READER Train](#reader-train)
    - [READER Result](#reader-result)
    - [RETRIEVER Train](#retriever-train)
    - [RETRIEVER Result](#retriever-result)
  - [Usage: Predict](#usage-predict)
    - [Predict result](#predict-result)
- [TDD](#tdd)


## File Structure  

### Input
  
```
input/
│ 
├── config/ - strategies
│   ├── ST01.json
│   └── ...
│
├── checkpoint/ - checkpoints&predictions (strategy_alias_seed)
│   ├── ST01_base_00
│   │   ├── checkpoint-500
│   │   └── ...
│   ├── ST01_base_95
│   └── ...
│ 
├── data/ - competition data
│   ├── dummy_data/
│   ├── train_data/
│   └── test_data/
│
├─── embed/ - embedding caches of wikidocs.json
│   ├── TFIDF
│   │   ├── TFIDF.bin
│   │   └── embedding.bin
│   ├── BM25
│   │   ├── BM25.bin
│   │   └── embedding.bin
│   ├── ATIREBM25
│   │   ├── ATIREBM25.bin
│   │   ├── ATIREBM25_idf.bin
│   │   └── embedding.bin
│   ├── DPRBERT
│   │   ├── DPRBERT.pth
│   │   └── embedding.bin
│   └── ATIREBM25_DPRBERT
│       └── classifier.bin
│
└── keys/ - secret keys or tokens
    └── secrets.json
```