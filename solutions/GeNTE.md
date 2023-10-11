# Gender-Neutral Translation Evaluator

We propose a reference-free method to assess gender-neutral translations in 
[GeNTE](https://mt.fbk.eu/gente/).
In this repository you can find the code, the checkpoint, and instructions for conducting GeNTE evaluations
with our reference-free solution. It is a classifier finetuned on
_[UmBERTo](https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1),_
a Roberta-based Language Model trained on large Italian Corpora.


## How to run

Our classifier (`$CLASSIFIER_FOLDER`) can be downloaded at the following
[link](https://fbk.sharepoint.com/:u:/s/MTUnit/EUMZhW8AympKmpTBjqARIa4BkuwbOt-P7-Pxn_koAHvDqA?e=Dm0RpS).
It contains the checkpoint and the config file. \
To use the classifier for evaluating whether the translation of GeNTE generated by your system
in a TXT file (`$DATA`) are neutral or gendered, run the following command.
The tsv output containing the sentences the true labels and the predicted label
will be saved in a tsv file (`$OUTPUT_FILE`).

```bash
python /path/to/GeNTE/src/cli/generate.py \
        --model Musixmatch/umberto-commoncrawl-cased-v1 \
        --checkpoint $CLASSIFIER_FOLDER \
        --num-classes 2 \
        --data-file $DATA \
        --batch-size 64 \
        --max-seq-len 64 \
        --lower-case False \
        --metrics accuracy class_f1 \
        --writer tsv \
        --save-file $OUTPUT_FILE
```

## Reproducibility

To ensure reproducibility of the results reported in our paper,
we also provide the training data used to train the final classifier,
the training setup, the translations that were used in our evaluation process.

### Training Data

The data used to train our classifier have been automatically generated by
[GPT-3.5](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates).
For more information, please refer to the reference paper.
You can download these data from the following
[link](https://fbk.sharepoint.com/sites/MTUnit/Shared%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2FMTUnit%2FShared%20Documents%2Fmodels%2FClassifier%5FGNT).


### Training Setup

To replicate the training of our classifier using the synthetic data (`$TRAIN_DATA` and `$DEV_DATA`, located
in `$DATA_FOLDER`, and downloadable above) run the following command. Checkpoints will be saved in `$SAVE_FOLDER`.

```bash
python /path/to/GeNTE/src/cli/train.py \
        --model Musixmatch/umberto-commoncrawl-cased-v1 \
        --num-classes 2 \
        --data-root $DATA_FOLDER \
        --train $DATA_FOLDER/$TRAIN_DATA \
        --validation $DATA_FOLDER/$DEV_DATA \
        --save-dir $SAVE_FOLDER \
        --num-epochs 2 \
        --batch-size 64 \
        --max-seq-len 64 \
        --lower-case False \
        --shuffle True \
        --learning-rate 0.00005 \
        --epsilon 0.00000001
```

### Output Translations

We provide translations generated by DeepL and Amazon Translate and used for our evaluations.
These translations were generated from COMMON-SET, a portion of GeNTE consisting of 200 source
sentences — 100 gendered (COMMON-SET-G) and 100 neutral (COMMON-SET-N).

As the MT systems were unable to produce neutral translations for COMMON-SET-N,
three human translators manually edited the 100 COMMON-SET-N translations.
They substituted the gendered forms with neutral alternatives while keeping the
rest of the sentences unchanged. For each system, we obtained three sets of neutral
output sentences, one from each translator: Amazon-N-PEbyTransl1, Amazon-N-PEbyTransl2, Amazon-N-PEbyTransl3.

Therefore you can download the following translations at this 
[link](https://fbk.sharepoint.com/:u:/s/MTUnit/EbjRY8Tu9G1HsXWute-t33EBeK4XyGqqCHnRCodphO7DDQ?e=reEzbp):
- Amazon: `Amazon-G-original` for COMMON-SET-G; `Amazon-N-PEbyTransl1`, `Amazon-N-PEbyTransl2`, `Amazon-N-PEbyTransl3`, for COMMON-SET-N
- DeepL: `DeepL-G-original` for COMMON-SET-G; `DeepL-N-PEbyTransl1`, `DeepL-N-PEbyTransl2`, `DeepL-N-PEbyTransl3`, for COMMON-SET-*N


## How to cite

The reference paper is:
[_Hi Guys_ or _Hi Folks_? Benchmarking Gender-Neutral Machine Translation with the GeNTE Corpus](https://arxiv.org/abs/2310.05294),
accepted at EMNLP 2023.

```
@inproceedings{piergentili-etal-2023-hi,
    title = "Hi Guys or Hi Folks? Benchmarking Gender-Neutral Machine Translation with the GeNTE Corpus",
    author = "Piergentili, Andrea and 
        Savoldi, Beatrice and 
        Fucci, Dennis and 
        Negri, Matteo and 
        Bentivogli, Luisa},
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "",
    doi = "",
    pages = "",
}
```