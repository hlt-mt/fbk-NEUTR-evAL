# fbk-NEUTR-evAL

This repository includes solutions related to the **evAL**uation of gender **NEUTR**ality, 
specifically focusing on metrics designed to assess whether a given text 
contains inclusive language with respect to gender, in both monolingual and cross-lingual contexts. \
The repository is maintained by the [Machine Translation unit](https://mt.fbk.eu/) of **FBK**.

## Installation

To clone and install our repository, execute the following commands:

```bash
git clone https://github.com/hlt-mt/fbk-NEUTR-evAL.git
cd fbk-NEUTR-evAL
pip install -e .
```

## Available Solutions

You can find a dedicated README for each available evaluation solution in the `solutions` directory,
along with Bibtex citations referencing the respective papers:

 - [Neutral Classifier for GeNTE v1](solutions/GeNTE.md) from [EMNLP 2023] **_Hi Guys_ or _Hi Folks_? Benchmarking Gender-Neutral Machine Translation with the GeNTE Corpus**
 - [Neutral Classifier for GeNTE v2](solutions/GeNTE.md) from [EACL 2024] **A Prompt Response to the Demand for Automatic Gender-Neutral Translation**
 - [Evaluation Script for Neo-GATE](solutions/Neo-GATE.md) from [EAMT 2024] **Enhancing Gender-Inclusive Machine Translation with Neomorphemes and Large Language Models**



## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
