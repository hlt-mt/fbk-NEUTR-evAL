# Gender-Inclusive Translation Evaluation

This repository contains the necessary code to perform gender-inclusive evaluation with 
[Neo-GATE](https://huggingface.co/datasets/FBK-MT/Neo-GATE). 

## How to run
Ensure you have downloaded Neo-GATE and installed this repository following the instructions in
the [README](../README.md).

You can then run the evaluation script with the following command:
```bash
neogate_eval --testset $NEO-GATE \
            --outputs $SYSTEM_OUTPUTS \
            --neomorphemes $NEOMORPHEMES_FILE 
```
**Parameters:**
- `$NEO-GATE` is the path to `Neo-GATE.tsv`.
- `$SYSTEM_OUTPUTS` is the path to the plain text files containing the outputs to evaluate, 
one per line, in the same order as the corresponding source sentences appear in Neo-GATE.
- `$NEOMORPHEMES_FILE` is the path to a plain text file containing the characters or symbols used
as neomorphemes, one per line, with no extra punctuation.

**Example:**
```bash
neogate_eval --testset /path/to/Neo-GATE.tsv \
            --outputs /path/to/system_outputs.txt \
            --neomorphemes /path/to/neomorphemes.txt
```

The script will print all the metrics used to evaluate with Neo-GATE to the standard output.

If you also need a detailed evaluation, add the `--export_tsv` flag to the command:
```bash
neogate_eval --testset $NEO-GATE \
            --outputs $SYSTEM_OUTPUTS \
            --neomorphemes $NEOMORPHEMES_FILE  \
            --export_tsv
```
The script will export a `${SYSTEM_OUTPUTS}_eval.tsv` file containing all the terms used to compute
the metrics.

### Standalone script usage

To avoid installing all the repository dependencies, you can download the 
[`neogate_evaluation.py`](../src/fbk_neutreval/cli/neogate_evaluation.py) script and run it
standalone with the following syntax:
```bash
python3 neogate_evaluation.py --testset $NEO-GATE \
            --outputs $SYSTEM_OUTPUTS \
            --neomorphemes $NEOMORPHEMES_FILE  \
            --export_tsv
```
The parameters and the output will be the same as those described above.

## How to cite

If you use Neo-GATE and/or the associated evaluation code, please cite the paper
[Enhancing Gender-Inclusive Machine Translation with Neomorphemes and Large Language 
Models](https://arxiv.org/abs/2405.08477),
published at EAMT 2024:

```
@inproceedings{piergentili-etal-2024-enhancing,
      title={{Enhancing Gender-Inclusive Machine Translation with Neomorphemes and Large Language Models}},
      author={Piergentili, Andrea and 
      Savoldi, Beatrice and 
      Negri, Matteo and 
      Bentivogli, Luisa},
      booktitle = "Proceedings of the 25th Annual Conference of the European Association for Machine Translation",
      month = jun,
      year="2024",
      address = "Sheffield, United Kingdom",
      publisher = "European Association for Machine Translation",
      pages = "298--312",
}
```
