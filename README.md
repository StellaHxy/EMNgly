# EMNGly: predicting N-linked glycosylation sites using the language models for feature extraction

**Xiaoyang Hou**, Yu Wang, Dongbo Bu, Yaojun Wang<sup>∗</sup>  and Shiwei Sun<sup>∗</sup> <br>
[Paper link on Bioinformatics](https://academic.oup.com/bioinformatics/article/39/11/btad650/7335841?login=false)

## Environments
```bash
git clone git@github.com:StellaHxy/EMNgly.git
conda create -n EMNgly python=3.7
conda activate EMNgly
pip install -r requirements.txt
```

## Datasets

- Download the N-GlycositeAltas dataset and [N-GlyDE dataset](https://github.com/dukkakc/DeepNGlyPred) under folder `data/`:
    ```
    ├── data
    │   ├── N-GlycositeAltas
    │   │     ├── N-GAltas_seq_train.csv
    │   │     ├── N-GAltas_seq_test.csv
    │   │     └── N-GAltas_structure
    │   ├── N-GlyDE
    │   │     ├── N-GlyDE_seq_train.csv
    │   │     ├── N-GlyDE_seq_test.csv
    │   │     └── N-GlyDE_structure
    ```

## Usage

### Train the classifier of EMNgly on N-GlycositeAltas dataset:
```
bash scritps/N-GlycositeAltas_train.sh
```

### Predict  N-linked glycosylation sites of N-GlycositeAltas dataset:
```
bash scritps/N-GAltas_test.sh 
```
