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
    │   │     ├── train.csv
    │   │     ├── test.csv
    │   │     └── pdb_file
    │   ├── N-GlyDE
    │   │     ├── train.csv
    │   │     ├── test.csv
    │   │     └── pdb_file
    ```

## Usage

### Quick predict N-linked glycosylation sites of N-GlycositeAltas dataset:
Download the [checkpoint](https://drive.google.com/drive/folders/1cCCIw5HIgtBylf2oVFgSEaVNE1LGAN4g?usp=sharing) of N-GlyAltas_classifier under folder `checkpoints/`:

    ├── checkpoints
    │       └── N-GlyAltas_classifier.pkl
    ├── data
    ├── log
    ├── model
    ├── scripts
    ├── main.py
    ├── predict.py

```
python predict.py --mode=test_features --data_path=../data/N-GlycositeAltas --ckpt_path=./checkpoints/N-GlyAltas_classifier.pkl 
```

### Train the classifier of EMNgly on N-GlycositeAltas dataset:
```
bash scritps/get_N-GlycositeAltas_train_features.sh
python main.py --mode=train --data_path=../data/N-GlycositeAltas --output_path=./checkpoints/N-GlyAltas_classifier.pkl
```

### Predict  N-linked glycosylation sites of N-GlycositeAltas dataset:
```
bash scritps/get_N-GlycositeAltas_test_features.sh
python predict.py --mode=test --data_path=../data/N-GlycositeAltas --ckpt_path=./checkpoints/N-GlyAltas_classifier.pkl 
```
