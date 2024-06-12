python model/get_esm_embedding.py --dataset=../data/N-GlycositeAltas/N-GAltas_seq_test.csv
python model/get_mif_embedding.py --dataset=../data/N-GlycositeAltas/N-GAltas_seq_test.csv
python predict.py --dataset=../data/N-GlycositeAltas/N-GAltas_seq_test.csv 