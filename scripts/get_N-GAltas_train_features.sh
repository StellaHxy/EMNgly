cd ../model
python get_esm_embedding.py --mode=train --data_path=../data/N-GlycositeAltas/ 
python get_mif_embedding.py --mode=train --data_path=../data/N-GlycositeAltas/
cd ..