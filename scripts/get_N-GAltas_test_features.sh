cd ../model
python get_esm_embedding.py --mode=test --data_path=../data/N-GlycositeAltas/ 
python get_mif_embedding.py --mode=test --data_path=../data/N-GlycositeAltas/
cd ..