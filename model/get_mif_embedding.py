import sys
sys.path.append("./MIF")

from MIF.sequence_models.pretrained import load_model_and_alphabet
import torch
import numpy as np
import pickle as pk
import os
import pandas as pd
import tqdm
import argparse
from MIF.sequence_models.pdb_utils import parse_PDB, process_coords


def run(name_list, model_name, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if type(name_list) != list:
        name_list = [name_list]
    batch = []
    output_name = []
    
    for name in name_list:
        # name = 'Q7L0Y3'
        pdb_path = os.path.join(data_root, f'{name}.pdb')
        # print(f"pdb_path: {pdb_path}")
        # To encode a sequence with its structure
        coords, wt, _ = parse_PDB(pdb_path) # parse pdb and get coordinates
        coords = {                          # encapsulate coords from list to dict
                'N': coords[:, 0],
                'CA': coords[:, 1],
                'C': coords[:, 2]
            }
        dist, omega, theta, phi = process_coords(coords)  # fill diagonal element with 'nan' value
        
        data = [wt, torch.tensor(dist, dtype=torch.float),
                torch.tensor(omega, dtype=torch.float),
                torch.tensor(theta, dtype=torch.float), torch.tensor(phi, dtype=torch.float)]
        batch.append(data)
        output_name.append(name)

    src, nodes, edges, connections, edge_mask = collater(batch)
    outputs = model(src, nodes, edges, connections, edge_mask)
    for i, name in enumerate(output_name):
        emb = outputs[i]
        output_dict = {'structure_emb': emb}
        save_name = os.path.join(output_dir, f'{name}.pkl')
        with open(save_name, 'wb') as f:
            pk.dump(output_dict, f, protocol = 4)
    return output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--data_path', type=str, default="../data/N-GlycositeAltas/")
    parser.add_argument('--device', type=str, default='cuda:3')
    args = parser.parse_args()
    
    data_root = os.path.join(args.data_path, "pdb_file")
    df_root = os.path.join(args.data_path, args.mode+'.csv')
    output_dir = os.path.join(args.data_path, "features", args.mode, "structure")

    error_path = os.path.join(args.data_path, "features", args.mode,'structure_error.txt')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    model_name = 'mif'  
    model, collater = load_model_and_alphabet('./MIF/weights/mif.pt')
    df = pd.read_csv(df_root)
    names = list(set(df['id']))
    print("Number of sequences",len(names))
    emb_num = 0
    batch=False
    
    if batch:
        name_list = []
        serial=0
        for name in tqdm.tqdm(names):
            serial += 1
            if serial%32!=0:
                name_list.append(name)
            else:
                try:
                    run(name_list, model_name, output_dir)
                    emb_num += 1
                except:
                    with open(error_path, "a+") as f:
                        f.write(name)
                    continue
    else:
        # for name in tqdm.tqdm(names):
        for i in tqdm.tqdm(range(len(names))):
            name = names[i]
            save_name = os.path.join(output_dir, f'{name}.pkl')
            if os.path.isfile(save_name):
                continue
            try:
                run(name, model_name, output_dir)
                emb_num += 1
            except:
                with open(error_path, "a+") as f:
                    f.write(name)
                    f.write("/n")
                continue
    
    print("success num", emb_num)
    print('Get structure features finshed.')