import os
import torch
import pickle as pk
import tqdm
import argparse
import pandas as pd

ESM_EMBED_LAYER = 33
ESM_EMBED_DIM = 1280
ESM_MODEL_PATH = ["facebookresearch/esm:main", "esm1b_t33_650M_UR50S"]

# adapted from https://github.com/facebookresearch/esm

class ESMEmbeddingExtractor:
    def __init__(self, repo_or_dir, model):
        self.model, alphabet = torch.hub.load(repo_or_dir, model)
        self.model.eval()
        self.batch_converter = alphabet.get_batch_converter()
        self.max_input_len = 1022
        self.max_step_len = 511

    def to(self, device):
        self.model = self.model.to(device)

    def extract(self, seqs, repr_layer=None, return_contacts=False, device=None):
        """ Returns the ESM embeddings for a protein.
            Inputs:
            * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
            Outputs: tensor of (batch, n_seqs, L, embedd_dim)
                * n_seqs: number of sequences in the MSA. 1 for ESM-1b
                * embedd_dim: number of embedding dimensions. 1280 for ESM-1b
        """
        # use ESM transformer
        device = device
        repr_layer = ESM_EMBED_LAYER
        max_seq_len = len(seqs[0])

        representations, contacts = [], []

        assert not return_contacts or max_seq_len <= self.max_input_len
        # Extract per-residue representations
        with torch.no_grad():
            #batch_seqs = [(' ' , seq) for seq in seqs]
            for i in range(0, max_seq_len, self.max_step_len):
                j = min(i + self.max_input_len, max_seq_len)
                delta = 0 if i == 0 else self.max_input_len - self.max_step_len
                if i > 0 and j < i + self.max_input_len:
                    delta += i + self.max_input_len - max_seq_len
                    i = max_seq_len - self.max_input_len

                batch_seqs = [('', seq[i:j]) for seq in seqs]
                batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_seqs)
                batch_tokens = batch_tokens.to(device)
                results = self.model(batch_tokens, repr_layers=[repr_layer], return_contacts=return_contacts)
                # index 0 is for start token. so take from 1 one
                representations.append(results['representations'][repr_layer][...,delta:j-i+1,:])
                if return_contacts:
                    contacts.append(results['contacts'])
                
                if j >= max_seq_len:
                    break
        
        if return_contacts:
            return torch.cat(representations, dim=1), torch.cat(contacts, dim=1)
        return torch.cat(representations, dim=1)[0]


def make_seq_data(input_path):
    rf = pd.read_csv(input_path)
    seq_set = set()
    seq_list = []
    label_list = []
    true_count = 0
    for index, items in rf.iterrows():
        if len(items)<3:
            continue
        label = items['label']
        seq = items['sequence']
        pos = items['pos']
        seq_set.add(seq)
        if label == '1':
            label_list.append(1)
            true_count += 1
        else:
            label_list.append(0)
        seq_list.append(seq)
        
    print(len(seq_list))
    return seq_set, seq_list, label_list


def get_local_features(input_path, output_dir, device):
    output_dir_local = output_dir + '/local'
    model =  ESMEmbeddingExtractor(ESM_MODEL_PATH[0], ESM_MODEL_PATH[1])
    model.to(device)

    rf = pd.read_csv(input_path)
    
    if not os.path.isdir(output_dir_local):
        os.makedirs(output_dir_local)

    for i, row in tqdm.tqdm(rf.iterrows()):
        seq = row['sequence']
        pos = row['pos']
        left = max(0, pos-21)
        right = min(pos+20, len(seq))
        local_seq = seq[left: right]
        save_name = os.path.join(output_dir_local, f'{i}.pkl')
        emb = model.extract([local_seq], device=device)[0].cpu()
        output_dict = {'local_emb': emb}
        with open(save_name, 'wb') as f:
            pk.dump(output_dict, f, protocol=4)
    
    return output_dir_local


def get_site_features(input_path, output_dir, device):
    output_dir_site = output_dir + '/site'
    model =  ESMEmbeddingExtractor(ESM_MODEL_PATH[0], ESM_MODEL_PATH[1])
    model.to(device)

    rf = pd.read_csv(input_path)
    
    if not os.path.isdir(output_dir_site):
        os.makedirs(output_dir_site)
    
    for i, row in tqdm.tqdm(rf.iterrows()):
        seq = row['sequence']
        pos = row['pos']
        try:
            save_name = os.path.join(output_dir_site, f'{i}.pkl')
            emb = model.extract([seq], device=device)[pos].cpu()
            output_dict = {'site_emb': emb}
            with open(save_name, 'wb') as f:
                pk.dump(output_dict, f, protocol=4)
        except:
            continue
    
    return output_dir_site

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--data_path', type=str, default="../data/N-GlycositeAltas")
    parser.add_argument('--device', type=str, default='cuda:3')
    args = parser.parse_args()

    device = torch.device(args.device)
    csv_file = os.path.join(args.data_path, args.mode+'.csv')
    emb_dir = os.path.join(args.data_path, "features", args.mode)

    print("Start getting local features!")
    output_dir_local = get_local_features(csv_file, emb_dir, device)
    print('Get local features finshed.')
    print("Start getting site features!")
    output_dir_site = get_site_features(csv_file, emb_dir, device)
    print('Get site features finshed.')
        