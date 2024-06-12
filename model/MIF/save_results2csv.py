
import os
import sys

import numpy as np
import pandas as pd
import requests
import time
import argparse
from datetime import datetime
from pathlib import Path
from requests.exceptions import HTTPError
from sequence_models.constants import get_key, IUPAC_CODES,PROTEIN_ALPHABET


def get_index_in_PROTEIN_ALPHABET(str):
    for j,i in enumerate(PROTEIN_ALPHABET[:20]):
        if str == i:
            return j


def num_to_aa(aa_str):
    AAs = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
    intLabels = [9,3,7,2,19,8,1,18,0,12,11,4,13,14,17,5,6,16,15,10]

    aaDict = dict(zip(intLabels, AAs))
    return aaDict[aa_str]


def PROTEIN_ALPHA_TO_AAs(prob): # arg:prob is the 20-probs for each wt aa
    AAs = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
    AA_protein = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
    output_list = []
    for aa in AAs:
        for j , i in enumerate(AA_protein):
            if aa == i:
                output_list.append(prob[j])
                break
    return output_list



def save_output( model_name, pdb_name , origin_seq, result_seq, res_matrix ):
    AAs = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
           'SER', 'THR', 'TRP', 'TYR', 'VAL']
    col_names = ['pdb_id', 'chain_id', 'pos', 'wtAA', 'prAA', 'wt_prob',
                 'pred_prob', 'avg_log_ratio'] +['pr' + i for i in AAs]
    output_list = []

    for i , prob in enumerate(res_matrix):
        chn_id = 'A' #
        pdb_id = 0  # PDB code only  # Need this so the nn_api can find the file correctly.
        # pdb_id = i[0]+'_'+i[1] # PDB and chain (How will the nn_api know the chain?)
        aa_id = str(pdb_id) + '_'+ chn_id
        wt_aa = get_key(IUPAC_CODES,origin_seq[i]).upper() # origin amino acid
        pred_aa = get_key(IUPAC_CODES,result_seq[i]).upper() # predicted amino acid
        wt_prob = prob[get_index_in_PROTEIN_ALPHABET(origin_seq[i])]
        pred_prob = prob[get_index_in_PROTEIN_ALPHABET(result_seq[i])]
        avgLogRatio = np.round(np.log(pred_prob/wt_prob), 3)

        # print(prob)
        # print(PROTEIN_ALPHA_TO_AAs(prob))
        output_list.append(
            [
                pdb_id, chn_id, i+1, wt_aa, pred_aa, wt_prob, pred_prob,
                avgLogRatio ]+ PROTEIN_ALPHA_TO_AAs(prob)  )
        # print([ pdb_id, chn_id, i+1, wt_aa, pred_aa, wt_prob, pred_prob,avgLogRatio ]+ PROTEIN_ALPHA_TO_AAs(prob) )

    output_df = pd.DataFrame(output_list, columns=col_names)
    output_df['aa_id'] = aa_id
    # print(output_df.head(5))
    output_df = output_df[
        [
            'aa_id', 'pdb_id', 'chain_id', 'pos', 'wtAA',
            'prAA', 'wt_prob', 'pred_prob', 'avg_log_ratio',
            'prALA', 'prARG', 'prASN', 'prASP', 'prCYS',
            'prGLN', 'prGLU', 'prGLY', 'prHIS', 'prILE',
            'prLEU', 'prLYS', 'prMET', 'prPHE', 'prPRO',
            'prSER', 'prTHR', 'prTRP', 'prTYR', 'prVAL'
        ]
    ]
    # print(output_df.head(5))
    output_filename = 'output/'+ pdb_name+'_'+model_name + ".csv"
    output_df.to_csv(output_filename)



