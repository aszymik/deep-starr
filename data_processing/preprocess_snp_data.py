# Script for preprocessing data (S2) from https://www.science.org/doi/10.1126/science.adh0559

import pandas as pd

DIR = 'data/lenti-mpra'
data = f'/Users/ania/Desktop/atac-seq/RegSeqNet/data/regression_lentiMPRA/DataS2-Variant-library-ratios_primary.tsv'
primary_ref = f'{DIR}/primary_ref.bed'
primary_alt = f'{DIR}/primary_alt.bed'
# cols = ['rsid', 'variant_chrom', 'variant_pos', 'insert_name', 'alt_ref_ratio', 'alt_ratio', 'ref_ratio', 'alt_is_active', 'ref_is_active']

COLS = {
    'rsid': 'str',
    'variant_chrom': 'str',
    'variant_pos': 'object',
    'insert_name': 'str',
    'ref_ratio': 'object', 
    'alt_ratio': 'object'
}

def snp_data_to_bed(data_path, output_path, alternative=False):
    """ Póki co przetwarza tylko referencję, ale docelowo dla SNP tez chcemy przewidywać """
    
    df = pd.read_csv(data_path, sep='\t', usecols=COLS.keys(), dtype=COLS)
    df = df.dropna(subset=['insert_name', 'ref_ratio'])

    df[['position_start', 'position_end']] = df['insert_name'].str.extract(r':(\d+)-(\d+)')
    # df = df.drop(columns=['insert_name'])

    if alternative:
        df[['variant_chrom', 'position_start', 'position_end', 'rsid', 'alt_ratio']].to_csv(output_path, sep='\t', header=False, index=False)
    else:    
        df[['variant_chrom', 'position_start', 'position_end', 'rsid', 'ref_ratio']].to_csv(output_path, sep='\t', header=False, index=False)

snp_data_to_bed(data, primary_ref)
snp_data_to_bed(data, primary_alt, alternative=True)