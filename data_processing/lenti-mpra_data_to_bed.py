# Script for preprocessing data (S2) from https://www.science.org/doi/10.1126/science.adh0559

import pandas as pd

VARIANT_DIR = 'data/lenti-mpra/variant_library'
DA_DIR = 'data/lenti-mpra/da_library'

var_primary_data = f'/Users/ania/Desktop/atac-seq/RegSeqNet/data/regression_lentiMPRA/DataS2-Variant-library-ratios_primary.tsv'
var_organoid_data = f'/Users/ania/Desktop/atac-seq/RegSeqNet/data/regression_lentiMPRA/DataS2-Variant-library-ratios_organoid.tsv'

da_organoids_tsv = f'{DA_DIR}/adh0559_organoids.tsv'
da_primary_tsv = f'{DA_DIR}/adh0559_primary.tsv'

VAR_COLS = {
    'rsid': 'str',
    'variant_chrom': 'str',
    'variant_pos': 'object',
    'insert_name': 'str',
    'ref_ratio': 'object', 
    'alt_ratio': 'object'
}
DA_COLS = {
    'insert_chrom': 'str',
    'insert_start': 'int',
    'insert_end': 'int',
    'rna_dna_ratio': 'float'
}

def variant_data_to_bed(data_path, output_path, alternative=False):
    """Parses SNP data to a bed file"""
    
    df = pd.read_csv(data_path, sep='\t', usecols=VAR_COLS.keys(), dtype=VAR_COLS)
    df = df.dropna(subset=['insert_name', 'ref_ratio'])

    df[['position_start', 'position_end']] = df['insert_name'].str.extract(r':(\d+)-(\d+)')
    # df = df.drop(columns=['insert_name'])

    if alternative:
        df[['variant_chrom', 'position_start', 'position_end', 'rsid', 'alt_ratio']].to_csv(output_path, sep='\t', header=False, index=False)
    else:    
        df[['variant_chrom', 'position_start', 'position_end', 'rsid', 'ref_ratio']].to_csv(output_path, sep='\t', header=False, index=False)


def da_data_to_bed(data_path, output_path):
    df = pd.read_csv(data_path, sep='\t', usecols=DA_COLS.keys(), dtype=DA_COLS)
    df.to_csv(output_path, sep='\t', header=False, index=False)


if __name__ == '__main__':
    # Prepare BED files for variant library
    var_primary_ref = f'{VARIANT_DIR}/primary_ref.bed'
    var_primary_alt = f'{VARIANT_DIR}/primary_alt.bed'
    var_organoid_ref = f'{VARIANT_DIR}/organoid_ref.bed'
    var_organoid_alt = f'{VARIANT_DIR}/organoid_alt.bed'

    variant_data_to_bed(var_primary_data, var_primary_ref)
    variant_data_to_bed(var_primary_data, var_primary_alt, alternative=True)

    variant_data_to_bed(var_organoid_data, var_organoid_ref)
    variant_data_to_bed(var_organoid_data, var_organoid_alt, alternative=True)

    # Prepare BED files for DA library
    da_organoids_bed = f'{DA_DIR}/organoids.bed'
    da_primary_bed = f'{DA_DIR}/primary.bed'

    da_data_to_bed(da_organoids_tsv, da_organoids_bed)
    da_data_to_bed(da_primary_tsv, da_primary_bed)
