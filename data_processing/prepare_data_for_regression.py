import pandas as pd
import numpy as np
from pybedtools.bedtool import BedTool
from pybedtools import featurefuncs, Interval

# znaleźć pliki z aktywnością
# zmienić bed do 249 bp zachowując informację o aktywności
# bed2fasta
# zamienić na capslock
# dodać info o aktywności do nagłówka albo zapisać w osobnym pliku
# zmodyfikować kod przetwarzający input

def convert_to_specified_length(bedtool: BedTool, midpoints=False, length=2000):
    """Converts input BedTool regions to 2 kb intervals centered around their midpoints."""
    substract = length // 2
    if length % 2 == 0:
        add = length // 2 - 1
    else:
        add = length // 2

    intervals = []
    for region in bedtool:
        # Find the midpoint of each region
        centroid = featurefuncs.midpoint(region) if not midpoints else region
        try:
            region = Interval(region.chrom, centroid.start - substract, centroid.end + add)  # elongating each side
            intervals.append(region)
        except OverflowError:  # 1st coordinate negative – three cases
            print(region.chrom, centroid.start - substract, centroid.end + add)
        
    return BedTool(intervals)

def prepare_bed(bed_file, output_bed=None, length=200):
    regions = BedTool(bed_file)
    regions = convert_to_specified_length(regions, length=length)
    if output_bed:
        regions.saveas(output_bed)
    return regions

def preprare_bed_for_regression(*bed_files, output_bed, length=200):

    with open(output_bed, 'w') as out_file:
        for bed_file in bed_files:
            regions = prepare_bed(bed_file, length=length)
            print(f'{len(regions)=}')
            for region in regions:
                out_file.write(str(region))

def process_fasta_to_list(input_path):
    lines = []

    with open(input_path, 'r') as file:
        filelines = file.read().split('>')

    # Process file so that each sequence represents one line
    for i in range(1, len(filelines)):
        filelines[i] = filelines[i].split('\n')
        lines.append('>' + filelines[i][0] + '\n')
        lines.append(''.join(filelines[i][1:]))

    return lines

def prepare_fasta_for_regression(fasta_file, bed_file, output_fasta, min_length=200, unique_region_names=False):
    # Parse bed file
    bed_columns = ['chrom', 'start', 'end', 'region_name', 'activity']
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=bed_columns)
    grouped_bed = bed_df.groupby(['chrom', 'start', 'end'])

    # Parse fasta file
    fasta_records = process_fasta_to_list(fasta_file)

    with open(output_fasta, 'w') as output_file:
        for i in range(0, len(fasta_records), 2):

            # fasta id preprocessing
            try:
                full_coordinates, name = fasta_records[i].strip().split('_')
            except ValueError:
                full_coordinates, name = fasta_records[i].strip().split()

            chrom, coordinates = full_coordinates.split(':')
            chrom = f'chr{chrom.split(">")[-1]}'
            start, end = coordinates.split('-')
            start = int(start)
            end = int(end.split('(')[0])
            seq = fasta_records[i+1]

            if len(seq) >= min_length:  
                try:
                    if unique_region_names:
                        # mozemy przeszukać bed po nazwie
                        activity = bed_df.loc[bed_df['region_name'] == name, 'activity'].values[0]
                    else:
                        # szukamy po współrzędnych
                        activity = grouped_bed.get_group((chrom, start, end))['activity'].values[0]

                    fasta_records[i] = f'{full_coordinates}_{name}_{activity}'  # dodajemy informację o aktywności do nagłówka fasta
                    output_file.write(f'{fasta_records[i]}\n{convert_to_specified_length(seq, 2000)}\n')

                except IndexError:
                    print(f'Error in {fasta_file}: ')
                    print(chrom, start, end)

LENGTH = 249
            
active_2022_bed = 'data/atac-starr-seq/GSE181317_GM12878_active_regions.bed'
silent_2022_bed = 'data/atac-starr-seq/GSE181317_GM12878_silent_regions.bed'

bed_2018 = 'data/atac-starr-seq/GSE104001_HiDRA_counts.bed'

output_bed_2022 = 'data/atac-starr-seq/2022_regions_249bp.bed'
output_bed_2018 = 'data/atac-starr-seq/2018_regions_249bp.bed'


if __name__ == '__main__':

    preprare_bed_for_regression(active_2022_bed, silent_2022_bed, output_bed=output_bed_2022, length=LENGTH)
    preprare_bed_for_regression(bed_2018, output_bed=output_bed_2018, length=LENGTH)


    # prepare_fasta_for_regression(f'{min_200bp_active}.fa', f'{min_200bp_active}.bed', f'{min_200bp_active}_final.fa')
    # prepare_fasta_for_regression(f'{min_200bp_silent}.fa', f'{min_200bp_silent}.bed', f'{min_200bp_silent}_final.fa')




