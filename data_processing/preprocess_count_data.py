import pandas as pd

raw_data_2018 = 'data/atac-starr-seq/GSE104001_HiDRA_counts_per_fragmentgroup.txt'
preprocessed_data_2018 = 'data/atac-starr-seq/GSE104001_HiDRA_counts.bed'

df = pd.read_csv(raw_data_2018, sep='\t')
print(df.head())
print(df.columns)

df[['chr', 'positions']] = df.FragmentGroupPosition_NumUniqFragments.str.split(':', expand=True)
df[['start', 'end_NumUniqFragments']] = df.positions.str.split('-', expand=True)
df[['end', 'NumUniqFragments']] = df.end_NumUniqFragments.str.split('_', expand=True)
df = df.drop(['FragmentGroupPosition_NumUniqFragments', 'positions', 'end_NumUniqFragments'], axis=1)

# Mean and std
df['plasmid_mean'] = df[['P1', 'P2', 'P3', 'P4', 'P5']].mean(axis=1)
df['RNA_mean'] = df[['R1', 'R2', 'R3', 'R4', 'R5']].mean(axis=1)

df['activity'] = df['RNA_mean'] / df['plasmid_mean']
df['region'] = [f'region {i}' for i in range(1, len(df)+1)]

df[['chr', 'start', 'end', 'region', 'activity']].to_csv(preprocessed_data_2018, sep='\t', index=False, header=False)