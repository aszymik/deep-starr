mkdir data
cd data
mkdir deep-starr
cd deep-starr

# FASTA files with DNA sequences of genomic regions from train/val/test sets
wget https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_Train.fa -O data/Sequences_Train.fa
wget https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_Val.fa -O data/Sequences_Val.fa
wget https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_Test.fa -O data/Sequences_Test.fa

# Files with developmental and housekeeping activity of genomic regions from train/val/test sets
wget https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_activity_Train.txt -O data/Sequences_activity_Train.txt
wget https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_activity_Val.txt -O data/Sequences_activity_Val.txt
wget https://data.starklab.org/almeida/DeepSTARR/Data/Sequences_activity_Test.txt -O data/Sequences_activity_Test.txt