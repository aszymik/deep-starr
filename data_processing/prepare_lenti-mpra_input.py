import pandas as pd
from collections import defaultdict
from Bio import SeqIO

org_bed_path = "data/lenti-mpra/da_library/organoids.bed"
prim_bed_path = "data/lenti-mpra/da_library/primary.bed"
fasta_path = "data/lenti-mpra/da_library/organoids.fa"

output_prefix = "data/lenti-mpra/da_library/preprocessed/Sequences"

VALID_CHR = ["14", "15", "16", "17"]
TEST_CHR = ["18", "19", "20", "21", "22"]

# Load bed files with activity
def read_bed_with_activity(path):
    return pd.read_csv(path, sep="\t", header=None, names=["chr", "start", "end", "activity"])

org_df = read_bed_with_activity(org_bed_path)
prim_df = read_bed_with_activity(prim_bed_path)

# Create a dict from organoids fasta (all sequences in primary): id -> (sequence, activity)
fasta_dict = {}
for record in SeqIO.parse(fasta_path, "fasta"):
    header_parts = record.description.split()
    loc = header_parts[0].replace(">", "").split("(")[0]  # chr:start-end
    activity = float(header_parts[1])
    fasta_dict[loc] = (str(record.seq), activity)

# Merge organoid and primary dfs on chr/start/end
merged = pd.merge(
    org_df, prim_df,
    on=["chr", "start", "end"],
    suffixes=("_organoid", "_primary")
)

# Add a column for id
merged["id"] = merged["chr"] + ":" + merged["start"].astype(str) + "-" + merged["end"].astype(str)

# Assign to split
def assign_split(chrom):
    chrom_num = chrom.replace("chr", "")
    if chrom_num in VALID_CHR:
        return "val"
    elif chrom_num in TEST_CHR:
        return "test"
    else:
        return "train"

merged["split"] = merged["chr"].apply(assign_split)

# Prepare writers and write to files
fasta_buffers = defaultdict(list)
tsv_buffers = defaultdict(list)

for _, row in merged.iterrows():
    loc = f"{row['chr']}:{row['start']}-{row['end']}"
    split = row["split"]

    if loc not in fasta_dict:
        continue  # skip missing

    seq, org_activity = fasta_dict[loc]
    prim_activity = row["activity_primary"]

    # Adjust sequence to 249 bp
    desired_len = 249
    current_len = len(seq)

    if current_len > desired_len:
        # Trim symmetrically
        extra = current_len - desired_len
        left_trim = extra // 2
        right_trim = extra - left_trim
        seq = seq[left_trim: current_len - right_trim]
    elif current_len < desired_len:
        # Pad with Ns symmetrically
        missing = desired_len - current_len
        left_pad = "N" * (missing // 2)
        right_pad = "N" * (missing - (missing // 2))
        seq = left_pad + seq + right_pad

    # Write to buffer lists for consistent order
    fasta_buffers[split].append(f">{loc}\n{seq}")
    tsv_buffers[split].append(f"{loc}\t{prim_activity}\t{org_activity}")


# Write to files with matching order
for split in ["train", "val", "test"]:
    with open(f"{output_prefix}_{split.capitalize()}.fa", "w") as f:
        f.write("\n".join(fasta_buffers[split]) + "\n")

    with open(f"{output_prefix}_activity_{split.capitalize()}.txt", "w") as f:
        f.write("Seq_id\tPrimary_log2_enrichment\tOrganoid_log2_enrichment\n")
        f.write("\n".join(tsv_buffers[split]) + "\n")
