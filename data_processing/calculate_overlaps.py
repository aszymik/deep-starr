import pandas as pd
import matplotlib.pyplot as plt


def read_bed(file_path):
    # Read BED file into DataFrame
    df = pd.read_csv(file_path, sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chr", "start", "end"])
    return df

def calculate_overlap(bed1, bed2):
    overlaps = []

    for i, row1 in bed1.iterrows():
        chr1, start1, end1 = row1["chr"], row1["start"], row1["end"]
        id1 = f"{chr1}:{start1}-{end1}"

        # Filter for matching chromosome
        chr_matches = bed2[bed2["chr"] == chr1]

        for j, row2 in chr_matches.iterrows():
            start2, end2 = row2["start"], row2["end"]
            id2 = f"{chr1}:{start2}-{end2}"

            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                overlaps.append((id1, id2, overlap))

    return overlaps

def save_results(overlaps, output_tsv):
    df = pd.DataFrame(overlaps, columns=["id1", "id2", "overlap"])
    df.to_csv(output_tsv, sep="\t", index=False)

def plot_histogram(overlaps, output_png=None):
    overlaps_only = [x[2] for x in overlaps]
    plt.hist(overlaps_only, bins=50, color='steelblue', edgecolor='black')
    plt.title("Histogram of Overlapping Bases")
    plt.xlabel("Overlap Length (bp)")
    plt.ylabel("Count")
    plt.tight_layout()
    if output_png:
        plt.savefig(output_png)
    plt.show()


def main(bed1_path, bed2_path, output_tsv, output_png=None):
    bed1 = read_bed(bed1_path)
    bed2 = read_bed(bed2_path)

    overlaps = calculate_overlap(bed1, bed2)
    save_results(overlaps, output_tsv)
    plot_histogram(overlaps, output_png)

if __name__ == "__main__":
    bed1 = 'data/lenti-mpra/da_library/primary.bed'
    bed2 = 'data/lenti-mpra/da_library/organoids.bed'
    output_tsv = 'data/lenti-mpra/da_library/primary_organoids_overlap.tsv'
    
    main(bed1, bed2, output_tsv)
