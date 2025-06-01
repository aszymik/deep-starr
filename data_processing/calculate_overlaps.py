import pandas as pd
import matplotlib.pyplot as plt
import pyranges as pr

def read_bed(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, usecols=[0, 1, 2],
                     names=["Chromosome", "Start", "End"])
    return pr.PyRanges(df)

def calculate_overlap(bed1, bed2):
    overlaps = bed1.join(bed2)
    df = overlaps.df
    df["id1"] = df["Chromosome"].astype(str) + ":" + df["Start"].astype(str) + "-" + df["End"].astype(str)
    df["id2"] = df["Chromosome"].astype(str) + ":" + df["Start_b"].astype(str) + "-" + df["End_b"].astype(str)
    df["overlap"] = (df[["Start", "Start_b"]].max(axis=1)
                     .rsub(df[["End", "End_b"]].min(axis=1), axis=0)).clip(lower=0)
    return df[["id1", "id2", "overlap"]].values.tolist()

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
