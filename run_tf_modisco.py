import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import logomaker
from tfmodisco.tfmodisco_workflow import workflow
from tfmodisco import Seqlet, Metacluster
from tqdm import tqdm

# === CONFIG ===
task_name = "dev"  # or "hk"
contrib_path = f"contribution_scores_{task_name}.npy"
onehot_path = "enhancer_sequences_onehot.npy"
out_dir = f"tfmodisco_{task_name}"
os.makedirs(out_dir, exist_ok=True)

# === Load data ===
print("Loading contribution scores and sequences...")
contrib_scores = np.load(contrib_path)  # shape: (N, 4, 249)
onehot_data = np.load(onehot_path)      # shape: (N, 4, 249)
assert contrib_scores.shape == onehot_data.shape

# === Run TF-MoDISco ===
print("Running TF-MoDISco...")
tfm = workflow.TfModiscoWorkflow(
    sliding_window_size=15,
    flank_size=5,
    target_seqlet_fdr=0.15,
    seqlet_aggregation_strategy="median",
    min_metacluster_size=30,
    max_seqlets_per_metacluster=50000,
    trim_to_window_size=15
)

tfmodisco_results = tfm(
    task_to_scores={task_name: contrib_scores},
    task_to_patterns={"": onehot_data},
    one_hot=onehot_data
)

# === Save results ===
results_path = os.path.join(out_dir, f"tfmodisco_results_{task_name}.h5")
print(f"Saving results to {results_path}...")
tfmodisco_results.save_hdf5(results_path)

# === Extract and visualize motifs ===
def plot_and_save_motifs(tfmodisco_results, out_dir, task_name):
    for i, pattern in enumerate(tfmodisco_results.metacluster_idx_to_submetacluster_results[0].seqlets_to_patterns_result.patterns):
        pwm = pattern["sequence"].fwd
        name = f"motif_{i}"
        
        # Save matrix
        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), pwm=pwm)
        np.savetxt(os.path.join(out_dir, f"{name}.txt"), pwm, fmt="%.4f")

        # Plot logo
        df = logomaker.transform_matrix(pwm, from_type="probability", to_type="information")
        logo = logomaker.Logo(df)
        plt.title(name)
        plt.savefig(os.path.join(out_dir, f"{name}.png"))
        plt.savefig(os.path.join(out_dir, f"{name}.pdf"))
        plt.close()

plot_and_save_motifs(tfmodisco_results, out_dir, task_name)
print("Motifs saved and visualized.")
