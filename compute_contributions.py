import torch
import shap
import numpy as np
from torch import nn
from your_model import DeepSTARR  # adjust if needed
from tqdm import tqdm

# -- Load trained model --
params = {
    'num_filters': 120, 'kernel_size1': 7, 'kernel_size2': 5,
    'kernel_size3': 5, 'kernel_size4': 3, 'pad': 1,
    'num_filters2': 120, 'num_filters3': 120, 'num_filters4': 120,
    'dense_neurons1': 512, 'dense_neurons2': 256,
    'dropout_prob': 0.3
}
model = DeepSTARR(params)
model.load_state_dict(torch.load('deepstarr_model.pt'))  # path to trained model
model.eval()

# -- Dataset: one-hot encoded input sequences (shape: [N, 4, 249]) --
X = np.load("enhancer_sequences_onehot.npy")  # your real input
X_tensor = torch.tensor(X, dtype=torch.float32)

# -- Create reference sequences: 100 dinucleotide-shuffled versions per input --
def dinuc_shuffle(seq_onehot):
    # Convert to string
    nucs = 'ACGT'
    seq_str = ''.join([nucs[i] for i in np.argmax(seq_onehot, axis=0)])
    # Dinuc shuffle
    import random
    from collections import defaultdict

    def dinuc_shuffle_str(s):
        from random import shuffle
        d = defaultdict(list)
        for i in range(len(s) - 1):
            d[s[i]].append(s[i + 1])
        for k in d: shuffle(d[k])
        result = [s[0]]
        for i in range(1, len(s)):
            prev = result[-1]
            next_base = d[prev].pop() if d[prev] else random.choice('ACGT')
            result.append(next_base)
        return ''.join(result)

    shuffled = dinuc_shuffle_str(seq_str)
    # Convert back to one-hot
    onehot = np.zeros_like(seq_onehot)
    for i, base in enumerate(shuffled):
        onehot["ACGT".index(base), i] = 1
    return onehot

# Use the first N sequences for explanation
num_explain = 100
X_sample = X[:num_explain]

# Reference sequences (mean of 100 dinuc-shuffles per input)
print("Generating background reference...")
background = np.stack([
    np.mean([dinuc_shuffle(x) for _ in range(100)], axis=0)
    for x in tqdm(X_sample)
])
background_tensor = torch.tensor(background, dtype=torch.float32)

# -- Wrap model output to focus on a single output (e.g., dev activity = index 0) --
class ModelWrapper(nn.Module):
    def __init__(self, model, task='dev'):
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, x):
        out_dev, out_hk = self.model(x)
        return out_dev if self.task == 'dev' else out_hk

wrapped_model = ModelWrapper(model, task='dev')  # or task='hk'

# -- Compute SHAP values --
explainer = shap.DeepExplainer(wrapped_model, background_tensor)
shap_values = explainer.shap_values(torch.tensor(X_sample, dtype=torch.float32))

# -- Get contribution scores by multiplying shap values with one-hot input --
# Shape: [num_samples, 4, 249]
contribution_scores = shap_values[0] * X_sample

# -- Save results --
np.save("contribution_scores_dev.npy", contribution_scores)
print("Saved contribution scores for development activity.")

# Repeat for housekeeping:
wrapped_model_hk = ModelWrapper(model, task='hk')
explainer_hk = shap.DeepExplainer(wrapped_model_hk, background_tensor)
shap_values_hk = explainer_hk.shap_values(torch.tensor(X_sample, dtype=torch.float32))
contribution_scores_hk = shap_values_hk[0] * X_sample
np.save("contribution_scores_hk.npy", contribution_scores_hk)
print("Saved contribution scores for housekeeping activity.")
