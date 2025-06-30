import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class RNADataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, ss_tensor=None, mfe_tensor=None, metadata=None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.ss_tensor = torch.tensor(ss_tensor, dtype=torch.long) if ss_tensor is not None else None
        self.mfe_tensor = torch.tensor(mfe_tensor, dtype=torch.float32).unsqueeze(1) if mfe_tensor is not None else None
        self.metadata = metadata

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            "sequences": self.sequences[idx],
            "labels": self.labels[idx]
        }
        if self.ss_tensor is not None:
            item["ss"] = self.ss_tokens[idx]  # shape (max_len,)
        if self.mfe_tensor is not None:
            item["mfe"] = self.mfe_values[idx]  # shape (1,)
        if self.metadata:
            item["meta"] = self.metadata[idx]
        return item
# One-hot encoding helper
def one_hot_encode(seq, vocab, max_len):
    vec = np.zeros((max_len, len(vocab)), dtype=np.float32)
    for i, char in enumerate(seq[:max_len]):
        if char in vocab:
            vec[i, vocab[char]] = 1.0
    return vec
    
def load_and_prepare_data(
    csv_path,
    max_len=201,
    test_size=0.2,
    random_state=42,
    use_ss=False,
    use_mfe=False
):
    df = pd.read_csv(csv_path)

    # Vocabularies
    seq_vocab = {'A': 0, 'G': 1, 'T': 2, 'C': 3}     # one-hot: 4D
    ss_vocab = {'.': 0, '(': 1, ')': 2}              # one-hot: 3D

    # === Sequence one-hot ===
    df["seq_encoded"] = df["sequence"].apply(lambda x: one_hot_encode(x, seq_vocab, max_len))
    X_seq = np.stack(df["seq_encoded"].values)  # shape: (N, L, 4)

    # === Label ===
    y = df["half_life"].values.astype(np.float32)

    # === SS one-hot (optional) ===
    if use_ss:
        df["ss_encoded"] = df["SS"].apply(lambda x: one_hot_encode(x, ss_vocab, max_len))
        X_ss = np.stack(df["ss_encoded"].values)  # shape: (N, L, 3)
    else:
        X_ss = None

    # === MFE scalar (optional) ===
    if use_mfe:
        X_mfe = df["MFE"].values.astype(np.float32)  # shape: (N,)
    else:
        X_mfe = None

    # === Metadata ===
    meta = df[["ids", "sequence", "SS", "MFE"]].to_dict(orient="records")

    # === Split ===
    #split is based on row alignment, not contents.
    split_args = [X_seq, y, meta]
    if use_ss:
        split_args.append(X_ss)
    if use_mfe:
        split_args.append(X_mfe)

    split = train_test_split(*split_args, test_size=test_size, random_state=random_state)

    # === Unpack ===
    X_seq_tr, X_seq_val, y_tr, y_val, meta_tr, meta_val = split[:6]
    X_ss_tr = X_ss_val = None
    X_mfe_tr = X_mfe_val = None

    if use_ss:
        X_ss_tr, X_ss_val = split[6:8] if use_mfe else split[6:8]
    if use_mfe:
        X_mfe_tr, X_mfe_val = split[8:10] if use_ss else split[6:8]

    # === Wrap into Datasets ===
    train_dataset = RNADataset(X_seq_tr, y_tr, ss_tensor=X_ss_tr, mfe_tensor=X_mfe_tr, metadata=meta_tr)
    val_dataset   = RNADataset(X_seq_val, y_val, ss_tensor=X_ss_val, mfe_tensor=X_mfe_val, metadata=meta_val)

    return train_dataset, val_dataset, seq_vocab, ss_vocab


if __name__ == "__main__":
    input_csv = "/home/nanoribo/rna_transformer_project/data/raw/Publicset_mRNASSpred_halflife.csv"
    train_ds, val_ds, seq_vocab, ss_vocab = load_and_prepare_data(
        csv_path=input_csv,  # ✅ use a small file for testing
        max_len=201,
        use_ss=True,
        use_mfe=True
    )

    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")

    # Show a sample
    sample = train_ds[0]
    print("\nSample keys:", sample.keys())
    print("Sequence shape:", sample['sequences'].shape)
    print("SS shape:", sample['ss'].shape if 'ss' in sample else 'N/A')
    print("MFE:", sample.get('mfe', 'N/A'))
    print("Label:", sample['labels'])
    print("Metadata:", sample.get('meta', 'N/A'))