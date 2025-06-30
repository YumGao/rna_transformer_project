# test_data_utils.py
from src.data_utils import load_and_prepare_data

def test_data_loading():
    csv_path = "data/raw/test_input.csv"
    max_len = 201
    train_dataset, val_dataset, seq_vocab, ss_vocab = load_and_prepare_data(
        csv_path=csv_path,
        max_len=max_len,
        use_ss=True,
        use_mfe=True
    )

    # Print vocab info
    print("Sequence vocab:", seq_vocab)
    print("SS vocab:", ss_vocab)

    # Check dataset length
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    # Print one sample
    sample = train_dataset[0]
    print("\n🔍 Sample keys:", sample.keys())
    print("sequences shape:", sample["sequences"].shape)
    print("ss shape:", sample["ss"].shape)
    print("mfe:", sample["mfe"])
    print("label:", sample["labels"])
    print("meta:", sample["meta"])

if __name__ == "__main__":
    test_data_loading()