
import pandas as pd
import os
os.chdir("/home/nanoribo/rna_transformer_project")
input_file = "./data/raw/Publicset_mRNASSpred_halflife.csv"

df = pd.read_csv(input_file)
df["len"] = df["sequence"].apply(len)
print(df["len"].describe()) # all length is 201

