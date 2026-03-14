import pandas as pd

print("Loading dataset...")

# load train data
df = pd.read_csv("data/multimodal_train.tsv", sep="\t")

print("Original size:", len(df))

# sample 10000 rows
subset = df.sample(n=10000, random_state=42)

# save subset
subset.to_csv("data/train_subset.tsv", sep="\t", index=False)

print("Subset created!")
print("New size:", len(subset))