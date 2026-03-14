import argparse
import pandas as pd
import os
from tqdm import tqdm
import urllib.request
import numpy as np

parser = argparse.ArgumentParser(description='Fakeddit image downloader')

# argument will be the path to dataset file
parser.add_argument('dataset_path', type=str, help='path to dataset tsv file')

args = parser.parse_args()

print("Loading dataset...")
df = pd.read_csv(args.dataset_path, sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

print("Dataset loaded:", len(df), "rows")

# create image folder
image_folder = "images"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

pbar = tqdm(total=len(df))
num_failed = 0

for index, row in df.iterrows():

    if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
        image_url = row["image_url"]

        try:
            image_path = os.path.join(image_folder, row["id"] + ".jpg")

            # avoid redownloading
            if not os.path.exists(image_path):
                urllib.request.urlretrieve(image_url, image_path)

        except:
            num_failed += 1

    pbar.update(1)

print("Download finished")
print("Failed images:", num_failed)