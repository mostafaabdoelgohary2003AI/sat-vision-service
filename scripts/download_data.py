"""Download the EuroSAT RGB parquet shards (test + validation splits) into data/."""
import os, urllib.request

URLS = [
    ("data/eurosat_test.parquet", "https://huggingface.co/api/datasets/blanchon/EuroSAT_RGB/parquet/default/test/0.parquet"),
    ("data/eurosat_val.parquet", "https://huggingface.co/api/datasets/blanchon/EuroSAT_RGB/parquet/default/validation/0.parquet"),
]
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for path, url in URLS:
        if os.path.exists(path):
            print("exists:", path); continue
        print("downloading", url)
        urllib.request.urlretrieve(url, path)
        print("saved:", path)
