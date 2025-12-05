import os
from turtle import down, end_fill
import requests
import tiktoken
import numpy as np

# configuration
BOOKS = {
    "Crime and Punishment": "https://www.gutenberg.org/cache/epub/2554/pg2554.txt",
    "The Brothers Karamazov": "https://www.gutenberg.org/cache/epub/28054/pg28054.txt",
    "The Idiot": "https://www.gutenberg.org/cache/epub/2638/pg2638.txt",
}


def download_and_clean():
    """download and clean dostoevsky's books"""
    combined_text = ""
    for title, url in BOOKS.items():
        print(f"downloading {title}...")
        try:
            r = requests.get(url)
            r.raise_for_status()
            text = r.text

            # simple cleaning: remove Gutenberg header/footer
            start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
            end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"

            start_idx = text.find(start_marker)
            end_idx = text.find(end_marker)

            # if the markers are found, slice the text accordingly
            if start_idx != -1 and end_idx != -1:
                # adjust indices to get the actual content
                clean_text = text[start_idx + 100 : end_idx].strip()
            else:
                # fallback
                clean_text = text

            combined_text += clean_text + "\n\n"

        except Exception as e:
            print(f"failed to {title}: {e}")

    return combined_text


if __name__ == "__main__":
    # get the data
    data = download_and_clean()
    print(f"total dataset size {len(data)/1024/1024:.2f} MB")

    # save the raw text
    input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(data)

    # tokenize the text
    print("tokenizing the dataset with gpt-2 encoder...")
    encoder = tiktoken.get_encoding("gpt2")
    train_ids = encoder.encode(data)
    print(f"tokenized dataset size: {len(train_ids):,} tokens")

    # split train 90% and val 10%
    n = len(train_ids)
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = train_ids[int(n * 0.9) :]  # take the last 10%
    train_ids = train_ids[: int(n * 0.9)]  # take the first 90%

    # save to binary files
    train_path = os.path.join(os.path.dirname(__file__), "train.bin")
    val_path = os.path.join(os.path.dirname(__file__), "val.bin")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(
        f"Dataset saved: {train_path} ({len(train_ids)} tokens), {val_path} ({len(val_ids)} tokens)"
    )
