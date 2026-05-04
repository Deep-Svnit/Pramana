import os
import re
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
INPUT_FILES = ["first.txt", "second.txt", "third.txt"]
OUTPUT_DIR = "processed_embeddings"
MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Better than MiniLM

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = SentenceTransformer(MODEL_NAME)


# =========================
# HELPERS
# =========================

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_pages(text):
    pages = re.split(r"=== PAGE \d+ START ===", text)
    return [p.strip() for p in pages if p.strip()]


def extract_tables(text):
    tables = re.findall(r"\[TABLE\](.*?)\[/TABLE\]", text, re.DOTALL)
    return tables


def remove_tables(text):
    return re.sub(r"\[TABLE\].*?\[/TABLE\]", "", text, flags=re.DOTALL)


# 🔥 VERY IMPORTANT: Convert table → sentences
def table_to_sentences(table_text):
    table_text = table_text.replace("\n", " ")

    # crude split based on known headers
    rows = re.split(r"(Particulars|Volume)", table_text)

    sentences = []
    for row in rows:
        row = row.strip()
        if len(row) < 20:
            continue

        # convert to readable sentence
        sentence = f"Table data: {row}"
        sentences.append(sentence)

    return sentences


# =========================
# CHUNKING STRATEGY
# =========================
def chunk_text(text, max_length=400):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < max_length:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    return chunks


# =========================
# MAIN PROCESSING
# =========================
all_chunks = []
all_metadata = []

for file in INPUT_FILES:
    if not os.path.exists(file):
        print(f"Skipping missing file: {file}")
        continue

    raw_text = read_file(file)
    pages = split_pages(raw_text)

    for page_idx, page in enumerate(pages):

        # ---- TABLES ----
        tables = extract_tables(page)
        for t_idx, table in enumerate(tables):
            table_sentences = table_to_sentences(table)

            for sent in table_sentences:
                all_chunks.append(sent)
                all_metadata.append({
                    "file": file,
                    "page": page_idx + 1,
                    "type": "table",
                    "table_id": t_idx
                })

        # ---- NORMAL TEXT ----
        clean_text = remove_tables(page)

        # Remove noisy sections
        clean_text = re.sub(r"IMAGES / INFOGRAPHICS:.*", "", clean_text, flags=re.DOTALL)

        chunks = chunk_text(clean_text)

        for chunk in chunks:
            if len(chunk.strip()) < 30:
                continue

            all_chunks.append(chunk)
            all_metadata.append({
                "file": file,
                "page": page_idx + 1,
                "type": "text"
            })


# =========================
# EMBEDDINGS
# =========================
print(f"Generating embeddings for {len(all_chunks)} chunks...")

embeddings = model.encode(
    all_chunks,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# =========================
# SAVE OUTPUT
# =========================

np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump([
        {"text": t, **m} for t, m in zip(all_chunks, all_metadata)
    ], f, indent=2)

print("✅ Done!")