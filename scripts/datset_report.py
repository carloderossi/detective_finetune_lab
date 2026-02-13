import os
import json
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns


DATA_FILE = "detective_finetune.jsonl"
REPORTS_DIR = Path("reports")
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "Alibaba-NLP/gte-large-en-v1.5"


def load_config():
    cfg_path = Path("config.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_data():
    rows = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "author": obj.get("author", ""),
                    "book": obj.get("book", ""),
                    "chapter": obj.get("chapter", ""),
                    "text": obj.get("text", ""),
                }
            )
    df = pd.DataFrame(rows)
    return df


def compute_qa_stats(df, tokenizer):
    # Empty entries
    df["text_stripped"] = df["text"].fillna("").apply(lambda x: x.strip())
    empty_mask = df["text_stripped"].str.len() < 10
    empty_entries = df[empty_mask]

    # Duplicates by text
    dup_mask = df["text"].duplicated(keep=False)
    dup_entries = df[dup_mask].copy()

    # Average token length
    def token_len(t):
        return len(tokenizer.encode(t))

    token_lengths = df["text"].fillna("").apply(token_len)
    avg_token_len = float(token_lengths.mean())

    stats = {
        "num_entries": len(df),
        "empty_entries": empty_entries,
        "duplicate_entries": dup_entries,
        "avg_token_length": avg_token_len,
        "token_lengths": token_lengths,
    }
    return stats


def build_embeddings(df, embedder, excerpt_len=512):
    def excerpt(text):
        if not isinstance(text, str):
            return ""
        return text[:excerpt_len]

    texts = df["text"].fillna("").apply(excerpt).tolist()
    embeddings = embedder.encode(texts, convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.asarray(embeddings)
    return embeddings


def plot_clusters(df, embeddings, out_path):
    # Map authors to labels
    def label_author(a):
        a = (a or "").lower()
        if "arthur_conan_doyle" in a:
            return "Holmes (Doyle)"
        if "agatha_christie" in a:
            return "Poirot (Christie)"
        return "Other"

    labels = df["author"].apply(label_author).tolist()

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(embeddings)

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    palette = sns.color_palette("Set2", n_colors=len(unique_labels))
    color_map = {lab: palette[i] for i, lab in enumerate(unique_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PCA plot
    ax = axes[0]
    for lab in unique_labels:
        mask = [l == lab for l in labels]
        ax.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            s=10,
            color=color_map[lab],
            label=lab,
            alpha=0.7,
        )
    ax.set_title("PCA – Style Clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=2, fontsize=8)

    # UMAP plot
    ax = axes[1]
    for lab in unique_labels:
        mask = [l == lab for l in labels]
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            s=10,
            color=color_map[lab],
            label=lab,
            alpha=0.7,
        )
    ax.set_title("UMAP – Style Clusters")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(markerscale=2, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_drift_heatmap(df, embedder, out_path, max_chapters=50, excerpt_len=512):
    # Select Holmes and Poirot chapters
    holmes_df = df[df["author"] == "arthur_conan_doyle"].copy()
    poirot_df = df[df["author"] == "agatha_christie"].copy()

    holmes_df = holmes_df.head(max_chapters).reset_index(drop=True)
    poirot_df = poirot_df.head(max_chapters).reset_index(drop=True)

    def excerpt(text):
        if not isinstance(text, str):
            return ""
        return text[:excerpt_len]

    holmes_texts = holmes_df["text"].fillna("").apply(excerpt).tolist()
    poirot_texts = poirot_df["text"].fillna("").apply(excerpt).tolist()

    if len(holmes_texts) == 0 or len(poirot_texts) == 0:
        print("Not enough Holmes or Poirot chapters for drift heatmap.")
        return

    holmes_emb = embedder.encode(holmes_texts, convert_to_tensor=True, show_progress_bar=True)
    poirot_emb = embedder.encode(poirot_texts, convert_to_tensor=True, show_progress_bar=True)

    # Compute cosine similarity matrix
    sim_matrix = util.cos_sim(holmes_emb, poirot_emb).cpu().numpy()

    # Build labels
    holmes_labels = [
        f"H-{i+1}:{row['chapter']}" for i, (_, row) in enumerate(holmes_df.iterrows())
    ]
    poirot_labels = [
        f"P-{i+1}:{row['chapter']}" for i, (_, row) in enumerate(poirot_df.iterrows())
    ]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        sim_matrix,
        xticklabels=poirot_labels,
        yticklabels=holmes_labels,
        cmap="viridis",
        cbar_kws={"label": "Cosine similarity"},
    )
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.title("Chapter-level Holmes–Poirot Drift (Cosine Similarity)")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def write_markdown_report(df, stats, out_path):
    num_entries = stats["num_entries"]
    empty_entries = stats["empty_entries"]
    dup_entries = stats["duplicate_entries"]
    avg_token_len = stats["avg_token_length"]

    # Counts per author and book
    author_counts = df["author"].value_counts()
    book_counts = df["book"].value_counts()

    lines = []
    lines.append("# Detective Dataset QA Report\n")
    lines.append(f"- Total entries: **{num_entries}**")
    lines.append(f"- Average token length: **{avg_token_len:.2f}**\n")

    if avg_token_len > 4000:
        interp = "Entries are very long (chapter-level). Consider shorter segments for some tasks."
    elif avg_token_len > 1000:
        interp = "Entries are long; this is typical for chapter-level data."
    else:
        interp = "Entries are relatively short; this is suitable for many fine-tuning tasks."
    lines.append(f"> Interpretation: {interp}\n")

    lines.append("## Counts per author\n")
    lines.append("| Author | Count |")
    lines.append("|--------|-------|")
    for author, count in author_counts.items():
        lines.append(f"| {author} | {count} |")
    lines.append("")

    lines.append("## Counts per book\n")
    lines.append("| Book | Count |")
    lines.append("|------|-------|")
    for book, count in book_counts.items():
        lines.append(f"| {book} | {count} |")
    lines.append("")

    # Empty entries
    lines.append("## Empty entries\n")
    lines.append(f"- Number of empty entries (len(text.strip()) < 10): **{len(empty_entries)}**\n")
    if len(empty_entries) > 0:
        lines.append("| Author | Book | Chapter |")
        lines.append("|--------|------|---------|")
        for _, row in empty_entries.head(10).iterrows():
            lines.append(
                f"| {row['author']} | {row['book']} | {row['chapter']} |"
            )
        lines.append("")

    # Duplicate entries
    lines.append("## Duplicate texts\n")
    lines.append(f"- Number of entries with duplicate text: **{len(dup_entries)}**\n")
    if len(dup_entries) > 0:
        lines.append("| Author | Book | Chapter | Text (truncated) |")
        lines.append("|--------|------|---------|------------------|")
        for _, row in dup_entries.head(10).iterrows():
            txt = (row["text"] or "").replace("\n", " ")
            if len(txt) > 80:
                txt = txt[:77] + "..."
            lines.append(
                f"| {row['author']} | {row['book']} | {row['chapter']} | {txt} |"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    _ = load_config()  # currently unused, but keeps structure ready

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} entries.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Computing QA stats...")
    stats = compute_qa_stats(df, tokenizer)

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)

    print("Building embeddings for clustering...")
    embeddings = build_embeddings(df, embedder)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    clusters_path = REPORTS_DIR / "style_clusters.png"
    print(f"Plotting clusters to {clusters_path}...")
    plot_clusters(df, embeddings, clusters_path)

    drift_path = REPORTS_DIR / "chapter_drift_heatmap.png"
    print(f"Plotting chapter-level drift heatmap to {drift_path}...")
    plot_drift_heatmap(df, embedder, drift_path)

    report_path = REPORTS_DIR / "dataset_qa_report.md"
    print(f"Writing Markdown report to {report_path}...")
    write_markdown_report(df, stats, report_path)

    print("Done.")


if __name__ == "__main__":
    main()