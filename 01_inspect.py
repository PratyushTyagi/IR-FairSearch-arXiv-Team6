"""Stream-inspect the arXiv JSONL dump.

Reports total rows, keys union, sample records, and identifies which fields
hold title / DOI / abstract.
"""
import json
import os
import sys
from collections import Counter

PATH = "/Users/pavanijain/Desktop/Kaggle_Preprocessing_Project/arxiv-metadata-oai-snapshot.json"


def main():
    size_bytes = os.path.getsize(PATH)
    print(f"File: {PATH}")
    print(f"Size: {size_bytes / 1e9:.2f} GB")

    n = 0
    keys_counter = Counter()
    samples = []
    doi_nonnull = 0
    title_nonnull = 0
    abstract_nonnull = 0
    id_nonnull = 0
    parse_errors = 0

    with open(PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            n += 1
            for k in obj.keys():
                keys_counter[k] += 1
            if obj.get("doi"):
                doi_nonnull += 1
            if obj.get("title"):
                title_nonnull += 1
            if obj.get("abstract"):
                abstract_nonnull += 1
            if obj.get("id"):
                id_nonnull += 1
            if len(samples) < 3:
                samples.append(obj)
            if i % 500_000 == 0 and i > 0:
                print(f"  ...streamed {i:,} lines so far")

    print(f"\nExact row count: {n:,}")
    print(f"Parse errors:    {parse_errors}")
    print(f"\nKeys (count of records that contain each key):")
    for k, c in keys_counter.most_common():
        print(f"  {k:20s} {c:>12,}")

    print(f"\nField non-null counts:")
    print(f"  id        non-null: {id_nonnull:,} / {n:,}")
    print(f"  title     non-null: {title_nonnull:,} / {n:,}")
    print(f"  abstract  non-null: {abstract_nonnull:,} / {n:,}")
    print(f"  doi       non-null: {doi_nonnull:,} / {n:,}  ({100*doi_nonnull/n:.1f}%)")

    print(f"\n=== 3 sample records ===")
    for i, s in enumerate(samples, 1):
        print(f"\n--- sample {i} ---")
        # truncate abstract for readability
        s_show = dict(s)
        if isinstance(s_show.get("abstract"), str) and len(s_show["abstract"]) > 300:
            s_show["abstract"] = s_show["abstract"][:300] + "..."
        print(json.dumps(s_show, indent=2)[:2000])


if __name__ == "__main__":
    main()
