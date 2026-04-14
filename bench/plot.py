import re
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_FILE = Path("resutls/nec.txt")
OUTPUT_FILE = Path("resutls/nec_mrr_plot.png")

TARGET_SORTERS = ["AISorter", "NoSorter", "SizeSorter"]


def parse_mrr_blocks(text: str):
    results = {}

    section_pattern = re.compile(
        r"###\s*(?P<section>.*?)\s*###(?P<body>.*?)(?=(?:\n###\s*.*?\s*###)|\Z)",
        re.DOTALL,
    )

    for match in section_pattern.finditer(text):
        section_name = match.group("section").strip()
        body = match.group("body")

        sorter_match = re.search(r"Sorter:\s*([^\n\r]+)", body)
        sorter_name = sorter_match.group(1).strip() if sorter_match else section_name

        if sorter_name not in TARGET_SORTERS:
            continue

        mmr_block_match = re.search(
            r"MMR\s*\n\s*Prefix\s*\|\s*Mean Reciprocal Rank\s*\n(?P<table>.*?)(?=\n\s*NDCG\b|\Z)",
            body,
            re.DOTALL,
        )
        if not mmr_block_match:
            continue

        table = mmr_block_match.group("table")
        mmr_values = {}

        for line in table.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 2:
                continue

            try:
                prefix = int(parts[0])
                mmr = float(parts[1])
                mmr_values[prefix] = mmr
            except ValueError:
                continue

        if mmr_values:
            results[sorter_name] = mmr_values

    return results


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    text = INPUT_FILE.read_text(encoding="utf-8")
    data = parse_mrr_blocks(text)

    if not data:
        raise ValueError("No MRR data found in the input file.")

    prefix_lengths = sorted(
        {prefix for sorter_data in data.values() for prefix in sorter_data.keys()}
    )

    style_map = {
        "AISorter": {"marker": "o", "linestyle": "-", "label": "AISorter"},
        "NoSorter": {"marker": "s", "linestyle": "--", "label": "NoSorter"},
        "SizeSorter": {"marker": "^", "linestyle": "-.", "label": "SizeSorter"},
    }

    plt.figure(figsize=(7.6, 6.4))

    for sorter in TARGET_SORTERS:
        if sorter not in data:
            continue

        y = [data[sorter].get(p) for p in prefix_lengths]
        style = style_map[sorter]

        plt.plot(
            prefix_lengths,
            y,
            color="black",
            linewidth=1.8,
            markersize=7,
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=style["label"],
        )

    plt.xlabel("Prefix Length", fontsize=16)
    plt.ylabel("Average MRR", fontsize=16)
    plt.xticks(prefix_lengths, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.1, 0.9)
    plt.grid(True, linestyle="-", linewidth=1, alpha=0.6)
    plt.legend(title="Strategy", fontsize=11, title_fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()