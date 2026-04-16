import re
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_DIR = Path("results")
TARGET_SORTERS = ["AISorter", "NoSorter", "SizeSorter"]


def parse_mmr_blocks(text: str):
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

        # Parse ONLY the MMR block
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


def make_plot(input_file: Path):
    text = input_file.read_text(encoding="utf-8")
    data = parse_mmr_blocks(text)

    if not data:
        print(f"No MMR data found in: {input_file}")
        return

    prefix_lengths = sorted(
        {prefix for sorter_data in data.values() for prefix in sorter_data.keys()}
    )

    style_map = {
        "AISorter": {"marker": "o", "linestyle": "-", "label": "AISorter"},
        "NoSorter": {"marker": "s", "linestyle": "--", "label": "NoSorter"},
        "SizeSorter": {"marker": "^", "linestyle": "-.", "label": "SizeSorter"},
    }

    output_file = input_file.with_name(f"{input_file.stem}_mmr_plot.png")

    plt.figure(figsize=(7.6, 6.4))

    for sorter in TARGET_SORTERS:
        if sorter not in data:
            continue

        y = [data[sorter].get(p) for p in prefix_lengths]
        style = style_map[sorter]

        plt.plot(
            prefix_lengths,
            y,
            linewidth=1.8,
            markersize=7,
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=style["label"],
        )

    plt.xlabel("Prefix Length", fontsize=16)
    plt.ylabel("Mean Reciprocal Rank", fontsize=16)
    plt.xticks(prefix_lengths, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="-", linewidth=1, alpha=0.6)
    plt.legend(title="Strategy", fontsize=11, title_fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {output_file}")


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {INPUT_DIR}")

    for txt_file in txt_files:
        make_plot(txt_file)


if __name__ == "__main__":
    main()