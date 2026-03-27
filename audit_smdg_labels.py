import argparse
import os
from collections import Counter

import pandas as pd


def list_valid_pairs(img_dir: str, roi_dir: str):
    if not os.path.isdir(img_dir) or not os.path.isdir(roi_dir):
        return []
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))])
    return [f for f in files if os.path.exists(os.path.join(roi_dir, f))]


def build_report(root: str, csv_path: str):
    neg_dir = os.path.join(root, "0_neg")
    neg_roi_dir = os.path.join(root, "0_roi_800_clahe")
    pos_dir = os.path.join(root, "1_pos")
    pos_roi_dir = os.path.join(root, "1_roi_800_clahe")

    neg_files = list_valid_pairs(neg_dir, neg_roi_dir)
    pos_files = list_valid_pairs(pos_dir, pos_roi_dir)

    df = pd.read_csv(csv_path)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("CSV must contain columns: x, y")

    csv_map = df.set_index("x")["y"].to_dict()
    csv_dups = int(df["x"].duplicated().sum())

    rows = []
    for f in neg_files:
        rows.append(
            {
                "filename": f,
                "subset": "0_neg",
                "img_exists": True,
                "roi_exists": True,
                "in_csv": f in csv_map,
                "csv_y": csv_map.get(f),
                "assigned_label": 0,
            }
        )

    missing_csv_for_pos = []
    for f in pos_files:
        in_csv = f in csv_map
        label = int(csv_map[f]) if in_csv else None
        if not in_csv:
            missing_csv_for_pos.append(f)
        rows.append(
            {
                "filename": f,
                "subset": "1_pos",
                "img_exists": True,
                "roi_exists": True,
                "in_csv": in_csv,
                "csv_y": label,
                "assigned_label": label,
            }
        )

    report_df = pd.DataFrame(rows)
    report_df = report_df.sort_values(["subset", "filename"]).reset_index(drop=True)

    label_counter = Counter(report_df["assigned_label"].dropna().astype(int).tolist())
    csv_counter = df["y"].value_counts().sort_index().to_dict()

    extra_csv_not_used = sorted(set(df["x"].tolist()) - set(pos_files))

    summary = {
        "neg_valid_pairs": len(neg_files),
        "pos_valid_pairs": len(pos_files),
        "total_valid_pairs": len(neg_files) + len(pos_files),
        "dataset_label_counts": dict(sorted(label_counter.items())),
        "csv_rows": len(df),
        "csv_label_counts": csv_counter,
        "csv_duplicated_filenames": csv_dups,
        "pos_files_missing_in_csv": len(missing_csv_for_pos),
        "csv_files_not_used_by_pos": len(extra_csv_not_used),
    }

    return summary, report_df, missing_csv_for_pos, extra_csv_not_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/ZYZ/GRINLAB/SMDG_test/test")
    parser.add_argument("--csv", default="/root/ZYZ/GRINLAB/SMDG_test/test/smdg_relabel.csv")
    parser.add_argument("--outdir", default="/root/ZYZ/GRINLAB/audit_outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    summary, report_df, missing_csv_for_pos, extra_csv_not_used = build_report(args.root, args.csv)

    report_path = os.path.join(args.outdir, "smdg_file_label_audit.csv")
    miss_path = os.path.join(args.outdir, "pos_missing_in_csv.txt")
    extra_path = os.path.join(args.outdir, "csv_not_used_by_pos.txt")

    report_df.to_csv(report_path, index=False)
    with open(miss_path, "w", encoding="utf-8") as f:
        f.write("\n".join(missing_csv_for_pos))
    with open(extra_path, "w", encoding="utf-8") as f:
        f.write("\n".join(extra_csv_not_used))

    print("=== SMDG label audit summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"audit_table: {report_path}")
    print(f"missing_pos_in_csv: {miss_path}")
    print(f"csv_not_used_by_pos: {extra_path}")


if __name__ == "__main__":
    main()
