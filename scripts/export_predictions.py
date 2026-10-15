"""Write every benchmark run in the course's prediction format (labs/nlp_utils.save_predictions):
eval/predictions/<system>.csv with ticket_id, lang, y_true, y_pred + slice columns, and <system>.json metadata.

Slices: testset (bitext_test | course_arabic | allam_dialects), lang (en | ar), dialect
(en, msa_nllb, najdi_nllb, msa_human, gulf_human, hijazi_allam, gulf_allam, najdi_allam).
Usage: python scripts/export_predictions.py [--price 0.30]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
B, OUT = ROOT / "results/bench", ROOT / "eval/predictions"
DIALECT = {"en": "en", "ar": "msa_nllb", "najdi": "najdi_nllb", "course_msa": "msa_human", "course_gulf": "gulf_human",
           "hijazi": "hijazi_allam", "gulf": "gulf_allam", "najdi_allam": "najdi_allam"}
TESTSET = {"en": "bitext_test", "ar": "bitext_test", "najdi": "bitext_test", "course_msa": "course_arabic",
           "course_gulf": "course_arabic", "hijazi": "allam_dialects", "gulf": "allam_dialects", "najdi_allam": "allam_dialects"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, default=0.30)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for info_file in sorted(B.glob("*.json")):
        info = json.load(open(info_file))
        m = info["method"]
        parts = [pd.read_parquet(B / f"{m}.parquet")]
        if (B / f"{m}_extra.parquet").exists():
            parts.append(pd.read_parquet(B / f"{m}_extra.parquet"))
        p = pd.concat(parts, ignore_index=True)
        df = pd.DataFrame({"ticket_id": p["id"], "lang": np.where(p["lang"] == "en", "en", "ar"),
                           "dialect": p["lang"].map(DIALECT), "testset": p["lang"].map(TESTSET),
                           "y_true": p["intent"], "y_pred": p["pred"]})
        df.to_csv(OUT / f"{m}.csv", index=False)
        lat = np.array(info["latency_ms"])
        meta = {"name": info["title"], "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                "cost_per_1k_usd": float(lat.mean()) * 1000 / 3.6e6 * args.price, "gpu": info["device"],
                "gpu_gb": info["gpu_gb"], "labels_used": info["labels"], "train_seconds": info["train_s"],
                "notes": f"cost = mean single-ticket latency x ${args.price}/h GPU rental (assumption); "
                         "Bitext intent, 27 classes; course_arabic = labs/data/arabic_tickets.csv (human-written)"}
        json.dump(meta, open(OUT / f"{m}.json", "w"), indent=2)
        print(m, len(df), df.groupby("dialect").size().to_dict())


if __name__ == "__main__":
    main()
