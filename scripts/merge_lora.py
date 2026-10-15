"""Merge the d1 LoRA into d1-3B's weights and save a standalone checkpoint that vLLM (or any Lfm2-VL loader) serves:
W <- W + (alpha / r) * B @ A for every adapted layer, all other tensors and files copied unchanged.
Usage: python scripts/merge_lora.py [--adapter runs/d1_lora_all] [--out runs/d1_merged_all]"""
import argparse, json, shutil
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="runs/d1_lora_all")
ap.add_argument("--out", default="runs/d1_merged_all")
args = ap.parse_args()
src = Path(snapshot_download("LiquidAI/d1-3B"))
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / args.adapter / "lora.json"))
scale = cfg["alpha"] / cfg["r"]
lora = torch.load(ROOT / args.adapter / "lora.pt")
w = load_file(src / "model.safetensors")
n = 0
for k in [k for k in lora if ".lora_A." in k]:
    base = k.replace(".lora_A.default.weight", ".weight")
    A, B = lora[k].float(), lora[k.replace("lora_A", "lora_B")].float()
    W = w[base]
    w[base] = (W.float() + scale * (B @ A)).to(W.dtype)
    n += 1
save_file(w, out / "model.safetensors", metadata={"format": "pt"})
for f in src.iterdir():   # config, tokenizer, chat template, d1's own code (for transformers), license
    if f.name != "model.safetensors" and f.is_file():
        shutil.copy(f, out / f.name)
json.dump({**cfg, "merged_into": "LiquidAI/d1-3B", "layers_merged": n}, open(out / "merge_info.json", "w"), indent=1, ensure_ascii=False)
print("merged", n, "layers ->", out)
