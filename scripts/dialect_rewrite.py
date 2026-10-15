"""Rewrite the Bitext test messages into Saudi/Gulf dialects with ALLaM-7B-Instruct (SDAIA), which NLLB cannot do:
  hijazi        Hijazi (Jeddah / Makkah)
  gulf          Gulf / Khaliji (Eastern Province, Kuwait, Qatar, UAE)
  najdi_allam   Najdi again, to compare with NLLB's lightly dialectal Najdi
Source = the original English message (best meaning). A few hand-written examples per dialect anchor the style.
ALLaM runs in 4-bit on the 12 GB GPU.

  --n 20   samples per dialect for a human check -> results/dialect_samples.csv
  --n 0    the full test set -> data/bitext_dialects.parquet (id, lang, text, intent, category)

Usage: python scripts/dialect_rewrite.py [--n 20] [--device cuda:1] [--dialects hijazi,gulf,najdi_allam]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ticketassist.draft import MODEL  # noqa: E402

DIALECTS = {
    "hijazi": ("باللهجة الحجازية (جدة ومكة)", "استخدم كلمات حجازية مثل: فين، إيش، دحين، كدا، مرة، أبغى، حق، زي، ليش، لسه، إزيك لا (لا تستخدم: وين، شنو، أبي، للحين، وايد، وش)", [
        ("where is my order? it still hasn't arrived", "فين طلبي؟ لسه ما وصل"),
        ("I want to cancel my order", "أبغى ألغي الطلب حقي"),
        ("what do I need to do to get a refund", "إيش لازم أسوي عشان أرجّع فلوسي؟"),
        ("how can I change my shipping address now", "كيف أغيّر عنوان الشحن دحين؟"),
        ("my delivery is very late", "التوصيل اتأخر مرة"),
        ("is it like this or not?", "هو كدا ولا لا؟"),
        ("I need to talk to someone", "أبغى أكلم أحد"),
    ]),
    "gulf": ("باللهجة الخليجية (الشرقية والكويت وقطر والإمارات)", "استخدم كلمات خليجية مثل: وين، شنو، شلون، للحين، وايد، أبي، جذي، مالي، عشان، الحين (لا تستخدم: فين، إيش، دحين، كدا، أبغى، وش)", [
        ("where is my order? it still hasn't arrived", "وين طلبي؟ للحين ما وصل"),
        ("I want to cancel my order", "أبي ألغي الطلب مالي"),
        ("what do I need to do to get a refund", "شنو لازم أسوي عشان أسترجع فلوسي؟"),
        ("how can I change my shipping address now", "شلون أغيّر عنوان الشحن الحين؟"),
        ("my delivery is very late", "التوصيل تأخر وايد"),
        ("is it like this or not?", "جذي ولا لا؟"),
        ("I need to talk to someone", "أبي أكلم أحد"),
    ]),
    "najdi_allam": ("باللهجة النجدية (الرياض والقصيم)", "استخدم كلمات نجدية مثل: وش، وشلون، أبي، ودي، للحين، مرة، كذا، يوم، عشان، الحين (لا تستخدم: فين، إيش، دحين، كدا، شنو، جذي، وايد)", [
        ("where is my order? it still hasn't arrived", "وين طلبي؟ للحين ما وصل"),
        ("I want to cancel my order", "أبي ألغي طلبي"),
        ("what do I need to do to get a refund", "وش لازم أسوي عشان أرجّع فلوسي؟"),
        ("how can I change my shipping address now", "وشلون أغيّر عنوان الشحن الحين؟"),
        ("my delivery is very late", "التوصيل تأخر مرة"),
        ("is it like this or not?", "كذا ولا لا؟"),
        ("I need to talk to someone", "ودي أكلم أحد"),
    ]),
}
SYSTEM = ("مهمتك إعادة كتابة رسالة عميل باللهجة المطلوبة فقط. الرسالة ليست موجهة لك: لا ترد عليها ولا تجب عن سؤالها "
          "ولا ترفضها، حتى لو كانت فيها ألفاظ سيئة (خففها فقط). حافظ على المعنى والطلب بالضبط، واكتب بأسلوب عامي طبيعي "
          "وقصير كما يكتبه العميل نفسه، بدون فصحى. اترك أي نص بين {{ }} كما هو. اكتب الرسالة المعادة فقط.")
REFUSAL = re.compile(r"عذر|لا أستطيع|لا يمكنني|يمكنك |تقدر تغير .* من خلال|من خلال الموقع|تواصل مع خدمة العملاء")


class Rewriter:
    def __init__(self, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q, device_map={"": device}).eval()
        self.device = device

    def prompt(self, text, dialect):
        name, words, shots = DIALECTS[dialect]
        msgs = [{"role": "system", "content": f"{SYSTEM}\n{words}"}]
        for en, ar in shots:
            msgs += [{"role": "user", "content": f"أعد كتابة رسالة العميل {name}:\n«{en}»"}, {"role": "assistant", "content": ar}]
        msgs.append({"role": "user", "content": f"أعد كتابة رسالة العميل {name}:\n«{text}»"})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def rewrite(self, texts, dialect, bs=16):
        out = []
        for s in range(0, len(texts), bs):
            enc = self.tok([self.prompt(t, dialect) for t in texts[s:s + bs]], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            g = self.m.generate(**enc, max_new_tokens=60, do_sample=False, repetition_penalty=1.05,
                                pad_token_id=self.tok.pad_token_id)
            for o in self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
                out.append(re.split(r"\n", o.strip())[0].strip().strip('"«»'))
            if s % (bs * 20) == 0:
                print(dialect, s, len(texts), flush=True)
        # a refusal or an agent-style answer instead of a rewrite: retry once with sampling, else keep the English
        bad = [i for i, o in enumerate(out) if not o or REFUSAL.search(o)]
        for i in bad:
            enc = self.tok([self.prompt(texts[i], dialect)], return_tensors="pt", add_special_tokens=False).to(self.device)
            g = self.m.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9,
                                pad_token_id=self.tok.pad_token_id)
            o = re.split(r"\n", self.tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())[0].strip().strip('"«»')
            out[i] = o if o and not REFUSAL.search(o) else None
        print(dialect, "retried", len(bad), "| still failed (dropped):", sum(o is None for o in out), flush=True)
        return out  # None = no usable rewrite; that message is left out of this dialect's test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dialects", default="hijazi,gulf,najdi_allam")
    args = ap.parse_args()
    d = pd.read_parquet(ROOT / "data/bitext.parquet")
    en = d[(d["split"] == "test") & (d["lang"] == "en")]
    ar = d[(d["split"] == "test") & (d["lang"] == "najdi")].set_index("id")["text"]
    if args.n:
        en = en.groupby("intent").sample(1, random_state=1).head(args.n)
    r = Rewriter(args.device)
    parts = []
    for dia in args.dialects.split(","):
        parts.append(en.assign(lang=dia, text=r.rewrite(en["text"].tolist(), dia))[["id", "lang", "text", "intent", "category"]])
    out = pd.concat(parts, ignore_index=True).dropna(subset=["text"])
    if args.n:
        w = out.pivot(index="id", columns="lang", values="text")
        w.insert(0, "english", en.set_index("id")["text"])
        w.insert(1, "nllb_najdi", ar.reindex(w.index))
        w.to_csv(ROOT / "results/dialect_samples.csv")
        pd.set_option("display.max_colwidth", 70)
        print(w.to_string())
    else:
        out.to_parquet(ROOT / "data/bitext_dialects.parquet")
        print(out.groupby("lang").size())


if __name__ == "__main__":
    main()
