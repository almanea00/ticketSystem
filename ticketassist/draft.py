"""ALLaM-7B-Instruct (SDAIA, Apache-2.0) writes the reply, grounded in the retrieved past tickets.

d1-3B cannot generate text, so drafting needs its own model. On a 12 GB GPU ALLaM loads in 4-bit (NF4);
on a 24 GB GPU in bf16.
"""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ALLaM-AI/ALLaM-7B-Instruct-preview"
LANG_NAME = {"en": "English", "ar": "Modern Standard Arabic", "najdi": "Saudi (Najdi) Arabic"}
ARABIC = re.compile(r"[؀-ۿ]")

SYSTEM = ("You are a customer support agent. Write a short, polite reply to the new ticket in {lang}. "
          "Use ONLY the facts, steps and policies found in the past tickets' resolutions below, and cite each past "
          "ticket you use as [ID]. Do not invent order numbers, dates, prices or promises. If the past tickets "
          "do not cover the problem, ask the customer for the specific details needed instead.")


def language_of(text: str) -> str:
    return "ar" if len(ARABIC.findall(text)) > 0.3 * max(1, len(re.findall(r"\w", text))) else "en"


class Drafter:
    def __init__(self, device: str = "cuda", four_bit: bool | None = None):
        if four_bit is None:  # 4-bit when the GPU has less than ~20 GB
            four_bit = torch.cuda.get_device_properties(torch.device(device)).total_memory < 20 * 2**30
        kw = {"dtype": torch.bfloat16}
        if four_bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": device}, **kw).eval()
        self.device = device

    def messages(self, subject, body, cases, lang):
        past = "\n\n".join(f"[{c['id']}] ({c['queue']}) {c['subject']}\nCustomer: {c['body']}\nResolution: {c['answer']}"
                           for c in cases) or "(no similar past tickets found)"
        return [{"role": "system", "content": SYSTEM.format(lang=LANG_NAME.get(lang, "English"))},
                {"role": "user", "content": f"PAST TICKETS:\n{past}\n\nNEW TICKET:\nSubject: {subject}\n{body}\n\nReply:"}]

    @torch.no_grad()
    def draft(self, subject: str, body: str, cases: list[dict], lang: str | None = None, max_new_tokens: int = 300) -> str:
        lang = lang or language_of(subject + " " + body)
        ids = self.tok.apply_chat_template(self.messages(subject, body, cases, lang), add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(self.device)
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
