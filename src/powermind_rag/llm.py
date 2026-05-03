from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalQwen:
    def __init__(self, model_path: Path, device: str):
        self.model_path = model_path
        self.device = device
        self._validate_model_files()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model_kwargs = {
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
        if device == "cpu":
            self.model.to("cpu")

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        target_device = next(self.model.parameters()).device
        inputs = self.tokenizer([prompt], return_tensors="pt").to(target_device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        generated = generated[:, inputs.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def _validate_model_files(self) -> None:
        if not self.model_path.exists():
            raise RuntimeError(f"Qwen model path does not exist: {self.model_path}")
        weight_files = list(self.model_path.glob("*.safetensors")) + list(self.model_path.glob("*.bin"))
        if not weight_files:
            partial = list(self.model_path.glob("*.crdownload"))
            hint = " Partial browser downloads are present." if partial else ""
            raise RuntimeError(f"No completed Qwen weight files found in {self.model_path}.{hint}")


class PropositionChunker:
    def __init__(self, llm: LocalQwen):
        self.llm = llm

    def chunk(self, page_text: str, table_markdown: str, doc_type: str, section: str, context: str) -> list[str]:
        source = "\n\n".join(part for part in [page_text, table_markdown] if part.strip())
        if not source.strip():
            return []
        user = f"""
Convert the source into atomic factual propositions for retrieval.
Preserve exact numbers, units, row-column relationships, periods, qualifiers, and exceptions.
Do not infer facts. Return only a JSON array of strings.

Document type: {doc_type}
Section: {section}
Context: {context}

SOURCE:
{source}
""".strip()
        raw = self.llm.generate(
            system="You produce faithful JSON only. No markdown fences.",
            user=user,
            max_new_tokens=1200,
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Proposition chunking did not return valid JSON: {raw[:500]}") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise RuntimeError("Proposition chunking must return a JSON array of strings.")
        return [item.strip() for item in parsed if item.strip()]
