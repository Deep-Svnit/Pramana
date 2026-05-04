"""Vision extraction and page verification for RAG v2."""
from __future__ import annotations

import base64
import logging
import threading
import time
from io import BytesIO
from pathlib import Path
from time import perf_counter

logger = logging.getLogger(__name__)


def _progress(message: str) -> None:
    logger.info(message)
    print(f"[vision-v2] {message}", flush=True)


EXTRACTION_PROMPT = """\
You are a precise document parser for a RAG system. Analyse this single page image and extract ONLY information that is visibly present on the page.

Rules:
- Do not guess, infer, complete missing values, or use outside knowledge.
- Preserve numbers, units, currencies, percentages, signs, dates, and footnote markers exactly.
- If a value is unclear, write [unclear] instead of guessing.
- Extract every visible text line, including headers, footers, captions, legends, axis labels, notes, and source text.
- For tables, reproduce every row and column in markdown table format. Keep blank/unclear cells explicit.
- For charts and graphs, list the title, axes, legend, all visible data labels/values, and the visual trend only when it is directly observable.
- For images, diagrams, and infographics, describe visible labels, numbers, arrows, and relationships without adding interpretation.
- Keep page order: top-to-bottom, left-to-right.

Output ONLY extracted page content. Do not add commentary, preamble, or conclusions.
"""

PAGE_QA_PROMPT = """\
You are verifying one document page image against a user query. Use ONLY information visibly present on this page.

Rules:
- Do not use memory, outside knowledge, or assumptions.
- Extract exact numbers, labels, units, dates, and table/chart values needed for the answer.
- If the answer is present, answer concisely and cite this page as [p{page_number}].
- If the page does not contain the answer, output exactly: Not found in the Documents
- If a needed value is unreadable or absent, output exactly: Not found in the Documents

Query: {query}
"""

SUMMARY_PROMPT = """\
You are building a RAPTOR hierarchical retrieval index.

Summarize the child nodes below using ONLY their content. Preserve all important entities, dates, numbers, units, financial metrics, table values, chart values, and relationships. Do not invent missing data.

Return a dense factual summary suitable for retrieval. Prefer compact bullets. If there are conflicts or uncertainty, state them explicitly.

Child nodes:
{content}
"""


class RotatingGeminiVisionClient:
    """Gemini 2.5 Flash client that rotates API keys and respects per-key RPM."""

    def __init__(
        self,
        api_keys: list[str],
        model: str = "gemini-2.5-flash",
        rpm_per_key: int = 5,
        rpd_per_key: int = 20,
    ) -> None:
        try:
            from google import genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Run: pip install google-genai") from exc

        keys = [key.strip().strip('"') for key in api_keys if key and key.strip().strip('"')]
        if not keys:
            raise ValueError("No Gemini API keys found. Set GEMINI_1..GEMINI_5 or GEMINI_API_KEY.")

        self._clients = [genai.Client(api_key=key) for key in keys]
        self._model = model
        self._lock = threading.Lock()
        self._next = 0
        self._last_used = [0.0 for _ in keys]
        self._min_interval = 60.0 / max(1, rpm_per_key)
        self._daily_limit = max(1, rpd_per_key)
        self._request_counts = [0 for _ in keys]
        self._success_counts = [0 for _ in keys]
        self._failure_counts = [0 for _ in keys]
        self._failures: list[dict[str, object]] = []
        self._quota_exhausted = False
        _progress(
            f"Gemini rotating client ready model={model} keys={len(keys)} "
            f"rpm_per_key={rpm_per_key} rpd_per_key={rpd_per_key}"
        )

    def _client(self, operation: str):
        with self._lock:
            idx = -1
            for _ in range(len(self._clients)):
                candidate = self._next
                self._next = (self._next + 1) % len(self._clients)
                if self._request_counts[candidate] < self._daily_limit:
                    idx = candidate
                    break
            if idx == -1:
                self._quota_exhausted = True
                message = (
                    f"all Gemini key slots exhausted for operation={operation}; "
                    f"counts={self._request_counts} daily_limit={self._daily_limit}"
                )
                self._failures.append({
                    "operation": operation,
                    "key_slot": None,
                    "error": message,
                    "kind": "quota_exhausted",
                })
                raise RuntimeError(message)

            wait = self._min_interval - (time.monotonic() - self._last_used[idx])
            if wait > 0:
                _progress(f"Gemini key slot {idx + 1}: sleeping {wait:.1f}s for RPM limit")
                time.sleep(wait)
            self._last_used[idx] = time.monotonic()
            self._request_counts[idx] += 1
            _progress(
                f"Gemini key slot {idx + 1}: request starting "
                f"operation={operation} daily_used={self._request_counts[idx]}/{self._daily_limit}"
            )
            return idx, self._clients[idx]

    def _record_failure(self, operation: str, key_slot: int | None, error: Exception | str) -> None:
        if key_slot is not None:
            self._failure_counts[key_slot] += 1
        failure = {
            "operation": operation,
            "key_slot": None if key_slot is None else key_slot + 1,
            "error": str(error),
        }
        self._failures.append(failure)
        _progress(f"Gemini failure tracked: {failure}")

    def _generate_content(self, *, contents, operation: str) -> str:
        last_error: Exception | None = None
        for attempt in range(len(self._clients)):
            idx: int | None = None
            try:
                idx, client = self._client(operation)
                response = client.models.generate_content(model=self._model, contents=contents)
                text = response.text or ""
                self._success_counts[idx] += 1
                _progress(
                    f"Gemini key slot {idx + 1}: operation={operation} "
                    f"attempt={attempt + 1} success chars={len(text)}"
                )
                return text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._record_failure(operation, idx, exc)
                if self._quota_exhausted:
                    break
                _progress(
                    f"Gemini operation={operation} attempt={attempt + 1} failed; "
                    "trying next available key"
                )
        if last_error is not None:
            _progress(f"Gemini operation={operation} failed after retries: {last_error}")
        return ""

    def report(self) -> dict[str, object]:
        return {
            "model": self._model,
            "key_slots": len(self._clients),
            "daily_limit_per_key": self._daily_limit,
            "requests_by_key": list(self._request_counts),
            "successes_by_key": list(self._success_counts),
            "failures_by_key": list(self._failure_counts),
            "quota_exhausted": self._quota_exhausted,
            "failures": list(self._failures),
        }

    def extract(self, image_bytes: bytes) -> str:
        from google.genai import types  # type: ignore[import-untyped]

        start = perf_counter()
        text = self._generate_content(
            operation="visual_extract",
            contents=[
                types.Part.from_text(text=EXTRACTION_PROMPT),
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
        )
        _progress(
            f"Gemini image extraction returned {len(text)} chars "
            f"in {perf_counter() - start:.1f}s"
        )
        return text

    def answer_page(self, image_bytes: bytes, query: str, page_number: int) -> str:
        from google.genai import types  # type: ignore[import-untyped]

        prompt = PAGE_QA_PROMPT.format(query=query, page_number=page_number)
        text = self._generate_content(
            operation="visual_page_qa",
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
        )
        _progress(f"Gemini page QA returned {len(text)} chars")
        return text

    def summarize(self, child_texts: list[str], max_chars: int = 2400) -> str:
        content = "\n\n".join(
            f"[child {index}]\n{text[:max_chars]}"
            for index, text in enumerate(child_texts, start=1)
        )
        prompt = SUMMARY_PROMPT.format(content=content)
        start = perf_counter()
        text = self._generate_content(operation="summary", contents=prompt)
        _progress(
            f"Gemini RAPTOR summary returned {len(text)} chars "
            f"in {perf_counter() - start:.1f}s"
        )
        return text

    def is_available(self) -> bool:
        return True


def pdf_page_to_png_bytes(path: Path, page_number: int, dpi: int = 150) -> bytes:
    """Render a single PDF page (1-indexed) to PNG bytes using pdfplumber."""
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            page = pdf.pages[page_number - 1]
            img = page.to_image(resolution=dpi)
            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception as exc:
        logger.warning("Could not render page %d of %s: %s", page_number, path.name, exc)
        return b""


class QwenVisionClient:
    """Uses a local Qwen2.5-VL model to extract and verify page content."""

    def __init__(
        self,
        model: str,
        device: str = "auto",
        max_new_tokens: int = 2048,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        try:
            import torch  # type: ignore[import-untyped]
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError("Run: pip install transformers torch accelerate") from exc

        self._torch = torch
        self._device = device
        self._max_new_tokens = max_new_tokens

        model_kwargs: dict[str, object] = {"torch_dtype": "auto", "local_files_only": True}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"": device}

        processor_kwargs: dict[str, object] = {"local_files_only": True}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        load_start = perf_counter()
        _progress(
            f"loading Qwen VLM model={model} device={device} "
            f"max_new_tokens={max_new_tokens}"
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_kwargs)
        _progress(f"Qwen VLM weights loaded in {perf_counter() - load_start:.1f}s")
        processor_start = perf_counter()
        _progress(f"loading Qwen processor from {model}")
        self._processor = AutoProcessor.from_pretrained(model, **processor_kwargs)
        _progress(f"Qwen processor loaded in {perf_counter() - processor_start:.1f}s")

    def extract(self, image_bytes: bytes) -> str:
        return self.ask(image_bytes=image_bytes, prompt=EXTRACTION_PROMPT)

    def answer_page(self, image_bytes: bytes, query: str, page_number: int) -> str:
        prompt = PAGE_QA_PROMPT.format(query=query, page_number=page_number)
        return self.ask(image_bytes=image_bytes, prompt=prompt, max_new_tokens=512)

    def ask(self, image_bytes: bytes, prompt: str, max_new_tokens: int | None = None) -> str:
        from PIL import Image

        total_start = perf_counter()
        _progress(f"preparing image prompt image_bytes={len(image_bytes)}")
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        _progress(f"image opened size={image.size[0]}x{image.size[1]}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        if self._device != "auto":
            inputs = inputs.to(self._device)
        elif self._torch.cuda.is_available():
            inputs = inputs.to("cuda")

        _progress(
            "Qwen generation started "
            f"input_tokens={int(inputs.input_ids.shape[-1])} "
            f"max_new_tokens={max_new_tokens or self._max_new_tokens}"
        )
        generation_start = perf_counter()
        with self._torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self._max_new_tokens,
                do_sample=False,
            )
        _progress(f"Qwen generation finished in {perf_counter() - generation_start:.1f}s")
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = (output[0] if output else "").strip()
        _progress(
            f"Qwen decoded {len(text)} chars total_elapsed={perf_counter() - total_start:.1f}s"
        )
        return text

    def is_available(self) -> bool:
        return True


class GeminiVisionClient:
    """Legacy Gemini client retained for older configurations."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        try:
            from google import genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Run: pip install google-genai") from exc
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def extract(self, image_bytes: bytes) -> str:
        from google.genai import types  # type: ignore[import-untyped]

        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_text(text=EXTRACTION_PROMPT),
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
        )
        return response.text or ""

    def is_available(self) -> bool:
        return True

    def summarize(self, child_texts: list[str], max_chars: int = 2400) -> str:
        content = "\n\n".join(
            f"[child {index}]\n{text[:max_chars]}"
            for index, text in enumerate(child_texts, start=1)
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=SUMMARY_PROMPT.format(content=content),
        )
        return response.text or ""


class OllamaVisionClient:
    """Calls a vision model running locally via Ollama."""

    def __init__(self, model: str = "llava", base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    def extract(self, image_bytes: bytes) -> str:
        import requests  # type: ignore[import-untyped]

        b64 = base64.b64encode(image_bytes).decode()
        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": EXTRACTION_PROMPT,
                    "images": [b64],
                    "stream": False,
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as exc:
            logger.error("Ollama vision call failed: %s", exc)
            return ""

    def is_available(self) -> bool:
        try:
            import requests  # type: ignore[import-untyped]

            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self._model.split(":")[0] in m for m in models)
        except Exception:
            return False


def build_vision_client(provider: str, **kwargs):
    """
    provider: "gemini" | "qwen" | "ollama" | "none"
    kwargs for qwen: model, device, max_new_tokens, min_pixels, max_pixels
    kwargs for gemini: api_key, model
    kwargs for ollama: model, base_url
    """
    if provider == "gemini":
        api_keys = kwargs.get("api_keys") or []
        if api_keys:
            return RotatingGeminiVisionClient(
                api_keys=list(api_keys),
                model=kwargs.get("model", "gemini-2.5-flash"),
                rpm_per_key=kwargs.get("rpm_per_key", 5),
                rpd_per_key=kwargs.get("rpd_per_key", 20),
            )
        return GeminiVisionClient(
            api_key=kwargs["api_key"],
            model=kwargs.get("model", "gemini-2.5-flash"),
        )
    if provider == "qwen":
        return QwenVisionClient(
            model=kwargs.get("model", r"D:\PowerMind\Qwen_VL"),
            device=kwargs.get("device", "auto"),
            max_new_tokens=kwargs.get("max_new_tokens", 2048),
            min_pixels=kwargs.get("min_pixels"),
            max_pixels=kwargs.get("max_pixels"),
        )
    if provider == "ollama":
        return OllamaVisionClient(
            model=kwargs.get("model", "llava"),
            base_url=kwargs.get("base_url", "http://localhost:11434"),
        )
    return None


def extract_page_with_vision(
    client,
    path: Path,
    page_number: int,
    dpi: int = 150,
) -> str | None:
    """Render `page_number` of `path` to an image and call the vision client."""
    if client is None:
        return None
    img_bytes = pdf_page_to_png_bytes(path, page_number, dpi)
    if not img_bytes:
        logger.warning("[VLM] Page %d of '%s' - could not render to image", page_number, path.name)
        return None
    try:
        logger.info("[VLM] Page %d of '%s' - sending to vision model", page_number, path.name)
        text = client.extract(img_bytes)
        if text and len(text.strip()) > 20:
            logger.info("[VLM] Page %d of '%s' - extracted %d chars", page_number, path.name, len(text))
            return text
        logger.warning("[VLM] Page %d of '%s' - model returned empty/short text", page_number, path.name)
    except Exception as exc:
        logger.warning("[VLM] Page %d of '%s' - extraction failed: %s", page_number, path.name, exc)
    return None
