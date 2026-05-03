from __future__ import annotations

from typing import Any


class LettuceClaimVerifier:
    def __init__(self):
        try:
            from lettucedetect import LettuceDetect
        except ImportError as exc:
            raise RuntimeError("LettuceDetect is mandatory. Install lettucedetect; HalluGuard is not allowed.") from exc
        self.detector = LettuceDetect()

    def verify(self, answer: str, evidence: list[str]) -> dict[str, Any]:
        context = "\n\n".join(evidence)
        if hasattr(self.detector, "predict"):
            result = self.detector.predict(context=context, answer=answer)
        elif hasattr(self.detector, "detect"):
            result = self.detector.detect(answer=answer, context=context)
        else:
            raise RuntimeError("Unsupported LettuceDetect API: expected predict() or detect().")
        if isinstance(result, dict):
            return result
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "__dict__"):
            return dict(result.__dict__)
        return {"raw": result}

    @staticmethod
    def has_unsupported_content(report: dict[str, Any]) -> bool:
        for key in ("unsupported_spans", "unsupported_claims", "hallucinated_spans"):
            value = report.get(key)
            if value:
                return True
        for key in ("verdict", "label", "status"):
            value = str(report.get(key, "")).lower()
            if value in {"unsupported", "hallucination", "hallucinated", "not_supported"}:
                return True
        for key in ("is_supported", "supported", "faithful"):
            if key in report:
                return not bool(report[key])
        return False
