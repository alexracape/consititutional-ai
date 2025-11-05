import os, re
from typing import Dict, List
from dataclasses import dataclass

from huggingface_hub import InferenceClient


@dataclass
class JudgeResult:
    judgment: str
    score: float

_SCORE_RE = re.compile(r"SCORE:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

def _extract_score(text: str) -> float:
    m = _SCORE_RE.search(text)
    if m:
        s = float(m.group(1))
        return max(1.0, min(10.0, s))
    nums = re.findall(r'\b([1-9]|10)\b', text)
    return float(nums[-1]) if nums else 5.0

class Judge:
    def judge(self, messages: List[Dict[str, str]]) -> JudgeResult:
        raise NotImplementedError



class HFJudge(Judge):
    def __init__(self, model: str, timeout: int = 60):
        token = os.environ.get("HF_TOKEN")
        self.client = InferenceClient(model=model, token=token, timeout=timeout)
        self.model = model

    def judge(self, messages: List[Dict[str, str]]) -> JudgeResult:
        # Deterministic output for scoring
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        content = resp.choices[0].message.content
        return JudgeResult(judgment=content, score=_extract_score(content))
