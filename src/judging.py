import os, re
from typing import Dict, List
from dataclasses import dataclass

from huggingface_hub import InferenceClient


@dataclass
class JudgeResult:
    judgment: str
    score: float


def _extract_score(text: str) -> float:
    # Get the last line with SCORE: X
    matches = list(re.finditer(r'(?i)SCORE:\s*(\d+(?:\.\d+)?)(?!\s*/)', text))
    if matches:
        s = float(matches[-1].group(1))  # take the last match
        return max(1.0, min(10.0, s))

    # Fallback: any integer 1..10 not part of a fraction, take the last
    nums = re.findall(r'(?<![\d/])\b(10|[1-9])\b(?!\s*/)', text)
    if nums:
        s = float(nums[-1])
        return max(1.0, min(10.0, s))

    return 5.0

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
