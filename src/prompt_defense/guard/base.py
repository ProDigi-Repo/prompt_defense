from typing import Tuple

class BaseGuard:

    def __init__(self):
        pass

    def has_probs(self) -> bool:
        """
        Returns whether a model outputs scores for their classification.
        """
        pass

    def detect_jailbreak(self, prompt: str) -> Tuple[bool, float]:
        """
        Uses an ML model to detect jailbreaking / prompt injections. Models
        that output just decisions fill the bool value whereas models that
        return probabilities (check with `has_probs`) fill the float value
        as well.
        """
        pass