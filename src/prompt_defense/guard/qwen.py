###
# Code based largely on Qwen's example code at:
# https://qwen.ai/blog?id=f0bbad0677edf58ba93d80a1e12ce458f7a80548&from=research.research-list
###

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import re

from .base import BaseGuard

class QwenGuard(BaseGuard):

    def __init__(self):
        super().__init__()

        # load the tokenizer and the model
        model_name = "Qwen/Qwen3Guard-Gen-4B"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def extract_label_and_categories(self, content):
        safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
        category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
        safe_label_match = re.search(safe_pattern, content)
        label = safe_label_match.group(1) if safe_label_match else None
        categories = re.findall(category_pattern, content)
        return label, categories

    def has_probs(self) -> bool:
        return False

    def detect_jailbreak(self, prompt: str) -> Tuple[bool, float]:

        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        # conduct text completion
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=128
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        safe_label, categories = self.extract_label_and_categories(content)
        
        return [safe_label == "Unsafe", None]