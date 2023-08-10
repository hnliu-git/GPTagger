import re

from GPTagger.validators.base import BaseValidator


class RegexValidator(BaseValidator):
    def __init__(self, regex: str) -> None:
        self.regex = regex

    def __call__(self, text: str) -> bool:
        if re.search(rf"{self.regex}", text):
            return True
        else:
            return False
