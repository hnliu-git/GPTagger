from GPTagger.validators.base import BaseValidator


class LengthValidator(BaseValidator):
    def __init__(self, length: int, mode: str = "lt") -> None:
        if mode not in ["lt", "gt", "le", "ge"]:
            raise ValueError(
                f"{mode} not support, supproted modes are [lt, gt, le, ge]"
            )

        self.mode = mode
        self.length = length

    def __call__(self, text: str) -> bool:
        if self.mode == "lt" and len(text) < self.length:
            return True
        if self.mode == "gt" and len(text) > self.length:
            return True
        if self.mode == "le" and len(text) <= self.length:
            return True
        if self.mode == "ge" and len(text) >= self.length:
            return True

        return False
