import json
import jinja2

from pathlib import Path
from typing import Union
from langchain.prompts import PromptTemplate


class BaseTemplate:
    """A class that may or may not help you with prompt versioning
    """
    def __init__(
        self,
        name: str,
        template_path: Union[Path, str] = None,
        template: PromptTemplate = None,
        memo_version: str = "v0",
        **kwargs,
    ) -> None:
        self.memo_version = memo_version

        if not template and not template_path:
            raise ValueError("Either `template` or `template_path` should not be None")

        if template_path:
            kwargs["template_path"] = str(template_path)

            template_path = BaseTemplate.path_wrapper(template_path)
            template = jinja2.Template(template_path.read_text())
            prompt = template.render(**kwargs)

            save_path = template_path.parent
            save_path.mkdir(exist_ok=True)
            self.save_prompt(save_path / "prompts" / f"{name}.prompt", prompt)
            self.save_arguments(save_path / "prompts" / f"{name}.json", **kwargs)

            template = PromptTemplate.from_template(template=prompt)

        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    # When you want to work with a fixed template
    @classmethod
    def from_prompt(cls, path: Union[Path, str]) -> "BaseTemplate":
        path = cls.path_wrapper(path)
        memo = path.stem.split("-")[-1]
        name = path.stem.split("-")[0]
        return cls(
            name=name,
            template=PromptTemplate.from_template(path.read_text()),
            memo_version=memo,
        )

    # When you want to change something in a template
    @classmethod
    def from_args_json(cls, path: Union[Path, str], **kwargs) -> "BaseTemplate":
        path = cls.path_wrapper(path)
        memo = path.stem.split("-")[-1]
        name = path.stem.split("-")[0]
        args = json.loads(path.read_text())
        for k, _ in args.items():
            if k in kwargs:
                args[k] = kwargs[k]
        return cls(name, memo_version=memo, **args)

    @classmethod
    def from_text(cls, name: str, text: str, memo: str = "v0") -> "BaseTemplate":
        return cls(
            name=name, template=PromptTemplate.from_template(text), memo_version=memo
        )

    def save_prompt(self, path: Path, template: str):
        path = path.with_stem(f"{path.stem}-{self.memo_version}")
        path.write_text(template)

    def save_arguments(self, path: Path, **kwargs):
        path = path.with_stem(f"{path.stem}-{self.memo_version}")
        path.write_text(json.dumps(kwargs, ensure_ascii=False, indent=True))

    @classmethod
    def path_wrapper(cls, path: Union[Path, str]) -> Path:
        if isinstance(path, str):
            return Path(path)
        elif isinstance(path, Path):
            return path
