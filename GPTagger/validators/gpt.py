import json
import tiktoken

from langchain.llms import OpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from GPTagger.validators.base import BaseValidator


class GPTValidator(BaseValidator):
    def __init__(
        self,
        template: PromptTemplate,
        model_name: str = "gpt-3.5-turbo",
        log_path: str = None,
    ) -> None:
        self.template = template
        self.model_name = model_name
        self.type = self.get_gpt_type(model_name)

        self.tkctr = 0
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # This log is only for training clf models
        # Don't use it when u don't need it
        self.log = open(log_path, "w") if log_path else None

    def __call__(self, text: str) -> bool:
        prompt = self.template.format(text=text)
        self.tkctr += len(self.enc.encode(prompt))
        resp = self.request_gpt(prompt)

        if self.log:
            log = {"text": text, "resp": resp}
            self.log.write(f"{json.dumps(log, ensure_ascii=False)}\n")

        if resp.strip() == "yes":
            return True
        else:
            return False

    def request_gpt(self, prompt: str) -> str:
        if self.type == "chat":
            model = ChatOpenAI(model=self.model_name, max_tokens=1)
            resp = model.predict_messages([HumanMessage(content=prompt)])
            return resp.content
        elif self.type == "comp":
            model = OpenAI(model=self.model_name, max_tokens=1)
            resp = model(prompt)
            return resp

    def get_gpt_type(self, model_name: str) -> str:
        if "gpt" in model_name:
            return "chat"
        else:
            return "comp"
