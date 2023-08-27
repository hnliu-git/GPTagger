import gradio as gr

from pathlib import Path
from langchain.prompts import PromptTemplate

from GPTagger import *

default_prompt = """
Please understand the instructions above and do extraction in the text below.

TEXT:
\"\"\"
{text}
\"\"\"
"""


def ner(
    model: str,
    nr_call: int,
    tag_name: str,
    tag_max_len: int,
    text: str,
    prompt: str,
):
    cfg = NerConfig(
        tag_name=tag_name,
        model=model,
        nr_calls=nr_call,
        tag_max_len=tag_max_len,
    )

    ner_pipeline = NerPipeline.from_config(cfg)
    template = PromptTemplate.from_template(prompt)

    extractions = ner_pipeline(text, template, "")

    if not extractions:
        output = []
    else:
        output = [
            {"entity": tag_name.upper(), "start": item.start, "end": item.end}
            for item in extractions
        ]
    return {"text": text, "entities": output}


with gr.Blocks(theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg)) as demo:
    with gr.Row():
        tag_name = gr.Textbox(label="tag name")
        tag_max_len = gr.Slider(
            minimum=10, maximum=1000, step=10, label="max length of the tag"
        )
    with gr.Row():
        model = gr.Dropdown(
            ["gpt-3.5-turbo-0613", "gpt-4-0613"],
            label="model_name",
            value="gpt-3.5-turbo-0613",
        )
        nr_call = gr.Number(label="nr_of_calls", minimum=1, value=1, precision=0)
    with gr.Row():
        prompt = gr.TextArea(
            placeholder="Enter your prompt here...",
            label="prompt",
            value=default_prompt,
        )
        text = gr.TextArea(placeholder="Enter your text here...", label="text")
    btn = gr.Button("Submit")
    output = gr.HighlightedText()
    btn.click(
        ner,
        inputs=[
            model,
            nr_call,
            tag_name,
            tag_max_len,
            text,
            prompt,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
