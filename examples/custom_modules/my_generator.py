"""自訂 generation 範例:接自家(非 OpenAI 相容)推論服務的骨架。

契約:generate(messages: list[BaseMessage]) -> str
(messages 是框架組好的 prompt;query() 回傳的 prompt 即其內容)

掛法::

    generation:
      method: custom
      params:
        file: ./examples/custom_modules/my_generator.py
"""

from langchain_core.messages import BaseMessage


def build(params, ctx):
    def generate(messages: list[BaseMessage]) -> str:
        # TODO(替換點):把 prompt 送進你的推論服務,回傳答案文字。
        prompt = messages[-1].text  # 最後一則就是完整的使用者 prompt
        return f"(自訂 generator)已收到 {len(prompt)} 字的 prompt,請在此接上你的 LLM。"

    return generate
