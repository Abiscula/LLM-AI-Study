from langchain_huggingface import (
  HuggingFacePipeline
)

from src.config import create_pipeline

def user_prompt(llm):
  prompt = "Quem foi a primeira pessoa no espaço"
  sys_prompt = "Você é um professor de história. Responda a pergunta em português."

  template = """<|system|>
  {}<|end|>
  <|user|>
  "{}"<|end|>
  <|assistant|>""".format(sys_prompt, prompt)

  output = llm.invoke(template)
  return output

def load_lang_chain():
  pipe = create_pipeline()
  llm = HuggingFacePipeline(pipeline = pipe)
  print(user_prompt(llm))