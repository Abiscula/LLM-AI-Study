from langchain_huggingface import (
  HuggingFacePipeline
)

from langchain_core.prompts import PromptTemplate

from src.config import create_pipeline

def user_template():
  template = """<|system|>
  {}<|end|>
  <|user|>
  "{}"<|end|>
  <|assistant|>"""

  return template

def chain_flow(llm, template):
  user_prompt = "Explique brevemente o conceito de {topic} de forma clara e objetiva." \
  "Escreva no máximo {length}"
  sys_prompt = "Você é um assistente e está respondendo perguntas."

  prompt = PromptTemplate.from_template(template.format(sys_prompt, user_prompt))
  chain = prompt | llm

  response = chain.invoke({"topic": "IA", "length": "1 frase"})
  print(response)

def load_lang_chain():
  pipe = create_pipeline()
  llm = HuggingFacePipeline(pipeline = pipe)
  template = user_template()

  chain_flow(llm, template)


  