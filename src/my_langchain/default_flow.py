from langchain_huggingface import (
  HuggingFacePipeline
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import create_pipeline

def user_template():
  template = """<|system|>
  {}<|end|>
  <|user|>
  "{}"<|end|>
  <|assistant|>"""

  return template

# Fluxo de criação de uma chain
def chain_flow(llm, template):
  user_prompt = "Explique brevemente o conceito de {topic} de forma clara e objetiva." \
  "Escreva no máximo {length}"
  sys_prompt = "Você é um assistente e está respondendo perguntas."

  prompt = PromptTemplate.from_template(template.format(sys_prompt, user_prompt))
  chain = prompt | llm

  return chain

# converte a resposta para string
def str_output_parser(chain):
  chain_str = chain | StrOutputParser()
  return chain_str

def load_lang_chain():
  pipe = create_pipeline()
  llm = HuggingFacePipeline(pipeline = pipe)
  template = user_template()

  chain = chain_flow(llm, template)

  parsed_chain = str_output_parser(chain)

  response = parsed_chain.invoke({"topic": "IA", "length": "1 frase"})
  print(response)


  