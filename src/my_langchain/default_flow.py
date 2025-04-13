from langchain_huggingface import (
  HuggingFacePipeline
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from datetime import date

from src.config import create_pipeline

def user_template():
  template = """<|system|>
  {}<|end|>
  <|user|>
  "{}"<|end|>
  <|assistant|>"""

  return template

def RAG_template():
  template = """<|system|>
  {system}<|end|>
  <|user|>
  "
  Question: {question}
  Context: {context}
  "<|end|>
  <|assistant|>"""
  return template

# Fluxo de criação de uma chain
def chain_flow(llm):
  template = user_template()
  user_prompt = "Explique brevemente o conceito de {topic} de forma clara e objetiva." \
  "Escreva no máximo {length}"
  sys_prompt = "Você é um assistente e está respondendo perguntas."

  prompt = PromptTemplate.from_template(template.format(sys_prompt, user_prompt))
  chain = prompt | llm | StrOutputParser()

  return chain

def chain_with_RAG(llm):
  template = RAG_template()
  today = date.today()

  question = "Que dia é hoje? Retorne a data em formato dd/mm/yyyy"
  context = f"Você sabe que hoje é dia '{today}'"
  system = "Você é um assistente e está respondendo perguntas."
  
  prompt = PromptTemplate.from_template(template)
  chain = prompt | llm | StrOutputParser()

  res = chain.invoke({
    "system": system,
    "question": question,
    "context": context
  })

  print(res)

# Runnable adiciona funções em tempo de execução
def runnable_count(parsed_chain):
  count = RunnableLambda(lambda x: f"Palavras: {len(x.split())}\n{x}")
  chain_with_function = parsed_chain | count
  return chain_with_function

# Retorna o feedback visual do texto sendo gerado em tempo real
def streaming(chain):
  for chunk in chain.stream({"topic": "IA", "length": "1 parágrafo"}):
    print(chunk, end='', flush=True)

def load_lang_chain():
  pipe = create_pipeline()
  llm = HuggingFacePipeline(pipeline = pipe)

  # chain = chain_flow(llm)
  # streaming(chain)

  chain_with_RAG(llm)

  