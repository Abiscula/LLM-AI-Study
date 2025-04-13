from langchain_huggingface import (
  HuggingFacePipeline
)
from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from src.config import create_pipeline


# Busca dados direto da wikipedia
def wikipedia_tool():
  wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
    top_k_results=1, doc_content_chars_max=3000
  ))
  tool = Tool(
    name='wikipedia', 
    description="You must never search for multiple concepts at a single step, " \
    "you need to search one at a time. When asked to compare two concepts for example, " \
    "search for each one individually.",
    func=wikipedia.run
  )
  return tool

# Função para obter o dia atual
def current_day(*args, **kwargs):
  from datetime import date

  day = date.today()
  day = day.strftime('%d/%m/%Y')
  return day

# Tool customizada para trabalhar com o dia atual
def date_tool():
  date = Tool(name="Day", func=current_day,
              description="Use when you need to know the current date")
  return date

def load_agent():
  pipe = create_pipeline()
  llm = HuggingFacePipeline(pipeline = pipe)
  tools = [wikipedia_tool(), date_tool()]
