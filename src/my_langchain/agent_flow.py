import os

from langchain_huggingface import (
  HuggingFacePipeline
)
from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain import hub
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults

from src.config import load_config

config = load_config()

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

def tavily_tool():
  tavily_token = config['TAVILY_API_KEY']
  os.environ["TAVILY_API_KEY"] = tavily_token

  search = TavilySearchResults(max_results=2)
  return search 

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

def ReAct():
  hf_token = config['LANG_CHAIN_API_KEY']
  os.environ["LANGCHAIN_API_KEY"] = hf_token

  prompt = hub.pull("hwchase17/react")
  return prompt

# Recupera a llm via cloud/api
def get_llama_llm():
  llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1)
  return llm

def create_agent(llm, tools, prompt):
  agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
  agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handling_parsing_errors=True
  )
  return agent_executor

def load_agent():
  llm = get_llama_llm()
  tools = [tavily_tool(), date_tool()]
  prompt = ReAct()

  agent = create_agent(llm, tools, prompt)
  response = agent.invoke({"input": "Existem quantos filmes de harry potter?"})
  print(response)