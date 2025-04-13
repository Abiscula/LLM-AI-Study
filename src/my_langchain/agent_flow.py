from langchain_huggingface import (
  HuggingFacePipeline
)
from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from src.config import create_pipeline


# Busca dados direto da wikipedia
def load_wikipedia_tool():
  wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
    top_k_results=1, doc_content_chars_max=3000
  ))
  wikipedia_tool = Tool(
    name='wikipedia', 
    description="You must never search for multiple concepts at a single step, " \
    "you need to search one at a time. When asked to compare two concepts for example, " \
    "search for each one individually.",
    func=wikipedia.run
  )


def load_agent():
  pipe = create_pipeline()
  llm = HuggingFacePipeline(pipeline = pipe)

  load_wikipedia_tool()