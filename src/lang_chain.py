from langchain_huggingface import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import create_pipeline


def user_prompt(llm):
  input = "Quem foi a primeira pessoa no espaço"
  output = llm.invoke(input)
  return output

def load_lang_chain():
  pipe = create_pipeline()

  llm = HuggingFacePipeline(pipeline = pipe)
  print(user_prompt(llm))