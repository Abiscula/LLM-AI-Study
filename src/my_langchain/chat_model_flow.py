from langchain_huggingface import (
  HuggingFacePipeline,
  ChatHuggingFace
)

from langchain_core.messages import (
  SystemMessage,
  HumanMessage
)

from transformers import pipeline

from src.config import create_ai_model


def user_prompt(chat_model):
  msgs = [
    SystemMessage(content = "Você é um assistente e esta respondendo perguntas gerais."),
    HumanMessage(content = "Expliqiue para mim brevemente o conceito de IA.")
  ]

  chat_model._to_chat_prompt(msgs)
  output = chat_model.invoke(msgs)
  return output.content

def chat_model_config(llm, tokenizer):
  model_template = tokenizer.chat_template
  chat_model = ChatHuggingFace(llm = llm, model_template = model_template)
  return chat_model

def set_pipeline(model, tokenizer):
  pipe = pipeline(
    model = model,
    tokenizer = tokenizer,
    task = "text-generation",
    temperature = 0.1,
    max_new_tokens = 500,
    do_sample = True,
    repetition_penalty = 1.1,
    return_full_text = False
  )
  return pipe

def load_chat_model():
  [model, tokenizer] = create_ai_model()

  pipe = set_pipeline(model, tokenizer)
  llm = HuggingFacePipeline(pipeline = pipe)

  chat_model = chat_model_config(llm, tokenizer)

  print(user_prompt(chat_model))