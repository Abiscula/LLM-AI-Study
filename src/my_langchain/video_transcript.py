import os
import io

from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import load_config

config = load_config()

# Recupera a llm via cloud/api
def get_llama_llm():
  hf_token = config['HF_TOKEN']
  os.environ["HF_TOKEN"] = hf_token

  llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct", 
    temperature=0.1,
    task="text-generation"
  )
  return llm

def video_transcription():
  video_url = 'https://www.youtube.com/watch?v=eXdVDhOGqoE'
  video_loader = YoutubeLoader.from_youtube_url(
      video_url,
      language=["pt", "pt-BR", "en"],
      translation='pt'
    )
  infos = video_loader.load()

  return infos

def write_video_transcription(infos):
  with io.open("transcricao.txt", "w", encoding="utf-8") as f:
    for doc in infos:
      f.write(doc.page_content + "\n\n")

def create_prompt():
  system_prompt = "Você é um assistente virtual prestativo e deve responder a uma" \
  "consulta com base na transcrição de um vídeo que será fornecida abaixo"

  inputs = "Consulta: {consulta} \n Transcrição: {transcricao}"
  user_prompt = "{}".format(inputs)

  prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", user_prompt)]
  )
  return prompt_template

def create_chain(prompt, llm):
  chain = prompt | llm | StrOutputParser()
  return chain

def load_video_transcript():
  infos = video_transcription()
  # write_video_transcription(infos)

  llm = get_llama_llm()
  prompt = create_prompt()
  chain = create_chain(prompt, llm)
  response = chain.invoke({
    "transcricao": infos[0].page_content,
    "consulta": "liste os temas desse video"
  })
  print(response)