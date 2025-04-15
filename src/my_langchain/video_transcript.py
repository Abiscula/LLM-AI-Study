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

  llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1)
  return llm

def video_transcription():
  video_loader = YoutubeLoader.from_youtube_url(
      'https://www.youtube.com/watch?v=1IyxRkNj9lU',
      language=["pt", "pt-BR", "en"]
    )
  infos = video_loader.load()

  return infos

def write_video_transcription(infos):
  with io.open("transcricao.txt", "w", encoding="utf-8") as f:
    for doc in infos:
      f.write(doc.page_content + "\n\n")

def load_video_transcript():
  llm = get_llama_llm()
  infos = video_transcription()
  write_video_transcription(infos)