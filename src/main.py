from simple_model import load_simple_model
from my_langchain.default_flow import load_lang_chain
from my_langchain.chat_model_flow import load_chat_model
from my_langchain.agent_flow import load_agent
from my_langchain.video_transcript import load_video_transcript

import torch

def main():
  # Verifica se tem GPU disponível - se não, usa CPU
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  if device == 'cpu':
      pass

  # load_simple_model()
  # load_lang_chain()
  # load_chat_model()
  # load_agent()
  load_video_transcript()

if __name__ == "__main__":
  main()
