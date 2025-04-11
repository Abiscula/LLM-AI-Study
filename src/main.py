from simple_model import load_simple_model
from my_langchain.default_flow import load_lang_chain
import torch

def main():
  # Verifica se tem GPU disponível - se não, usa CPU
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  if device == 'cpu':
      pass

  # load_simple_model()
  load_lang_chain()

if __name__ == "__main__":
  main()
