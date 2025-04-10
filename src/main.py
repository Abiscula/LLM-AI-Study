from simple_model import load_simple_model
from lang_chain import load_lang_chain
import torch

if __name__ == "__main__":

  # Verifica se tem GPU disponível - se não, usa CPU
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  
  if device == 'cpu':
    pass

  # load_simple_model()
  load_lang_chain()
