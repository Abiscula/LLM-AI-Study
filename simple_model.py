import torch
from config import create_pipeline

# Criação do template/prompt do usuário
def user_prompt():
  prompt = "Quem foi a primeira pessoa no espaço?"
  sys_prompt = "Você é um professor de história. Responda a pergunta em português."

  template = """<|system|>
  {}<|end|>
  <|user|>
  "{}"<|end|>
  <|assistant|>""".format(sys_prompt, prompt)

  return template

def load_simple_model():
  # Garantindo a reprodutibilidade entre diversas execuções
  torch.random.manual_seed(40)
    
  pipe = create_pipeline()
  template = user_prompt()

  output = pipe(template)
  print(output[0]['generated_text'])