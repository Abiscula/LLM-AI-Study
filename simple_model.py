from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from config import load_config
from utils.quantization import quantization

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

  config = load_config()
  hf_token = config['HF_TOKEN']

  if not hf_token:
    pass
      
  quantization_config = quantization()

  model_id = 'microsoft/Phi-3-mini-4k-instruct'

  model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id, 
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager",
    token=hf_token,
    quantization_config=quantization_config
  )

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  
  # Criando o pipeline de geração de texto
  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
  generations_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.1,  # Varia de 0.1 até 0.9
    "do_sample": True
  }

  template = user_prompt()

  output = pipe(template, **generations_args)
  print(output[0]['generated_text'])