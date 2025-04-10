from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from config import load_config

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

def load_model():
  # Verifica se tem GPU disponível - se não, usa CPU
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # Garantindo a reprodutibilidade entre diversas execuções
  torch.random.manual_seed(40)

  config = load_config()
  hf_token = config['HF_TOKEN']

  if device == 'cpu' or not hf_token:
    pass
      
  quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
  )

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