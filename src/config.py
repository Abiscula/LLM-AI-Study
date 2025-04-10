import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from dotenv import load_dotenv
from utils.quantization import quantization

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

def load_config():
  hf_token = os.getenv('HF_TOKEN')  # Carrega o token da Hugging Face da variável de ambiente
  if not hf_token:
    raise ValueError("HF_TOKEN não encontrado nas variáveis de ambiente.")
  return {
    'HF_TOKEN': hf_token
  }

def create_ai_model():
  model_id = 'microsoft/Phi-3-mini-4k-instruct'

  config = load_config()
  hf_token = config['HF_TOKEN']

  if not hf_token:
    pass

  quantization_config = quantization()

  model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path = model_id, 
    quantization_config = quantization_config,
    token = hf_token,
  )

  tokenizer = AutoTokenizer.from_pretrained(model_id)

  return [model, tokenizer]

def create_pipeline():
  [model, tokenizer] = create_ai_model()

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