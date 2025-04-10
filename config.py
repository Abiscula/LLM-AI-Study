import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

def load_config():
    hf_token = os.getenv('HF_TOKEN')  # Carrega o token da Hugging Face da variável de ambiente
    if not hf_token:
        raise ValueError("HF_TOKEN não encontrado nas variáveis de ambiente.")
    return {
        'HF_TOKEN': hf_token
    }
