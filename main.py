from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from config import load_config


def main():
    # Verifica se tem GPU disponível - se não, usa CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Garantindo a reprodutibilidade entre diversas execuções
    torch.random.manual_seed(40)

    config = load_config()
    hf_token = config['HF_TOKEN']

    model_id = 'microsoft/Phi-3-mini-4k-instruct'

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id, 
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        token=hf_token
    )


if __name__ == "__main__":
    main()
