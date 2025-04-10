from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from config import load_config


def main():
    # Verifica se tem GPU disponível - se não, usa CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Garantindo a reprodutibilidade entre diversas execuções
    torch.random.manual_seed(40)

    config = load_config()
    hf_token = config['HF_TOKEN']

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


if __name__ == "__main__":
    main()
