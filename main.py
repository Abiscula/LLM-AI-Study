from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


def main():
    # Verifica se tem GPU disponível - se não, usa CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Garantindo a reprodutibilidade entre diversas execuções
    torch.random.manual_seed(40)



if __name__ == "__main__":
    main()
