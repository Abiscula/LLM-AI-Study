import sys
import os

# Adiciona a pasta 'src' ao sys.path
src_path = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_path)

# Agora pode importar main normalmente
from main import main

if __name__ == "__main__":
    main()
