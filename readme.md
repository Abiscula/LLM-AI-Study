# 📚 [ESTUDO] Geração de Texto com Modelos de Linguagem Quantizados

Este é um projetinho de estudo com foco em aprendizado prático sobre **Modelos de Linguagem (LLMs)**, **uso da GPU**, **quantização 4-bit**, e boas práticas na construção de pipelines com a biblioteca `transformers` da Hugging Face.

---

## Objetivo

Explorar, estudar e testar o uso de modelos de linguagem otimizados para uso local com **quantização em 4 bits**, reduzindo o uso de memória e viabilizando a execução em máquinas com GPUs mais modestas.

---

## Tecnologias Usadas

- **Transformers** da Hugging Face
- **Modelo:** [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) – um modelo poderoso, leve e recente
- **Quantização:** `BitsAndBytesConfig` com `load_in_4bit=True`
- **CUDA** para uso de GPU
- `.env` para configuração segura do Hugging Face token

---
