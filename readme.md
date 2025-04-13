## 📚 [ESTUDO] Geração de Texto com Modelos de Linguagem Quantizados

Este é um projetinho de estudo com foco em aprendizado prático sobre **Modelos de Linguagem (LLMs)**, **uso da GPU**, **quantização 4-bit**, integração com **APIs em nuvem**, e boas práticas na construção de agentes inteligentes utilizando a biblioteca `langchain`.

---

### 🎯 Objetivo

Explorar, estudar e testar o uso de modelos de linguagem otimizados para execução local com **quantização em 4 bits**, reduzindo o consumo de memória e permitindo o uso eficiente em máquinas com GPUs mais modestas.  
O projeto evoluiu também para investigar a integração com **modelos na nuvem via Hugging Face Endpoints**, possibilitando a execução de modelos maiores sem sobrecarregar o hardware local.  
Além disso, foram implementados agentes interativos com base na arquitetura **ReAct (Reasoning + Acting)**, capazes de utilizar ferramentas externas para raciocínio e tomada de decisão baseada em buscas em tempo real.

---

### 🛠️ Tecnologias Utilizadas

- **🧠 Transformers (Hugging Face)**  
  Utilizado para carregar e rodar modelos de linguagem localmente, com suporte à quantização.

- **📦 Modelo Local: [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)**  
  Modelo leve, recente e eficiente, otimizado para tarefas de instrução.

- **🧮 Quantização com BitsAndBytesConfig**  
  Reduz o uso de memória via `load_in_4bit=True`, viabilizando o uso de LLMs em GPUs com recursos limitados.

- **🚀 CUDA (GPU)**  
  Aceleração da execução dos modelos quantizados localmente.

- **🧩 LangChain**  
  Biblioteca principal usada para estruturar a lógica de agentes com ferramentas, prompts e LLMs.

- **📄 LangChain Hub (Prompt ReAct)**  
  Utilização do prompt `hwchase17/react` para estruturar agentes baseados no padrão ReAct (Reason + Act).

- **🌐 HuggingFaceEndpoint**  
  Permite rodar modelos como o Meta-Llama-3-8B-Instruct em nuvem, sem depender da capacidade local de processamento.

- **🤖 AgentExecutor e create_react_agent**  
  Orquestram a execução do agente com ferramentas conectadas, definindo como a LLM interage com o ambiente.

- **🛠️ Tools (Ferramentas Integradas)**:

  - **📚 WikipediaQueryRun**  
    Busca e recuperação de conteúdo da Wikipedia.
  - **🔍 TavilySearchResults**  
    Ferramenta integrada à API da Tavily para buscas online em tempo real.
  - **📅 Tool personalizada: Data atual**  
    Função simples que retorna a data atual como uma ferramenta acessível ao agente.

- **🔐 .env + os.environ**  
  Utilização de variáveis de ambiente para configurar tokens da Hugging Face e Tavily de forma segura e reutilizável no código.

---
