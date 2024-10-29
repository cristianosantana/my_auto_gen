# Instruções de instalação

## Fora do enviromment

### Instalar Ollama

* curl -fsSL <https://ollama.com/install.sh> | sh

### Download LLM

* ollama pull llama3.2:3b

### Listar LLM`s

* ollama list

### Servir o LLM

* litellm --model ollama/llama3.2:3b

## Criar environment

* python3 -m venv .venv
* source .venv/bin/activate
* deactivate

## Dentro do enviromment

### Instalar AutoGen

* pip3 install autogen-agentchat==0.2.36


