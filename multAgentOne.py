from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import DockerCommandLineCodeExecutor
from dotenv import load_dotenv
import os

# Load the environment variables
load_dotenv()

config_list = [
    {"model": os.getenv('MODEL'),"api_key": os.getenv('API_KEY'),"base_url": os.getenv('BASE_URL'),"price": [0, 0]}
]

# create an AssistantAgent instance named "assistant" with the LLM configuration.
assistant = AssistantAgent(name="assistant", llm_config={"config_list": config_list})

# create a UserProxyAgent instance named "user_proxy" with code execution on docker.
code_executor = DockerCommandLineCodeExecutor()
user_proxy = UserProxyAgent(name="user_proxy", code_execution_config={"executor": code_executor})

# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""Oi, sou programador, me sugira alguma fontes alternativas de renda?""",
)

""" 
Ilustração
1. O assistente recebe uma mensagem do user_proxy, que contém a descrição da tarefa.
2. O assistente então tenta escrever código Python para resolver a tarefa e envia a resposta para o user_proxy.
3. Uma vez que o user_proxy recebe uma resposta do assistente, ele tenta responder solicitando entrada humana
   ou preparando uma resposta gerada automaticamente. Se nenhuma entrada humana for fornecida, o user_proxy 
   executa o código e usa o resultado como a resposta automática.
4. O assistente então gera uma resposta adicional para o user_proxy. O user_proxy pode então decidir se encerra
   a conversa. Se não, as etapas 3 e 4 são repetidas.
"""