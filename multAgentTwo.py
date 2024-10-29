from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv
import os

# Load the environment variables
load_dotenv()

config_list = [
    {"model": os.getenv('MODEL'),"api_key": os.getenv('API_KEY'),"base_url": os.getenv('BASE_URL'),"price": [0, 0]}
]

llm_config={"config_list": config_list,"cache_seed": 42,}

user_proxy = UserProxyAgent(
    name="User_proxy",
    system_message="Um administrador humano.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },
    human_input_mode="TERMINATE",
)

coder = AssistantAgent(
    name="Coder",
    llm_config=llm_config,
)

pm = AssistantAgent(
    name="Product_manager",
    system_message="Criativo em ideias de produtos de software.",
    llm_config=llm_config,
)

groupchat = GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager, message="encontre um curso java/ kotlin, curso deve ter no maximo 60 horas e com otima avaliação dos que o fizeram"
)