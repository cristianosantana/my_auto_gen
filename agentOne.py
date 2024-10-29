from autogen import ConversableAgent, UserProxyAgent
import autogen
import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

local_llm_config = {
    "config_list": [
        {
            "model": os.getenv('MODEL'),
            "api_key": os.getenv('API_KEY'),
            "base_url": os.getenv('BASE_URL'),
            "price": os.getenv('PRICE'),
        }
    ],
    "cache_seed": None,
}

# Create the agent that uses the LLM.
assistant = ConversableAgent("agent", llm_config=local_llm_config)

# Create the agent that represents the user in the conversation.
user_proxy = UserProxyAgent("user", code_execution_config=False)

# Let the assistant start the conversation.  It will end when the user types exit.
res = assistant.initiate_chat(user_proxy, message="How can I help you today?")

print(assistant)