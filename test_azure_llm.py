import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Recupera configurações de ambiente
llm_api_key = os.getenv("LLM_API_KEY")
llm_endpoint = os.getenv("LLM_ENDPOINT")
llm_api_version = os.getenv("LLM_API_VERSION")
llm_model = os.getenv("LLM_MODEL")

if not llm_api_key:
    raise ValueError("LLM_API_KEY não configurada no arquivo .env")

client = AzureOpenAI(
    api_key=llm_api_key,
    base_url=llm_endpoint,
    api_version=llm_api_version,
)

resp = client.chat.completions.create(
    model=llm_model,
    messages=[
        {"role": "user", "content": "Responda OK"}
    ],
)

print(resp.choices[0].message.content)
