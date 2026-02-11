import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Carrega variáveis de ambiente
load_dotenv()

# Recupera configurações de ambiente
embedding_api_key = os.getenv("EMBEDDING_API_KEY")
embedding_endpoint = os.getenv("EMBEDDING_ENDPOINT")
embedding_api_version = os.getenv("EMBEDDING_API_VERSION")
embedding_model = os.getenv("EMBEDDING_MODEL")

if not embedding_api_key:
    raise ValueError("EMBEDDING_API_KEY não configurada no arquivo .env")

client = AzureOpenAI(
    api_key=embedding_api_key,
    base_url=embedding_endpoint,
    api_version=embedding_api_version,
)

# Texto de teste
text = "Teste de embedding com Azure OpenAI"

# Chamada de embedding
response = client.embeddings.create(
    model=embedding_model,
    input=text,
)

embedding = response.data[0].embedding

print("Embedding gerado com sucesso!")
print("Dimensão do embedding:", len(embedding))
print("Primeiros 10 valores:", embedding[:10])
