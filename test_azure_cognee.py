import asyncio
import os
import cognee
from dotenv import load_dotenv

load_dotenv(override=True)

# Validar se as variáveis de ambiente foram configuradas
llm_api_key = os.getenv("LLM_API_KEY")
embedding_api_key = os.getenv("EMBEDDING_API_KEY")

if not llm_api_key:
    raise ValueError("LLM_API_KEY não configurada no arquivo .env")
if not embedding_api_key:
    raise ValueError("EMBEDDING_API_KEY não configurada no arquivo .env")

async def main():

    print("→ Limpando bancos do Cognee...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    texto = """
    A Internet é um sistema global de redes de computadores interligadas...
    """

    print("Adicionando conteúdo...")
    await cognee.add(texto)

    print("Processando com OpenAI / Azure OpenAI...")
    await cognee.cognify()

    print("Realizando busca semântica...")
    results = await cognee.search(
        query_text="Quem desenvolveu o protocolo TCP/IP e em qual década?"
    )

    print("\n=== RESULTADOS ===\n")
    for r in results:
        print(r)

if __name__ == "__main__":
    asyncio.run(main())
