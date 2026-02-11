# Cognee + Azure OpenAI

Projeto de integração entre a biblioteca **Cognee** e os serviços de IA do **Azure OpenAI**. Este repositório contém exemplos práticos de como utilizar embeddings, modelos de linguagem e processamento de dados com Cognee.

## Descrição

O projeto demonstra a utilização de:
- **Cognee**: Framework para processamento e análise semântica de dados
- **Azure OpenAI**: Serviços de IA da Microsoft (embeddings e modelos de linguagem)
- **Busca Semântica**: Recuperação de informações com base em similaridade

## Funcionalidades

### 1. **Teste de Embeddings** (`test_azure_embedding.py`)
- Geração de embeddings com Azure OpenAI
- Modelo: `embedding-3-small-global`
- Verificação de dimensionalidade e valores dos embeddings

### 2. **Teste de Modelo LLM** (`test_azure_llm.py`)
- Interação com modelo de linguagem Azure OpenAI
- Modelo: `o4-mini-petrobras`
- Exemplo básico de chat completions

### 3. **Teste com Cognee** (`test_azure_cognee.py`)
- Limpeza de bancos de dados do Cognee
- Adição de conteúdo para processamento
- Requirements de um código com adaptadores: git+https://github.com/tiagoriosrocha/cognee.git@adapters-0.5.0
- Utiliza um fork do projeto cognee que usa os adaptadores de LLM e Embeddings do Azure OpenAI
- Cognificação (processamento com OpenAI/Azure OpenAI)
- Busca semântica em dados processados

## Dependências

- `requests` - Requisições HTTP
- `openai` - SDK do OpenAI
- `python-dotenv` - Variáveis de ambiente
- `numpy` - Operações numéricas
- `cognee` - Framework de processamento (branch adapters-0.5.0)
- `playwright` - Automação de navegador
- `transformers` - Modelos de transformers
- `python-certifi-win32` - Certificados SSL (Windows) -> ambiente virtual

## Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd cognee-azure-openai
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
```bash
cp .env.example .env
```

Adicione suas credenciais de Azure OpenAI no arquivo `.env`:

## Uso

Execute os testes individualmente:

### Teste de Embeddings
```bash
python test_azure_embedding.py
```

### Teste de LLM
```bash
python test_azure_llm.py
```

### Teste Completo com Cognee
```bash
python test_azure_cognee.py
```

## Notas Importantes

- O projeto utiliza o branch `adapters-0.5.0` do Cognee, pois a versão 0.5.0 não possui suporte para adapters
- As credenciais de API não devem ser commitadas no repositório
- Utilize um arquivo `.env` para armazenar dados sensíveis
- A busca semântica é realizada através da cognificação dos dados adicionados

## Segurança

**Não commite dados sensíveis** como chaves de API ou tokens de autenticação

Sempre utilize um arquivo `.env` ou um gerenciador de secrets para proteger suas credenciais.

## Documentação

- [Cognee GitHub](https://github.com/tiagoriosrocha/cognee)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)

## Autor

Desenvolvido por Tiago Rios da Rocha

## Licença

Este projeto é fornecido como está. Verifique as licenças das dependências utilizadas.
