import json
import os
import httpx
from typing import List
from dotenv import load_dotenv
from openai import AzureOpenAI, AsyncAzureOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.evaluate import AsyncConfig
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.models import AzureOpenAIModel
from deepeval.models import LiteLLMModel
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    BaseMetric
)
from deepeval.models.base_model import DeepEvalBaseLLM



class AzureOpenAICustom(DeepEvalBaseLLM):
    def __init__(self):
        self.sync_client = AzureOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION"),
            base_url=os.getenv("LLM_ENDPOINT"),
            http_client=httpx.Client(verify=os.getenv("SSL_CERT_FILE"))
        )

        self.async_client = AsyncAzureOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION"),
            base_url=os.getenv("LLM_ENDPOINT"),
            http_client=httpx.AsyncClient(verify=os.getenv("SSL_CERT_FILE"))
        )

        self.deployment = os.getenv("LLM_MODEL")

    def load_model(self):
        return self.sync_client

    def generate(self, prompt: str) -> str:
        response = self.sync_client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.async_client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content

    def get_model_name(self):
        return f"Azure-{self.deployment}"
    



class DeepEvalEvaluator:
    def __init__(self, model_name: str = "phi4:latest", base_url: str = "http://localhost:11434", temperature: float = 0):
        
        #Carrega as variáveis de ambiente do .env
        load_dotenv()
        self.http_client = httpx.Client(verify=os.getenv("SSL_CERT_FILE"))

        #Instancia o modelo Ollama
        # self.ollama_model = OllamaModel(
        #     model=self.model_name,
        #     base_url=self.base_url,
        #     temperature=self.temperature
        # )

        #Instancia o modelo LiteLLM pra conseguir rodar no Ollama K8s - cluster da UFRGS
        #self.ollama_model = LiteLLMModel(
        #    model="openai/gpt-oss:20b",  # Provider must be specified
        #    api_key="",  # optional, can be set via environment variable
        #    base_url="https://ollama.k8s.inf.ufrgs.br/ollama/v1",  # optional, for custom endpoints
        #    temperature=0
        #)

        #Instancia o modelo Azure OpenAI para usar os modelos da Petrobras -> não funcionou
        #self.model = AzureOpenAIModel(
        #    model="gpt-5-petrobras",
        #    deployment_name="gpt-5-petrobras",
        #    api_key=os.getenv("LLM_API_KEY"),
        #    api_version="2025-01-01-preview",
        #    base_url="https://apit.petrobras.com.br/ia/openai/v1/openai-azure/openai",
        #    temperature=0,
        #    http_client=self.http_client
        #)

        # Instancio um objeto de um modelo custom que usa a classe AzureOpenAIModel para avaliação das métricas -> funcionou
        azure_openai = AzureOpenAICustom()

        #Define as métricas
        self.metrics: List[BaseMetric] = [
            #ContextualRelevancyMetric(threshold=0.7, model=self.ollama_model),
            #ContextualRecallMetric(threshold=0.7, model=self.ollama_model),
            #ContextualPrecisionMetric(threshold=0.7, model=self.ollama_model),
            AnswerRelevancyMetric(threshold=0.7, model=azure_openai),
            FaithfulnessMetric(threshold=0.7, model=azure_openai),
        ]


    def evaluate_test_case(self, test_case: LLMTestCase) -> str:
        #Executa a avaliação
        test_results = evaluate([test_case], 
                                metrics=self.metrics, 
                                async_config=AsyncConfig(run_async=False))
        
        #Pega o primeiro (e único) resultado
        test_result = test_results.test_results[0]
        #cria o json
        evaluation_results_json = {
            #"test_name": test_result.name,
            "success": test_result.success,
            "input": test_result.input,
            "actual_output": test_result.actual_output,
            "expected_output": test_result.expected_output,
            #"retrieval_context": test_result.retrieval_context,
            "metrics": []
        }
        #Navega pelas métricas e salva os dados no json
        for metric in test_result.metrics_data:
            evaluation_results_json["metrics"].append({
                "metric_name": metric.name,
                "score": metric.score,
                "passed": metric.success,
                "reasoning": metric.reason,
                #"threshold": metric.threshold,
                #"strict_mode": metric.strict_mode,
                #"evaluation_model": metric.evaluation_model,
                #"error": str(metric.error) if metric.error else None,
                #"evaluation_cost": metric.evaluation_cost,
                #"verbose_logs": metric.verbose_logs
            })

        #Converte o resultado para string JSON compacta
        #evaluation_results_json_str = json.dumps(evaluation_results_json, separators=(',', ':'))
        
        return evaluation_results_json




# Preparando os dados de teste (exemplo)
question = "Qual é a capital da França?"
final_answer = "Paris é a capital da França e a maior cidade do país."
expected_output = "Paris"
retrieval_context = ["A capital da França é Paris.", "Paris é conhecida por sua arte e culinária."]

# Cria o caso de teste
test_case_example = LLMTestCase(
    input=question,
    actual_output=final_answer,
    expected_output=expected_output,
    retrieval_context=retrieval_context,
)

# Instancia a classe
evaluator = DeepEvalEvaluator()

# Avalia e obtém o resultado em formato de dicionário
evaluation_results = evaluator.evaluate_test_case(test_case_example)
print(evaluation_results)