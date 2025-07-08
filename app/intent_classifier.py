from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

INTENT_FALLBACK_QUERIES = {
    "diagnosis": "examples of common plant diseases",
    "health_check": "examples of healthy vs unhealthy plant symptoms",
    "prevention": "how to prevent common plant diseases and pests",
    "treatment": "treatment for common plant diseases",
    "generic": "plant disease and pest examples",
}


class IntentClassifier:
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama(
            model="mistral", base_url="http://host.docker.internal:11434"
        )
        self.prompt = PromptTemplate.from_template(
            """
            Classify the following query into one of these categories:
            - diagnosis
            - health_check
            - prevention
            - treatment
            - generic

            Query: {query}
            Category:
            """
        )

    def classify(self, query: str) -> str:
        chain = self.prompt | self.llm | (lambda x: x.content.strip().lower())
        return chain.invoke({"query": query})
