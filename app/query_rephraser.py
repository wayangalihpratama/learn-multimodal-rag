from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama


class QueryRephraser:
    """
    This class uses a local Ollama LLM (e.g., Mistral, LLaMA) to convert
    user-friendly natural language queries into precise search queries
    suitable for a vector search (e.g., ChromaDB).
    """

    def __init__(self, llm=None):
        # Initialize the LLM
        # (defaults to a local Ollama instance using the 'mistral' model)
        self.llm = llm or ChatOllama(
            model="mistral", base_url="http://host.docker.internal:11434"
        )  # You can override with any compatible LangChain LLM

        # Define a prompt that instructs the LLM how to rewrite the query
        self.prompt = PromptTemplate.from_template(
            "Rephrase this user query into a disease search query: {input}"
        )

        # Create a chain:
        # Step 1: Inject input into the prompt
        # Step 2: Pass the prompt to the LLM
        # Step 3: Extract the `content` field from the LLM response
        self.chain = (
            self.prompt | self.llm | RunnableLambda(lambda x: x.content)
        )

    def rephrase(self, user_input: str) -> str:
        """
        Rephrase a user-friendly question into a clean search query
        using the LLM chain.

        Args:
            user_input (str): Original question from the user
            (e.g., "Show me example of gumosis disease")

        Returns:
            str: Rephrased search-ready query
            (e.g., "gumosis disease symptoms in plants")
        """
        return self.chain.invoke({"input": user_input})


# --- Example Usage ---
# rephraser = QueryRephraser()
# query = rephraser.rephrase("Show me an example of gumosis disease")
# print(query)  # -> gumosis disease in plants
