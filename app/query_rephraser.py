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

        # Prompt when caption is available
        self.prompt_with_caption = PromptTemplate.from_template(
            """You are a plant pathology assistant helping users search for plant diseases using both images and natural language.

            Your task is to analyze the image description and the user's question, and then rephrase the question to better match a plant disease search query.

            Always consider what disease the plant might have based on the image description. If the user's query is vague (e.g., "Is this healthy?"), infer the intended diagnosis question based on visible symptoms in the image.

            Respond with only the rephrased search query, without any explanation.

            Image Description: {caption}
            User Query: {input}

            Rephrased Disease Search Query:"""
        )

        # Prompt for text-only queries
        self.prompt_text_only = PromptTemplate.from_template(
            """You are a plant pathology assistant helping users search for plant diseases.

            Rephrase the user's query into a direct plant disease search query.

            If the query is vague (e.g., "Is this healthy?"), turn it into something more specific and useful for diagnosing disease symptoms.

            Respond with only the rephrased query.

            User Query: {input}

            Rephrased Disease Search Query:"""
        )

        # Chains
        # Step 1: Inject input into the prompt
        # Step 2: Pass the prompt to the LLM
        # Step 3: Extract the `content` field from the LLM response
        self.chain_with_caption = (
            self.prompt_with_caption
            | self.llm
            | RunnableLambda(lambda x: x.content)
        )
        self.chain_text_only = (
            self.prompt_text_only
            | self.llm
            | RunnableLambda(lambda x: x.content)
        )

    def rephrase(self, user_input: str, image_caption: str = None) -> str:
        """
        Rephrase a user-friendly question into a clean search query
        using the LLM chain.

        Args:
            user_input (str): Original question from the user
            (e.g., "Show me example of gumosis disease")
            caption (str): Caption of image query from user provided by BLIP

        Returns:
            str: Rephrased search-ready query
            (e.g., "gumosis disease symptoms in plants")
        """
        if image_caption:
            return self.chain_with_caption.invoke(
                {"input": user_input, "caption": image_caption}
            )
        else:
            return self.chain_text_only.invoke({"input": user_input})
