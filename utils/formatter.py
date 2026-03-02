# utils/formatter.py

from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import logging

logger = logging.getLogger(__name__)

def format_text_for_knowledge_base(raw_text: str) -> str:
    """
    Formats raw text into a structured markdown format for the knowledge base using an LLM.
    """
    try:
        llm = Ollama(
            model="mistral:7b-instruct-q4_K_M",
            temperature=1,
            top_p=0.7,
            num_ctx=4096,
            top_k=20,
            repeat_penalty=1.1
        )

        formatting_prompt_template = """
        You are an expert data preprocessor. Your task is to reformat the following raw text into a well-structured markdown document that will serve as a knowledge base for a technical support LLM.

        **Formatting Instructions:**

        1. Use Markdown: `##` for headings, bullets for points, etc.
        2. Extract questions and answers as:  
           **Q:** ...  
           **A:** ...
        3. Don't add new content. Just format what's given.
        4. Keep clear spacing between sections.

        **Raw Text to Format:**
        ---
        {text_to_format}
        ---

        **Formatted Knowledge Base:**
        """

        prompt = PromptTemplate(
            input_variables=["text_to_format"],
            template=formatting_prompt_template
        )

        formatter_chain = prompt | llm
        logger.info("Formatting knowledge base...")
        return formatter_chain.invoke({"text_to_format": raw_text})

    except Exception as e:
        logger.error(f"Formatting failed: {e}")
        return raw_text  # Fallback to raw text if formatting fails
