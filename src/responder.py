from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

class ContextualLLMResponder:
    def __init__(self, model_name: str = "gpt-5-mini", temperature: float = 0.5):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = PromptTemplate(
            input_variables=["contexto", "pergunta"],
            template=PROMPT_TEMPLATE
        )
       

    def generate_answer(self, context: str, question: str) -> str:
        try:
            sequence = self.prompt | self.llm
            result = sequence.invoke({"contexto": context, "pergunta": question})
            return result.content.strip()
        except Exception as e:
            logger.exception(f"Erro ao gerar resposta: {e}")
