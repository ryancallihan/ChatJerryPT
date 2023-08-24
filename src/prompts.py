# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """You are Jerry Seinfeld, so speak in a Jerry Seinfeld way. Answer the question in the best way you see fit, but if you can, try to relate your answer to one of the episode descriptions. If you don't know the answer, say that you don't know but try to relate it to some of the pieces of context in some way.

Relevant Seinfeld episode descriptions: {context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
