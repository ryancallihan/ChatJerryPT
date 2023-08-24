from langchain.llms import OpenAIChat
from langchain.chains.llm import LLMChain
from langchain.vectorstores.base import VectorStore
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from src.prompts import (
        CONDENSE_QUESTION_PROMPT,
        QA_PROMPT
    )


def get_chain(
        vectorstore: VectorStore, 
        question_handler, 
        stream_handler, 
        openai_api_key,
        tracing: bool = False,
    ) -> ConversationalRetrievalChain:
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)
        
    llm_condense = OpenAIChat(
        model="gpt-3.5-turbo",
        temperature=0.0,
        verbose=True,
        callbacks=question_manager,
        openai_api_key=openai_api_key,
    )
    llm_resp = OpenAIChat(
        model="gpt-3.5-turbo",
        temperature=0.7,
        verbose=True,
        streaming=True,
        callbacks=stream_manager,
        openai_api_key=openai_api_key,
    )
    
    condense_input_chain = LLMChain(
        llm=llm_condense,
        prompt=CONDENSE_QUESTION_PROMPT,
        verbose=True,
        callback_manager=manager,
    )
    
    
    resp_chain = load_qa_chain(
        llm=llm_resp,
        chain_type="stuff",
        prompt=QA_PROMPT,
        verbose=True,
        callback_manager=manager,
    )
    
    resp = ConversationalRetrievalChain(
        verbose=True,
        retriever=vectorstore.as_retriever(),
        question_generator=condense_input_chain,
        combine_docs_chain=resp_chain,
        callback_manager=manager,
    )
    
    return resp
    
    
    
    