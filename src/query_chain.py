"""Create a ChatVectorDBChain"""
from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.base import VectorStore
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from src.chains import EpisodeContextChain
from src.prompts import (
        CONDENSE_QUESTION_PROMPT,
        QA_PROMPT
    )


def get_chain(
        vectorstore: VectorStore, 
        question_handler, 
        stream_handler, 
        tracing: bool = False,
    ) -> EpisodeContextChain:
    """Create a ChatVectorDBChain"""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    condense_manager = AsyncCallbackManager([question_handler])
    resp_manager = AsyncCallbackManager([stream_handler])
    
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        condense_manager.add_handler(tracer)
        resp_manager.add_handler(tracer)
        
    llm_condense = OpenAI(
        temperature=1.0,
        verbose=True,
        callback_manager=condense_manager,
    )
    llm_resp = OpenAI(
        temperature=1.0,
        verbose=True,
        callback_manager=resp_manager,
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
    
    
    
    