import os
import re
from pydantic import HttpUrl
from langchain_gigachat.chat_models import GigaChat
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, List, Optional


class Agent(TypedDict):
    query: str
    id: int
    options: Optional[List[str]]
    llm_answer: Optional[str]
    search_results: Optional[List[dict]]
    final_answer: Optional[dict]
    tool_calls: Optional[List]

def parse(state: Agent) -> Agent:
    query = state["query"]
    options = re.findall(r"\n(\d+)\..+", query)
    state["options"] = options if len(options) > 1 else None
    return state

def generate_answer(state: Agent) -> Agent:
    llm = GigaChat(
        credentials=os.environ["GIGACHAT_CREDENTIALS"],
        verify_ssl_certs=False
    )
    
    search_tool = TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        include_domains=['itmo.ru']
    )
    
    model_with_tools = llm.bind_tools([search_tool])
    
    prompt_template = PromptTemplate.from_template(
        """Ты являешься экспертом по Университету ИТМО.
        Ответь на поставленный вопрос: {query}
        Правила:
        1. Используй инструменты поиска информации.
        2. Ответ должен быть кратким и фактологически точным.
        3. Будь уверенным в своих ответах.
        4. Если есть сомнения, то лучше перепроверь свой ответ"""
    )
    
    messages = [
        SystemMessage(content="Выполни запрос пользователя, используя доступные инструменты."),
        HumanMessage(content=prompt_template.format(query=state["query"]))
    ]
    
    response = model_with_tools.invoke(messages)
    
    # Обработка ответа
    state["llm_answer"] = response.content
    
    return state

def search(state: Agent) -> Agent:
    search_tool = TavilySearchResults(
        include_domains=['itmo.ru'],
        include_answer=True,
        max_results=3,
        include_raw_content=False,
        include_images=False,
    )
    query = state["query"]
    results = search_tool.invoke({"query": query})

    state["search_results"] = [res for res in results[:3]] 
    return state

def decide_answer(state: Agent) -> Agent:
    answer_template = PromptTemplate.from_template(
        """Сформулируй окончательный ответ на основе следующих критериев:
        Исходный вопрос: {query}
        Ответ LLM: {llm_answer}
        Варианты: {options_text}

        Правила:
        1. Если вопроса с вариантом выбора, следует вернуть выбранный вариант (цифрой).
        2. Если нет подходящего ответа, то следует вернуть null.
        3. Ответ может быть только цифрой либо null."""
    )
    
    sources = []


    reasoning = []
    
    if state["llm_answer"]:
        reasoning.append(f"GigaChat: {state['llm_answer']}")
    
    if state["search_results"]:
        sources = [HttpUrl(res["url"]) for res in state["search_results"]]

    answer = None
    if state["options"]:
        options_text = "\n".join(state["options"])
        llm = GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            verify_ssl_certs=False,
            temperature=0
        )
        answer = llm.invoke([
            HumanMessage(content=answer_template.format(
                query=state["query"],
                llm_answer=state["llm_answer"],
                options_text=options_text
            ))
        ]).content

    if answer is not None:
        answer = int(answer) if answer.isdigit() else None

    state["final_answer"] = {
        "id": state["id"],
        "answer": answer,
        "reasoning": " ".join(reasoning),
        "sources": sources
    }
    return state

def create_agent():
    workflow = StateGraph(Agent)
    
    workflow.add_node("parse", parse)

    workflow.add_node("generate_answer", generate_answer)
    
    workflow.add_node("search", search)
    
    workflow.add_node("decide_answer", decide_answer)
    
    workflow.set_entry_point("parse")
    
    workflow.add_edge("parse", "generate_answer")
    
    workflow.add_edge("generate_answer", "search")
    
    workflow.add_edge("search", "decide_answer")
    
    workflow.add_edge("decide_answer", END)
    
    return workflow.compile()