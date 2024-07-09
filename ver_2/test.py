
from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
import datetime
import sqlite3
import config
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolExecutor
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
import datetime
from agent_tool import *
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

# 判别输入信息类型
class ContentDiscrimination(BaseModel):
    """判别用户输入的内容是否与，输出返回yes或者no."""

    binary_score: str = Field(
        description="内容是否和产品定义相关, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-4", temperature=0)
structured_llm_dis = llm.with_structured_output(ContentDiscrimination)

# Prompt
filter_system = """
    目标是过滤掉错误的检索结果。\n
    如果文档包含与用户问题相关的关键词或语义，请将其评为相关。\n
    给出二元评分 'yes' 或 'no'，以指示文档是否与问题相关。\n
    作为一个日程管理工具的智能助手，你需要判断用户的问题是否与产品功能（如日程添加、提醒、删除、查询等）有关。如果问题与这些功能无关，请返回消息：“AOVA还在学习中，请和我聊聊你的记忆吧。\n”
    相关问题的示例：\n
    - 如何添加一个新的日程？能提醒我明天的会议吗？我如何删除一个日程？可以查询一下我下周的日程安排吗？\n
    不相关问题的示例：\n
    - 你知道今天的天气吗？能帮我查一下附近的餐馆吗？你能讲个笑话吗？你喜欢什么电影？\n

    任务：1. 如果用户的问题与日程管理功能相关，请评为“是”并提供相关的帮助或信息。 2. 如果用户的问题与日程管理功能无关，请评为“否”并返回消息：“AOVA还在学习中，请和我聊聊你的记忆吧。\n"""
discriminatione_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", filter_system),
        ("human",  "User input: {question}"),
    ]
)
discrimination_agent = discriminatione_prompt | structured_llm_dis

question = "我今天有什么安排？"

print(discrimination_agent.invoke({"question": question}))