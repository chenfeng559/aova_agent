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
        description="content related to schedule addition, schedule query, schedule deletion, and schedule summary, 'yes' or 'no'"
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
        ("human",  "question: {question}"),
    ]
)
discrimination_agent = discriminatione_prompt | structured_llm_dis





def create_agent(llm, tools, system_message: str,current_time: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个日程管理助手，专门负责帮助用户管理和记录各种会议和活动的时间和详情。\n"
                "你可以使用以下数据库工具来对日程进行记录和查询: {tool_names}.\n"
                "请确保准确记录每个日程的开始时间和描述，使用对应的工具记录到数据库中。\n"
                "你需要根据当前的时间来对目标时间进行推断,并以'%Y-%m-%d %H:%M:%S'的格式用于对该日程时间的更改\n"
                "如果查询结果存在，则进行相应的操作，如果查询的结果不存在，则说没有相应的内容。\n"
                "所有的内容都用中文进行记录和输出。\n"
                "{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    prompt = prompt.partial(current_time=current_time)  # Add current time to the prompt
    return prompt | llm.bind_tools(tools)



tools = [
         python_repl,
         add_schedule,
         delete_schedule_by_time,
         get_schedules_by_date

        ]
tool_executor = ToolExecutor(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question:str
    score: str

model = ChatOpenAI(model = "gpt-4",temperature=0, streaming=True)

model = model.bind_tools(tools)

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], operator.add]

def binary_filter(state):
    score = state["score"]
    print("binary_filter分数",score)
    # last_message = messages[-1]
    # print("last_message:",last_message)
    # 返回最新的消息

    if score == "no":
        return "end"

    else:
        return "main_agent"


def should_continue(state):
    messages = state["messages"]
    print("这里是should_continue")
    last_message = messages[-1]
    print(last_message)
    # 返回最新的消息，判断是否是函数调用
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

def discrimination(state):
    """
    根据用户输入的内容，判断是否与日程管理功能相关。
    如果与日程管理功能相关，则返回yes，否则返回no。

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates the result of the discrimination function.

    """
    messages = state["question"]
    binary_score = discrimination_agent.invoke(
            {"question": messages}
        )
    grade = binary_score.binary_score
    print("这里是grade：",grade)
    if grade == "yes":
        print("内容相关")
        return {"messages": [messages] ,"score":grade}
    else:
        print("内容不相关")
        return {"messages": [messages] ,"score":grade}
    # return {"binary_score": binary_score, "question": messages}


# Define the function that calls the model
def main_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_call = last_message.tool_calls[0]
    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = ToolMessage(
        content=str(response), name=action.tool, tool_call_id=tool_call["id"]
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("main_agent", main_model)
workflow.add_node("binary_input", discrimination)
workflow.add_node("action", call_tool)

workflow.set_entry_point("binary_input")

# We now add a conditional edge
# workflow.add_conditional_edges(

#     "agent",
#     should_continue, 
#     {
#         # If `tools`, then we call the tool node.
#         "continue": "action",
#         # Otherwise we finish.
#         "end": END,
#     },
# )


# workflow.add_edge("action", "agent")


workflow.add_conditional_edges(

    "binary_input",
    binary_filter, 
    {
        "main_agent": "main_agent",
        "no": END,
    },
)

workflow.add_conditional_edges(
        "main_agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# 实边
workflow.add_edge("action", "main_agent")

app = workflow.compile()


def process_message(current_time, user_input):
    """
    处理用户输入并返回模型输出。

    参数：
    - current_time: str，当前时间的字符串形式
    - user_input: str，用户输入的计划内容

    返回：
    - str，模型处理后的消息内容
    """
    try:
        # 创建 HumanMessage 对象
        content = f"当前时间：{current_time}，{user_input}"
        message = HumanMessage(content=content)
        inputs = {"messages": [message]}
        
        # 运行模型并获取输出
        for output in app.stream(inputs):
            for key, value in output.items():
                return ((value['messages'])[0]).content
    except Exception as e:
        return f"发生错误：{e}"
    
if __name__ == '__main__':
   
    while True:  # 创建一个无限循环，直到用户决定退出
        try:
            # 获取当前时间
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 通过键盘输入获取用户计划
            user_input = input("请输入你的计划（输入'退出'结束程序）：")
            if user_input.lower() == '退出':  # 允许用户通过输入特定命令退出循环
                break
            
            # 创建 HumanMessage 对象
            # content = f"当前时间：{current_time}，{user_input}"
            content = f"{user_input}"
            message = HumanMessage(content=content)
            inputs = {"question": [message]}
            
            # 运行模型并获取输出
            for output in app.stream(inputs):
                for key, value in output.items():
                    print(f"Output from node '{key}':")
                    print("---")
                    # print(((value['messages'])[0]).content)
                    print(value)
                print("\n---\n")
        except KeyboardInterrupt:
            # 允许用户通过 Ctrl+C 退出循环
            print("\n程序中断，已退出。")
            break
        except Exception as e:
            print(f"发生错误：{e}")
            # 可以选择在这里也设置一个退出条件，例如输入特定的错误代码
