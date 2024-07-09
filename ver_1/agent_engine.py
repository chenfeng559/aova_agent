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

from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph


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
                "所有的内容都用中文进行记录和输出。\n"
                "{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    prompt = prompt.partial(current_time=current_time)  # Add current time to the prompt
    return prompt | llm.bind_tools(tools)

repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

def connect_db():
    """ 连接到数据库 """
    conn = sqlite3.connect('ko.db')
    return conn
    
@tool
def add_schedule(start_time: str, end_time: str, location: str, description: str, participants: str) -> str:
    """ 添加新的日程。提供开始时间、结束时间、地点、日程描述和参加人员。"""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ko (start_time, end_time, location, description, participants)
        VALUES (?, ?, ?, ?, ?);
    """, (start_time, end_time, location, description, participants))
    conn.commit()
    conn.close()
    return "true"

@tool
def delete_schedule_by_time(start_time: str) -> str:
    """ 根据开始时间删除日程。"""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM ko WHERE start_time = ?;
    """, (start_time,))
    conn.commit()
    conn.close()
    return "true"

@tool
def get_schedules_by_date(query_date: str) -> str:
    """ 根据日期查询日程，将该天的所有日程查询出来。"""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, start_time, end_time, location, description, participants FROM ko
        WHERE start_time LIKE ?;
    """, (f"{query_date}%",))
    schedules = cursor.fetchall()
    conn.close()
    # 格式化查询结果为字符串
    formatted_schedules = []
    for schedule in schedules:
        formatted_schedule = (
            f"ID: {schedule[0]}, "
            f"开始时间: {schedule[1]}, "
            f"结束时间: {schedule[2]}, "
            f"地点: {schedule[3]}, "
            f"描述: {schedule[4]}, "
            f"参加人员: {schedule[5]}"
        )
        formatted_schedules.append(formatted_schedule)
    return "\n".join(formatted_schedules)

tools = [
         python_repl,
         add_schedule,
         delete_schedule_by_time,
         get_schedules_by_date

        ]
tool_executor = ToolExecutor(tools)


model = ChatOpenAI(model = "gpt-3.5-turbo",temperature=0, streaming=True)

model = model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # 返回最新的消息，判断是否是函数调用
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
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
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)


workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(

    "agent",
    should_continue, 
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
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
            content = f"当前时间：{current_time}，{user_input}"
            message = HumanMessage(content=content)
            inputs = {"messages": [message]}
            
            # 运行模型并获取输出
            for output in app.stream(inputs):
                for key, value in output.items():
                    # print(f"Output from node '{key}':")
                    print("---")
                    print(((value['messages'])[0]).content)
                print("\n---\n")
        except KeyboardInterrupt:
            # 允许用户通过 Ctrl+C 退出循环
            print("\n程序中断，已退出。")
            break
        except Exception as e:
            print(f"发生错误：{e}")
            # 可以选择在这里也设置一个退出条件，例如输入特定的错误代码
