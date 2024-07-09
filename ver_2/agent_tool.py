from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
import sqlite3

__all__ = ['python_repl', 'add_schedule','delete_schedule_by_time','get_schedules_by_date']

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
    conn = sqlite3.connect(r"E:\WorkStation\Upload\test.db")
    return conn

@tool
def add_schedule(start_time: str, end_time: str, location: str, description: str, participants: str) -> str:
    """ 添加新的日程。提供开始时间、结束时间、地点、日程描述和参加人员。"""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO test (start_time, end_time, location, description, participants)
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
        DELETE FROM test WHERE start_time = ?;
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
        SELECT id, start_time, end_time, location, description, participants FROM test
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