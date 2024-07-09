import os

os.environ["TAVILY_API_KEY"]=""
os.environ["LANGCHAIN_API_KEY"]=""
os.environ["OPENAI_API_KEY"]=""
os.environ["DASHSCOPE_API_KEY"] = ''

# Optional, add tracing in LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = ""
os.environ["LANGCHAIN_API_KEY"] =""