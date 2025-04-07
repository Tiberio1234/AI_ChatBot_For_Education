
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import display, Image
from langchain_core.tools import tool
import io
from PIL import Image as List
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from typing import Dict, TypedDict, Optional, Any, Union
from custom_logging import setup_logger
from rag_model.DatabaseManager import ChromaDB as ChromaDBWrapper
from rag_model.DatabaseManager import DocumentTextSplitChunker
from PIL import Image as PILImage
from test_kroki import generate_diagram_from_prompt

from typing import Literal
from pydantic import BaseModel
from school_scheduler_agent import agent as school_scheduler_model, generate_schedule, run

from rag_model.EduGPT import EduGPT


# Initialize the logger
logger = setup_logger(__name__)

openKEY = 'AIzaSyDJvjsBxTcrGHRA5pRZIBL-yI1i5l4_ttU' # gemini key
chromaDB = ChromaDBWrapper(embedding_function = 'HuggingFace', openai_key=openKEY,
                           text_chunker=DocumentTextSplitChunker(chunk_size=50, chunk_overlap=10))
chromaDB.create_collection(collection='educational',
    save_path='./chroma_db_openai',
    source_path='./temp', # or folder; all inner subfolders will be traversed and pdf's parsed
    document_preparer=None,
    status_callback=lambda msg, idx: logger.info(msg))
nutri_bot = EduGPT(model='gemini-1.5-flash', api_key=openKEY,
                    retriever=chromaDB.as_retriever(k=4, search_type="similarity"),
                    include_history=False)


def extract_message_contents(data):
    """Extracts just the content from all messages in the conversation."""
    # if not isinstance(data, dict) or 'messages' not in data:
    #     return []
    
    # contents = []
    # for message in data['messages']:
    #     if isinstance(message, dict) and 'content' in message:
    #         contents.append(message['content'])
    #     elif hasattr(message, 'content'):  # Handle case where messages are objects
    #         if isinstance(message, HumanMessage):
    #             contents.append(HumanMessage(message.content))
    #         elif isinstance(message, AIMessage):
    #             contents.append(AIMessage(message.content))
    #         elif isinstance(message, ToolMessage):
    #             contents.append(ToolMessage(message.content))
    
    return data


api_key = 'AIzaSyDJvjsBxTcrGHRA5pRZIBL-yI1i5l4_ttU' # gemini key
members = ["summary_extractor", "diagram_generator", "problem_adviser", "school_scheduler"]
options = members + ["FINISH"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
summary_agent_prompt = """YOU MUST USE the provided tool for ALL responses. 
    STRICTLY USE THE PROVIDED TOOL. Strictly extract the user id and prompt from the state."""

diagram_gnerator_agent_prompt = """You are a specialized diagram generator. Strictly
extract an appropriate prompt for the tool generator. Do not ask for more details. STRICTLY use the provided tool. The generation is ALWAYS succesful.
"""

# problem_adviser_agent_prompt = """You MUST USE the problem_adviser_tool for ALL responses. STRICTLY USE THE PROVIDED TOOL. Strictly extract the problem."""
problem_adviser_agent_prompt =  """You are a helpful study assistant. Analyze the provided extracted text and\ 
provide specific feedback based on the user's request without giving the entire solution. Rather, guide the user towards the solution,\ 
based on the provided blocker. Be concise but thorough. Focus on key areas for improvement."""

school_scheduler_agent_prompt = """You are a specialized school scheduler. Do NOT ask for more 
details. STRICTLY use the provided tool. You can call the prompt with INITIALIZE GENERATION
or MODIFY as prefix. You ALWAYS WORK ON A DEFAULT SCHEDULE."""

class Router(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["summary_extractor", "diagram_generator", "problem_adviser", "school_scheduler", "FINISH"]

class State(MessagesState):
    next: str

llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    temperature=0.3,
    google_api_key=api_key)

def supervisor_node(state: State) -> Command[Literal["summary_extractor", "diagram_generator", "problem_adviser", "school_scheduler", "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response.next  # Get the 'next' value from the Router object
    if state["messages"][-1].name == goto:
        goto = "FINISH"
    logger.info(f"SUPERVISOR_NODE: {goto}")
    if goto == "FINISH":
        goto = "__end__"  # Use "__end__" instead of END for consistency

    return Command(goto=goto, update={"next": goto})

@tool
def context_extractor_tool(prompt: str) -> str:
    """
    Use this tool to extract the materials of the uuid user, relevant to the prompt.
    """
    logger.info("Context extractor tool called")
    result, complete = nutri_bot.ask(prompt)
    # print(f"Context extracted for {uuid} and prompt: {prompt}")
    return result

summary_agent = create_react_agent(
    llm, tools=[context_extractor_tool], prompt=summary_agent_prompt
)

def summary_node(state: State) -> Command[Literal["supervisor"]]:
    result = summary_agent.invoke(state) 
    # output = result["messages"][-1].content
    
    logger.info(f"\nSummary output: {extract_message_contents(result)}")  # Debug print
    
    return Command(
        update={
            "messages": state['messages'] + [
                HumanMessage(content=result["messages"][-2].content, name="summary_extractor")
            ]
        },
        goto="supervisor",
    )

@tool
def diagram_generator_tool(prompt: str) -> str:
    """
    Use this tool to generate a diagram from a context and returns it as an URL. The generation is succesful.
    """
    logger.info("Diagram generator tool called")
    return generate_diagram_from_prompt(prompt)

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
diagram_agent = create_react_agent(llm, tools=[diagram_generator_tool], prompt=diagram_gnerator_agent_prompt)

def diagram_node(state: State) -> Command[Literal["supervisor"]]:
    result = diagram_agent.invoke(state)
    logger.info(f"\nDiagram Output: {extract_message_contents(result)}")  # Debug print
    # print(f"\nMessages: {state['messages']}")  # Debug print
    return Command(
        update={
            "messages": state['messages'] + [
                HumanMessage(content=result["messages"][-2].content, name="diagram_generator")
            ]
        },
        goto="supervisor",
    )

@tool
def problem_adviser_tool(problem: str, prompt: str) -> str:
    """
    Use this tool to provide problem solving hints for the given problem.
    """
    # print("~~~~~~~advice provided~~~~~~")
    logger.info("Problem adviser tool called")
    global last_tool_called_result
    last_tool_called_result = f"Advice provided for {problem} and question: {prompt}"
    return f"Advice provided"


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
problem_adviser_agent = create_react_agent(llm, tools=[], prompt=problem_adviser_agent_prompt)

def problem_adviser_node(state: State) -> Command[Literal["supervisor"]]:
    result = problem_adviser_agent.invoke(state)
    logger.info(f"Adviser Output: {extract_message_contents(result)}")  # Debug print
    
    return Command(
        update={
            "messages": state['messages'] + [
                HumanMessage(content=result["messages"][-1].content, name="problem_adviser")
            ]
        },
        goto="supervisor",
    )

@tool
def school_scheduler_tool(prompt: str) -> str:
    """
    Use this tool to generate a school schedule.
    """
    logger.info("School scheduler tool called")
    if prompt.startswith("INITIALIZE GENERATION"):
        result = generate_schedule()
    else:
        result = run(prompt[7:])
    if "fail" in result:
        result = "The schedule is already optimal."
    return result

school_scheduler_agent = create_react_agent(llm, tools=[school_scheduler_tool], prompt=school_scheduler_agent_prompt)

def school_scheduler_node(state: State) -> Command[Literal["supervisor"]]:
    result = school_scheduler_agent.invoke(state)
    logger.info(f"\nSchool Scheduler Output: {extract_message_contents(result)}")  # Debug print
    return Command(
        update={
            "messages": state['messages'] + [
                HumanMessage(content=result["messages"][-2].content, name="school_scheduler")
            ]
        },
        goto="supervisor",
    )


builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("summary_extractor", summary_node)
builder.add_node("diagram_generator", diagram_node)
builder.add_node("problem_adviser", problem_adviser_node)
builder.add_node("school_scheduler", school_scheduler_node)
graph = builder.compile()


class ModuleSupervisor:

    def __init__(self):
        pass

    def query(self, prompt: str) -> Dict[str, Any]:
        """
        Query the module with a user id and prompt.
        """
        # "Extract the materials provided by user 5 for 'flow of data'. Generate a diagram from them."
        for s in graph.stream( 
            {"messages": [HumanMessage(content=prompt, name="user")], "next": "supervisor"}, 
            subgraphs=True
            ):
            if s[1].get("supervisor", {"next": "o"})["next"] == "__end__":
                continue
            last_s = s
        try:
            logger.debug("~~~~~~ LAST RESULT ~~~~~")
            logger.debug(last_s)
            result = list(last_s[1].values())[-1]['messages'][-1].content
        except:
            result = "An error occured"
        return result
        

