import requests
import zlib
import base64
from typing import Dict, TypedDict, Optional
from enum import Enum
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph


gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key="AIzaSyDJvjsBxTcrGHRA5pRZIBL-yI1i5l4_ttU"
)


class DiagramType(str, Enum):
    PLANTUML = "plantuml"


class WorkflowState(TypedDict):
    natural_language_prompt: str
    diagram_type: Optional[DiagramType]
    diagram_code: str
    diagram_url: str
    diagram_image: bytes


def encode_diagram_source(source: str) -> str:
    """Encode diagram source for Kroki URL"""
    compressed = zlib.compress(source.strip().encode('utf-8'), level=9)
    encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
    return encoded.rstrip('=')

def generate_kroki_url(diagram_type: DiagramType, diagram_code: str, output_format: str = "svg") -> str:
    """Generate Kroki API URL for the diagram"""
    encoded = encode_diagram_source(diagram_code)
    return f"https://kroki.io/{diagram_type.value}/{output_format}/{encoded}"

def download_diagram(url: str) -> bytes:
    """Download diagram from Kroki"""
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def determine_diagram_type(state: WorkflowState) -> Dict:
    """Use Gemini to determine the most appropriate diagram type"""
    prompt = f"""Based on the following description, determine which type of diagram would be most appropriate.
    Return only the diagram type name from these options: {', '.join([t.value for t in DiagramType])}

    Description: {state['natural_language_prompt']}"""
    
    response = gemini.invoke([HumanMessage(content=prompt)])
    diagram_type = DiagramType(response.content.strip().lower())
    return {"diagram_type": diagram_type}

def generate_diagram_code(state: WorkflowState) -> Dict:
    """Use Gemini to generate the diagram code"""
    prompt = f"""Generate {state['diagram_type'].value} code for the following description.
    Return only the code, no explanations or markdown formatting.

    Description: {state['natural_language_prompt']}"""
    
    response = gemini.invoke([HumanMessage(content=prompt)])
    return {"diagram_code": response.content.strip()}

def create_kroki_diagram(state: WorkflowState) -> Dict:
    """Generate and download the diagram from Kroki"""
    url = generate_kroki_url(state['diagram_type'], state['diagram_code'])
    image = download_diagram(url)
    return {
        "diagram_url": url,
        "diagram_image": image
    }


workflow = StateGraph(WorkflowState)


workflow.add_node("determine_diagram_type", determine_diagram_type)
workflow.add_node("generate_diagram_code", generate_diagram_code)
workflow.add_node("create_kroki_diagram", create_kroki_diagram)


workflow.add_edge("determine_diagram_type", "generate_diagram_code")
workflow.add_edge("generate_diagram_code", "create_kroki_diagram")
workflow.add_edge("create_kroki_diagram", END)


workflow.set_entry_point("determine_diagram_type")


app = workflow.compile()

def generate_diagram_from_prompt(prompt: str):
    """Generate a diagram from natural language prompt"""
    try:
       
        result = app.invoke({"natural_language_prompt": prompt})
    
        # with open(output_file, 'wb') as f:
        #     f.write(result['diagram_image'])
        
        # print(f"Diagram saved as {output_file}")
        # print(f"Kroki URL: {result['diagram_url']}")
        return result['diagram_url']
    except Exception as e:
        print(f"Error generating diagram: {e}")
        return None

    
   
# print(generate_diagram_from_prompt("Generate a class diagram for a simple banking system with classes: Bank, Account, and Customer."))
