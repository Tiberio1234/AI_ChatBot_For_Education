import base64
import cv2
import numpy as np
import easyocr
from typing import Optional, Tuple, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate
import asyncio

class StudyFeedbackAssistant:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """
        Initialize the feedback assistant with Google Gemini.
        
        Args:
            api_key (str): Your Google AI Studio API key
            model (str): Which model to use (defaults to gemini-1.5-flash)
        """
        # self.llm = ChatGoogleGenerativeAI(
        #     model=model,
        #     google_api_key=api_key,
        #     temperature=0.3
        # )
        # self.context_history = []  # Stores conversation context
        # self.max_context_items = 5  # How much history to keep
        
        # Initialize EasyOCR reader (only do this once)
        self.reader = easyocr.Reader(['en'])  # For English text
        
        # # Define the prompt template for text-based analysis
        # self.prompt_template = ChatPromptTemplate.from_messages([
        #     SystemMessage(content=(
        #         "You are a helpful study assistant. Analyze the provided extracted text from handwritten work and "
        #         "provide specific feedback based on the user's request without giving the entire solution. "
        #         "Rather, guide the user towards the solution, based on the provided blocker. "
        #         "Be concise but thorough. "
        #         "Focus on key areas for improvement. If something in the text seems unclear, note that it might be due to OCR limitations."
        #     )),
        #     ("human", "Extracted Text:\n{extracted_text}\n\nQuestion: {question}\nContext: {context}")
        # ])

    def extract_text_from_image(self, base64_image: str) -> str:
        """
        Extract text from the image using EasyOCR.
        
        Args:
            base64_image (str): Base64 encoded image string
            
        Returns:
            str: Extracted text from the image
        """
        # Decode base64 to image bytes
        image_bytes = base64.b64decode(base64_image)
        
        # Convert bytes to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Use EasyOCR to detect text
        results = self.reader.readtext(enhanced)
        
        # Extract and combine text
        extracted_text = "\n".join([result[1] for result in results])
        
        return extracted_text
    
    def format_context(self) -> str:
        """Format the context history for the prompt."""
        return "\n".join(self.context_history) if self.context_history else "No previous context"
    
    def update_context(self, question: str, feedback: str) -> None:
        """Update the context history with the latest interaction."""
        self.context_history.append(f"Q: {question}\nA: {feedback[:200]}")
        if len(self.context_history) > self.max_context_items:
            self.context_history.pop(0)
    
    async def generate_feedback(
        self, 
        base64_image: str, 
        user_question: str,
    ) -> str:
        """
        Generate feedback based on extracted text from the image and user question.
        
        Args:
            base64_image (str): Base64 encoded image
            user_question (str): The specific question/request for feedback
            
        Returns:
            str: The generated feedback
        """
        try:
            # Extract text from the image
            extracted_text = self.extract_text_from_image(base64_image)
            
            if not extracted_text.strip():
                return "No text could be extracted from the image. Please try with a clearer image or manually transcribe the content."
            
            # Create the prompt with extracted text and context
            prompt = self.prompt_template.format_messages(
                extracted_text=extracted_text,
                question=user_question,
                context=self.format_context()
            )
            
            # Get response from Gemini
            response = await self.llm.agenerate([prompt])
            feedback = response.generations[0][0].text
            
            # Update context
            self.update_context(user_question, feedback)
            
            return feedback
        
        except Exception as e:
            return f"Error generating feedback: {str(e)}"

class StudyFeedbackWrapper:
    def __init__(self, api_key: str):
        """
        Simplified wrapper for the StudyFeedbackAssistant.
        
        Args:
            api_key (str): Your Google AI Studio API key
        """
        self.assistant = StudyFeedbackAssistant(api_key)
    
    async def get_feedback(
        self,
        base64_image: str,
        question: str,
    ) -> str:
        """
        Get feedback on a base64 encoded image with a single method call.
        
        Args:
            base64_image (str): Base64 encoded image string
            question (str): Your question about the content
            
        Returns:
            str: The generated feedback
        """
        # Process and get feedback
        feedback = await self.assistant.generate_feedback(
            base64_image=base64_image,
            user_question=question
        )
        return feedback

async def main():
    # Initialize with your API key
    wrapper = StudyFeedbackWrapper(api_key="AIzaSyDJvjsBxTcrGHRA5pRZIBL-yI1i5l4_ttU")
    
    image_path = "test.png"  # Example: photo of handwritten math work
    frame = cv2.imread(image_path)
    
    _, buffer = cv2.imencode('.png', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    # Get feedback
    feedback = await wrapper.get_feedback(
        base64_image=base64_image,
        question="What should I do at this point?"
    )
    
    print("Generated Feedback:")
    print(feedback)

# Run the example
if __name__ == "__main__":
    asyncio.run(main())