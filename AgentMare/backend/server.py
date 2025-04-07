import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from PIL import Image
import io
import uvicorn
import base64
from datetime import datetime
from custom_logging import setup_logger
from module_supervisor import ModuleSupervisor, chromaDB
from rag_model.DatabaseManager import ChromaDB as ChromaDBWrapper
from rag_model.DatabaseManager import DocumentTextSplitChunker
from rag_model.EduGPT import EduGPT
from image_advisor import StudyFeedbackAssistant

module_supervisor = ModuleSupervisor()
logger = setup_logger(__name__)

app = FastAPI()

image_to_text_assistant = StudyFeedbackAssistant('AIzaSyDJvjsBxTcrGHRA5pRZIBL-yI1i5l4_ttU', model='gemini-1.5-flash')

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/process/")
async def process_data(
        text: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None)
):
    response = {
        "timestamp": datetime.utcnow().isoformat(),
        "text_response": None,
        "image_response": None
    }

    # Process text
    # if text:
        # response["text_response"] = module_supervisor.query(prompt=text)
    logger.info("Request received with prompt %s", text)
    if file:
        if file.headers["Content-Type"] == "application/pdf":
            logger.info("Received PDF material: %s", file.filename)
            path_to_pdf = os.path.join('temp', 'temp.pdf')
            # Save the uploaded PDF to a temporary file
            os.makedirs('temp', exist_ok=True)
            with open(path_to_pdf, "wb") as pdf_file:
                pdf_file.write(await file.read())
            chromaDB.add_documents(path_to_pdf, status_callback=lambda msg, idx: logger.info(msg), document_preparer=None)
        else:
            # Process image
            try:
                # Read and process image
                image_bytes = await file.read()
                img = Image.open(io.BytesIO(image_bytes))
                img = img.convert("L")  # Convert to grayscale

                # Convert to base64 for JSON response
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                base64_img = f"data:image/png;base64,{img_str}"
                text_from_image = image_to_text_assistant.extract_text_from_image(base64_image=img_str)
                text = f"{text}? Pentru:\n {text_from_image}"
                # response["original_filename"] = image.filename

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
    supervisor_response = module_supervisor.query(prompt=text)
    if supervisor_response.startswith("https://"):
        response["image_response"] = supervisor_response
        response["text_response"] = "Diagrama ta:"
    else:
        response["text_response"] = supervisor_response
    return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)