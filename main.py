from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from deep_translator import GoogleTranslator
import os
import io
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = ""

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the domain of your frontend in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...), query: str = Form(...), translation_language: str = Form(None)):
    try:
        # Read the uploaded file
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        
        # Extract text from the PDF
        raw_text = ''
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?"],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_text(raw_text)

        # Generate embeddings and perform similarity search
        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)
        docs = document_search.similarity_search(query)
        
        # Load QA chain and get answer
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        context = "You are a lawyer and provide assistance with legal questions."
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nPlease provide a detailed answer based on the given documents."
        answer = chain.run(input_documents=docs, question=prompt)

        # Handle translation if requested
        if translation_language:
            try:
                # Providing context to translation API
                context_translation = f"Translate this legal answer to {translation_language}: {answer}"
                translated_answer = GoogleTranslator(source='auto', target=translation_language).translate(context_translation)
                
                # Postprocess translated answer (Optional)
                # Add any specific postprocessing logic here
                
                return JSONResponse(content={"answer": answer, "translated_answer": translated_answer})
            except Exception as e:
                logger.error(f"Translation error: {e}")
                return JSONResponse(content={"answer": answer, "error": str(e)})

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

