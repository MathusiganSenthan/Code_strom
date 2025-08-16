# =============================================================================
# utils/document_processor.py - Text Extraction Utilities
# =============================================================================

import tempfile
import os
import asyncio
import fitz  # PyMuPDF

from fastapi import UploadFile
import logging
import numpy as np
from PIL import Image
import io
import re
from typing import Optional

# Try to import PaddleOCR with proper error handling
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PaddleOCR not available: {e}")
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize PaddleOCR reader only if available
ocr: Optional[PaddleOCR] = None
if PADDLEOCR_AVAILABLE:
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        logger.info("PaddleOCR initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize PaddleOCR: {e}")
        ocr = None
        PADDLEOCR_AVAILABLE = False

async def extract_text_from_pdf(file: UploadFile) -> str:
    """Extract text from uploaded PDF file using PyMuPDF and PaddleOCR (if available)."""
    
    # Read file content
    content = await file.read()
    
    # Use the bytes version for processing
    return extract_text_from_pdf_bytes(content)

def extract_text_from_pdf_bytes(pdf_content: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF and PaddleOCR (if available)."""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_content)
        tmp_path = tmp_file.name
    
    try:
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(tmp_path)
        extracted_text = ""
        
        # Process each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # First try to extract text directly (for text-based PDFs)
            page_text = page.get_text()
            
            # If no text found or very little text, try OCR (if available)
            if (not page_text.strip() or len(page_text.strip()) < 50) and PADDLEOCR_AVAILABLE and ocr is not None:
                try:
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                    img_data = pix.tobytes("png")
                    
                    # Convert to PIL Image and then to numpy array
                    pil_image = Image.open(io.BytesIO(img_data))
                    image_np = np.array(pil_image)
                    
                    # Extract text using PaddleOCR
                    ocr_results = ocr.ocr(image_np, cls=True)
                    
                    # Combine text from all detected regions
                    ocr_text = ""
                    if ocr_results and ocr_results[0]:
                        for line in ocr_results[0]:
                            if len(line) > 1 and line[1][1] > 0.5:  # confidence threshold
                                ocr_text += line[1][0] + " "
                    
                    # Use OCR text if it found more content
                    if len(ocr_text.strip()) > len(page_text.strip()):
                        page_text = ocr_text
                        logger.info(f"Used OCR for page {page_num + 1}")
                
                except Exception as ocr_error:
                    logger.warning(f"OCR failed for page {page_num + 1}: {ocr_error}")
                    # Continue with the direct text extraction result
            
            # If still no text found, add a note
            if not page_text.strip():
                page_text = f"[No text could be extracted from page {page_num + 1}]"
                if not PADDLEOCR_AVAILABLE:
                    page_text += " (OCR not available - install PaddleOCR for scanned documents)"
            
            extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        pdf_document.close()
        
        # Clean up extracted text
        text = clean_extracted_text(extracted_text)
        
        return text
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove page numbers and headers/footers (basic patterns)
    text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Fix common OCR issues
    text = text.replace('|', 'I')  # Common OCR mistake
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words
    
    return text.strip()