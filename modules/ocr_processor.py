import cv2
import pytesseract
from PIL import Image
import PyPDF2
from pdf2image import convert_from_path
import os


class OCRProcessor:
    def __init__(self, tesseract_cmd=None, language='eng'):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.language = language

    def extract_text(self, file_path):
        """Extract text from image or PDF"""
        try:
            if file_path.lower().endswith('.pdf'):
                return self._extract_from_pdf(file_path)
            else:
                return self._extract_from_image(file_path)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': ''
            }

    def _extract_from_image(self, image_path):
        """Extract text from image"""
        # Read and preprocess image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gray = cv2.medianBlur(gray, 3)

        # Perform OCR
        text = pytesseract.image_to_string(gray, lang=self.language)

        return {
            'success': True,
            'text': text.strip(),
            'word_count': len(text.split()),
            'char_count': len(text)
        }

    def _extract_from_pdf(self, pdf_path):
        """Extract text from PDF"""
        text = ""

        # Try direct text extraction first
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

        # If no text found, use OCR on images
        if not text.strip():
            images = convert_from_path(pdf_path)
            for img in images:
                img_cv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
                text += pytesseract.image_to_string(img_cv, lang=self.language) + "\n"

        return {
            'success': True,
            'text': text.strip(),
            'word_count': len(text.split()),
            'char_count': len(text)
        }