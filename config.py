import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}

    # Hugging Face API - Using free, un-gated model
    HF_API_KEY = os.getenv('HF_API_KEY', '')
    # Alternative working models (no license required):
    # HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

    # OCR settings
    TESSERACT_CMD = os.getenv('TESSERACT_CMD', None)
    OCR_LANG = 'eng'

    # Supported languages
    SUPPORTED_LANGUAGES = {
        'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
        'zh': 'Chinese', 'ja': 'Japanese', 'ar': 'Arabic', 'hi': 'Hindi'
    }

    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)