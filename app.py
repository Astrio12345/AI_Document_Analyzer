from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from config import Config
from modules.ocr_processor import OCRProcessor
from modules.llm_processor import LLMProcessor

app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# Initialize modules
ocr = OCRProcessor(
    tesseract_cmd=app.config['TESSERACT_CMD'],
    language=app.config['OCR_LANG']
)

llm = LLMProcessor(
    api_key=app.config['HF_API_KEY'],
    api_url=app.config['HF_API_URL']
)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html', languages=app.config['SUPPORTED_LANGUAGES'])


@app.route('/process', methods=['POST'])
def process_document():
    """Process document through complete pipeline"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    target_lang = request.form.get('target_lang')
    summarize_flag = request.form.get('summarize', 'false').lower() == 'true'

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file'}), 400

    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Step 1: OCR
        ocr_result = ocr.extract_text(filepath)

        if not ocr_result['success']:
            os.remove(filepath)
            return jsonify(ocr_result), 500

        extracted_text = ocr_result['text']
        result = {'ocr': ocr_result}

        # Step 2: Translation (if requested)
        if target_lang and app.config['HF_API_KEY']:
            lang_name = app.config['SUPPORTED_LANGUAGES'].get(target_lang, target_lang)
            trans_result = llm.translate(extracted_text, lang_name)
            result['translation'] = trans_result
            text_to_summarize = trans_result.get('translated_text', extracted_text)
        else:
            text_to_summarize = extracted_text

        # Step 3: Summarization (if requested)
        if summarize_flag and app.config['HF_API_KEY']:
            max_length = int(request.form.get('max_length', 150))
            summ_result = llm.summarize(text_to_summarize, max_length)
            result['summarization'] = summ_result

        # Clean up
        os.remove(filepath)

        return jsonify({'success': True, 'results': result})

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({
            'success': False,
            'error': f'Processing error: {str(e)}'
        }), 500


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'ocr': True,
        'llm': bool(app.config['HF_API_KEY'])
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': f'File too large. Max: {app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)}MB'
    }), 413


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)