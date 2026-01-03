import requests
import time


class LLMProcessor:
    def __init__(self, api_key, api_url):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _call_llm(self, prompt, max_retries=3):
        """Call Hugging Face API with retry logic"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True
            }
        }

        for attempt in range(max_retries):
            try:
                print(f"🔄 API Call Attempt {attempt + 1}/{max_retries}")
                print(f"📡 URL: {self.api_url}")

                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )

                print(f"📊 Status Code: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ API Response: {result}")

                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        # Format: [{"generated_text": "..."}, ...]
                        if isinstance(result[0], dict):
                            return result[0].get('generated_text', '').strip()
                        return str(result[0]).strip()
                    elif isinstance(result, dict):
                        return result.get('generated_text', '').strip()

                    return str(result).strip()

                elif response.status_code == 503:
                    print(f"⏳ Model loading... waiting 20 seconds")
                    if attempt < max_retries - 1:
                        time.sleep(20)
                        continue
                    return f"ERROR: Model is still loading after {max_retries} attempts"

                elif response.status_code == 401:
                    return "ERROR: Invalid API key. Check your HF_API_KEY in .env file"

                elif response.status_code == 403:
                    return "ERROR: Access denied. Model may require license acceptance at huggingface.co"

                else:
                    error_msg = response.text
                    print(f"❌ API Error: {error_msg}")
                    return f"ERROR: API returned {response.status_code} - {error_msg}"

            except requests.exceptions.Timeout:
                print(f"⏰ Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return "ERROR: Request timed out after multiple attempts"

            except Exception as e:
                print(f"💥 Exception: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return f"ERROR: {str(e)}"

        return "ERROR: Max retries exceeded"

    def translate(self, text, target_lang):
        """Translate text using LLM"""
        if not text.strip():
            return {'success': False, 'error': 'Empty text', 'translated_text': ''}

        # Truncate long texts
        words = text.split()
        if len(words) > 300:
            text = ' '.join(words[:300]) + "..."

        # Simple, clear prompt for translation
        prompt = f"Translate this text to {target_lang}: {text}"

        print(f"\n🌐 TRANSLATION REQUEST")
        print(f"Target Language: {target_lang}")
        print(f"Text length: {len(text)} chars")

        result = self._call_llm(prompt)

        if result and not result.startswith("ERROR:"):
            # Clean up the result - remove the original prompt if model echoes it
            cleaned = result.replace(prompt, "").strip()

            return {
                'success': True,
                'translated_text': cleaned,
                'target_lang': target_lang
            }
        else:
            return {
                'success': False,
                'error': result or 'Translation failed',
                'translated_text': ''
            }

    def summarize(self, text, max_length=150):
        """Summarize text using LLM"""
        if not text.strip():
            return {'success': False, 'error': 'Empty text', 'summary': ''}

        # Truncate very long texts
        words = text.split()
        if len(words) > 500:
            text = ' '.join(words[:500])

        # Simple, clear prompt for summarization
        prompt = f"Summarize this text in {max_length} words or less: {text}"

        print(f"\n📝 SUMMARIZATION REQUEST")
        print(f"Text length: {len(text)} chars")
        print(f"Max summary length: {max_length} words")

        result = self._call_llm(prompt)

        if result and not result.startswith("ERROR:"):
            # Clean up the result
            cleaned = result.replace(prompt, "").strip()

            return {
                'success': True,
                'summary': cleaned,
                'original_length': len(text.split()),
                'summary_length': len(cleaned.split())
            }
        else:
            return {
                'success': False,
                'error': result or 'Summarization failed',
                'summary': ''
            }