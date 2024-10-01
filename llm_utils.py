from litellm import completion
import os
from dotenv import load_dotenv
import groq
import openai

load_dotenv()

def get_llm_response(model, prompt, image_data=None, provider=None):
    try:
        if image_data:
            if provider == "groq":
                groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
                response = groq_client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                            ]
                        }
                    ]
                )
                return response.choices[0].message.content.strip()
            elif provider == "openai":
                openai.api_key = os.getenv("OPENAI_API_KEY")
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                            ]
                        }
                    ]
                )
                return response.choices[0].message.content.strip()
        else:
            # Use LiteLLM for text tasks
            messages = [{"role": "user", "content": prompt}]
            response = completion(model=model, messages=messages)
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in LLM response: {str(e)}")
        return None

def get_text_llm(provider):
    if provider == "deepinfra":
        return "deepinfra/Qwen/Qwen2.5-72B-Instruct"
    elif provider == "deepseek":
        return "deepseek-ai/deepseek-chat"
    else:
        return os.getenv("TEXT_LLM_MODEL", "gpt-3.5-turbo")

def get_vision_llm(provider):
    if provider == "groq":
        return "llama-3.2-11b-vision-preview"
    elif provider == "openai":
        return "gpt-4-vision-preview"
    else:
        return os.getenv("VISION_LLM_MODEL", "gpt-4-vision-preview")