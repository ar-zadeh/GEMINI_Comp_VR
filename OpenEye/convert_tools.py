import re

with open('OpenEye/vr_agent_qwen/tools.py', 'r') as f:
    text = f.read()

text = text.replace("from google.genai import types", "from pydantic import BaseModel, Field\nimport base64")

old_call = """            response = _grounder.client.models.generate_content(
                model=_grounder.model_name,
                contents=[types.Content(role="user", parts=[
                    types.Part(text=prompt),
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=clean_image_data))
                ])],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )

            if not response.candidates or not response.candidates[0].content.parts:
                return "Error: Gemini returned no grounding results."

            resp_text = response.candidates[0].content.parts[0].text"""

new_call = """            base64_img = base64.b64encode(clean_image_data).decode('utf-8')
            response = _grounder.client.chat.completions.create(
                model=_grounder.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }],
                temperature=0.2,
                extra_body={
                   "chat_template_kwargs": {"enable_thinking": False}  
                },
                response_format={"type": "json_object"} if hasattr(_grounder.client.chat.completions, 'response_format') else None
            )

            resp_text = response.choices[0].message.content
            if not resp_text:
                return "Error: Qwen returned no grounding results." """

text = text.replace(old_call, new_call)
text = text.replace("Gemini returned no grounding results", "Qwen returned no grounding results")
text = text.replace("Uses Gemini for", "Uses Qwen for")
text = text.replace("Gemini returned NO", "Qwen returned NO")
with open('OpenEye/vr_agent_qwen/tools.py', 'w') as f:
    f.write(text)
