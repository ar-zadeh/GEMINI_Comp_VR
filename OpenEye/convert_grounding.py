with open("OpenEye/vr_agent_qwen/grounding.py", "r") as f:
    text = f.read()

text = text.replace("from google.genai import types", "from pydantic import BaseModel, Field\nimport base64")

old_call = """            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(
                            mime_type="image/jpeg", data=clean_image_data
                        ))
                    ])
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GroundingResponse,
                }
            )

            if not response.candidates:"""

new_call = """            base64_img = base64.b64encode(clean_image_data).decode('utf-8')
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                        ]
                    }
                ],
                temperature=0.2,
                extra_body={
                   "chat_template_kwargs": {"enable_thinking": False}  
                },
                response_format={"type": "json_object"} if hasattr(self.client.chat.completions, 'response_format') else None
            )
            
            response_text = response.choices[0].message.content
            if not response_text:"""

text = text.replace(old_call, new_call)

old_check1 = """            try:
                parsed_response = response.parsed
                if not parsed_response:
                    parsed_response = GroundingResponse.model_validate_json(response.text)"""
new_check1 = """            try:
                parsed_response = GroundingResponse.model_validate_json(response_text)"""
text = text.replace(old_check1, new_check1)

text = text.replace("response.text", "response_text")
text = text.replace("response.prompt_feedback", "'Unknown'")

with open("OpenEye/vr_agent_qwen/grounding.py", "w") as f:
    f.write(text)
print("grounding done")
