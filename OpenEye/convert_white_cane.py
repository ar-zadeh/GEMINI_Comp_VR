with open('OpenEye/vr_agent_qwen/white_cane.py', 'r') as f:
    text = f.read()

# Fix types import
text = text.replace("from google.genai import types", "from pydantic import BaseModel, Field\nimport base64")

# Structured generate_content
old_struct = """        parts.append(types.Part(text=prompt))
        parts.append(types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes)))

        if self.conversation_history:
            history_text = "\\nPrevious observations:\\n"
            for entry in self.conversation_history[-4:]:
                history_text += f"- At {entry['timestamp']}: {entry['summary']}\\n"
            parts.append(types.Part(text=history_text))

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=parts)],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": WhiteCaneResponse
                }
            )

            try:
                parsed = response.parsed
                if not parsed:
                    parsed = WhiteCaneResponse.model_validate_json(response.text)"""

new_struct = """        content_list = [{"type": "text", "text": prompt}]
        base64_img = base64.b64encode(img_bytes).decode('utf-8')
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}})

        if self.conversation_history:
            history_text = "\\nPrevious observations:\\n"
            for entry in self.conversation_history[-4:]:
                history_text += f"- At {entry['timestamp']}: {entry['summary']}\\n"
            content_list.append({"type": "text", "text": history_text})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content_list}],
                temperature=0.2,
                extra_body={
                   "chat_template_kwargs": {"enable_thinking": False}  
                },
                response_format={"type": "json_object"} if hasattr(self.client.chat.completions, 'response_format') else None
            )

            response_text = response.choices[0].message.content
            try:
                parsed = WhiteCaneResponse.model_validate_json(response_text)"""

text = text.replace(old_struct, new_struct)
text = text.replace("response.text", "response_text")

# Unstructured multi
old_multi = """            parts.append(types.Part(text=prompt))
            for label, img_data in captured_images:
                parts.append(types.Part(text=f"\\n[{label} View]:"))
                parts.append(types.Part(inline_data=types.Blob(mime_type="image/png", data=img_data)))

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=parts)]
            )
            result_text = response.text.strip()"""

new_multi = """            content_list = [{"type": "text", "text": prompt}]
            for label, img_data in captured_images:
                base64_img = base64.b64encode(img_data).decode('utf-8')
                content_list.append({"type": "text", "text": f"\\n[{label} View]:"})
                content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content_list}],
                temperature=0.2,
                extra_body={
                   "chat_template_kwargs": {"enable_thinking": False}  
                }
            )
            result_text = response.choices[0].message.content.strip()"""

text = text.replace(old_multi, new_multi)


with open('OpenEye/vr_agent_qwen/white_cane.py', 'w') as f:
    f.write(text)

