with open('OpenEye/vr_agent_qwen/verification.py', 'r') as f:
    text = f.read()

# single image verify
old_verify = """        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Content(role="user", parts=[
                types.Part(text=prompt),
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_data))
            ])]
        )
        return response.text"""

new_verify = """        
        base64_img = base64.b64encode(image_data).decode('utf-8')
        response = self.client.chat.completions.create(
            model=self.model_name,
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
            }
        )
        return response.choices[0].message.content"""

text = text.replace(old_verify, new_verify)

# single image describe
old_describe = """        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Content(role="user", parts=[
                types.Part(text=full_query),
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_data))
            ])]
        )
        return response.text"""

new_describe = """        base64_img = base64.b64encode(image_data).decode('utf-8')
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }],
            temperature=0.2,
            extra_body={
               "chat_template_kwargs": {"enable_thinking": False}  
            }
        )
        return response.choices[0].message.content"""

text = text.replace(old_describe, new_describe)

# multi image
old_multi = """        parts = [types.Part(text=question)]
        for label, img_data in labeled_images:
            parts.append(types.Part(text=f"\\n[{label.upper()} VIEW]:"))
            parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_data)))

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Content(role="user", parts=parts)]
        )
        return response.text"""

new_multi = """        content_list = [{"type": "text", "text": question}]
        for label, img_data in labeled_images:
            base64_img = base64.b64encode(img_data).decode('utf-8')
            content_list.append({"type": "text", "text": f"\\n[{label.upper()} VIEW]:"})
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content_list}],
            temperature=0.2,
            extra_body={
               "chat_template_kwargs": {"enable_thinking": False}  
            }
        )
        return response.choices[0].message.content"""

text = text.replace(old_multi, new_multi)

with open('OpenEye/vr_agent_qwen/verification.py', 'w') as f:
    f.write(text)
