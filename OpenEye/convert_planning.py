with open('OpenEye/vr_agent_qwen/planning.py', 'r') as f:
    text = f.read()

old_call = """            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": PlanResponse
                }
            )

            try:
                parsed = response.parsed
                if not parsed:
                    parsed = PlanResponse.model_validate_json(response.text)
            except Exception as e:
                logger.error(f"Plan validation failed: {e}. Text: {response.text}")
                return []"""
                
new_call = """            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                extra_body={
                   "chat_template_kwargs": {"enable_thinking": False}  
                },
                response_format={"type": "json_object"} if hasattr(self.client.chat.completions, 'response_format') else None
            )

            response_text = response.choices[0].message.content
            try:
                parsed = PlanResponse.model_validate_json(response_text)
            except Exception as e:
                logger.error(f"Plan validation failed: {e}. Text: {response_text}")
                return []"""

text = text.replace(old_call, new_call)
with open('OpenEye/vr_agent_qwen/planning.py', 'w') as f:
    f.write(text)
