import re

def convert_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Imports
    content = re.sub(r'from google\.genai import types\n?', 'from pydantic import BaseModel, Field\nimport base64\n', content)
    content = re.sub(r'from google\.genai\n?', 'from pydantic import BaseModel, Field\nimport base64\n', content)
    
    # 1. Single text generation (e.g. Planning)
    # response = self.client.models.generate_content( ... model=self.model_name, contents=prompt, config={ ... response_schema=PlanResponse } )
    
    text_json_pattern = r'''response = self\.client\.models\.generate_content\(\s*model=self\.model_name,\s*contents=(.*?),\s*config=\{.*?response_schema:\s*(.*?)\s*\}?\s*\)'''
    def text_json_repl(m):
        prompt_var = m.group(1)
        schema_var = m.group(2)
        return (f"response = self.client.chat.completions.create(\n"
                f"                model=self.model_name,\n"
                f"                messages=[{{'role': 'user', 'content': {prompt_var}}}],\n"
                f"                temperature=0.2,\n"
                f"                extra_body={{\n"
                f"                   'chat_template_kwargs': {{'enable_thinking': False}}  \n"
                f"                }},\n"
                f"                response_format={{'type': 'json_object'}} if hasattr(self.client.chat.completions, 'response_format') else None\n"
                f"            )\n            response_text = response.choices[0].message.content")
    
    content = re.sub(text_json_pattern, text_json_repl, content, flags=re.DOTALL)
    
    # Text plain generation (no json) if any match
    text_plain_pattern = r'''response = self\.client\.models\.generate_content\(\s*model=self\.model_name,\s*contents=(.*?)\s*\)'''
    def text_plain_repl(m):
        prompt_var = m.group(1)
        if '[' in prompt_var or 'types.Content' in prompt_var: return m.group(0) # Not simple text
        return (f"response = self.client.chat.completions.create(\n"
                f"            model=self.model_name,\n"
                f"            messages=[{{'role': 'user', 'content': {prompt_var}}}],\n"
                f"            temperature=0.2,\n"
                f"            extra_body={{'chat_template_kwargs': {{'enable_thinking': False}}}}\n"
                f"        )\n        response_text = response.choices[0].message.content")

    content = re.sub(text_plain_pattern, text_plain_repl, content)
    
    # 2. Image + Text generation
    # e.g. contents=[types.Content(role="user", parts=[ types.Part(text=prompt), types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_data)) ])]
    # This is trickier to regex. We can just replace the whole block since it's quite specific.
    def replace_image_code(code):
        # We find every location of generate_content and rewrite it.
        lines = code.split('\n')
        out = []
        in_gen = False
        gen_block = []
        for line in lines:
            if 'client.models.generate_content' in line:
                in_gen = True
                gen_block.append(line)
                if ')' in line and 'types.Content' not in line: 
                    # one liner case?
                    pass
            elif in_gen:
                gen_block.append(line)
                # Check if parentheses are balanced
                block_str = '\n'.join(gen_block)
                if block_str.count('(') == block_str.count(')'):
                    in_gen = False
                    # We have the full block
                    out.append(transform_gen_block(block_str))
                    gen_block = []
            else:
                out.append(line)
        return '\n'.join(out)
        
    def transform_gen_block(block_str):
        if 'messages=[' in block_str: return block_str # Already transformed
        # Extract prompt and image variables
        indent = len(block_str) - len(block_str.lstrip(' '))
        prefix = ' ' * indent
        
        has_schema = 'response_schema' in block_str
        schema_var = None
        if has_schema:
            sm = re.search(r'response_schema["\']?\s*:\s*([A-Za-z0-9_]+)', block_str)
            if sm: schema_var = sm.group(1)

        is_multi = 'parts=parts' in block_str or 'parts=img_parts' in block_str
        if is_multi:
            # Multi image logic -> parts are already built? Let's just output manual instructions for multi
            pass
            
        print("Transforming block:", repr(block_str[:50]))
        return block_str
    
    if 'generate_content' in content:
        content = replace_image_code(content)
        
    with open(path, 'w') as f:
        f.write(content)

for path in ['OpenEye/vr_agent_qwen/planning.py', 'OpenEye/vr_agent_qwen/verification.py', 'OpenEye/vr_agent_qwen/white_cane.py', 'OpenEye/vr_agent_qwen/tools.py']:
    convert_file(path)
print("Conversion logic attempted")
