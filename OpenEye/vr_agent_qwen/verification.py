"""
vr_agent/verification.py
------------------------
Verifier: checks whether an action succeeded (Gemini 2.5 Flash).
Describer: describes a scene for accessibility (Gemini 2.5 Flash Lite).
"""

from typing import List, Tuple

import base64

from .config import MODEL_VERIFICATION, MODEL_DESCRIPTION


class Verifier:
    """Uses Gemini 2.5 Flash to verify whether an action succeeded."""

    def __init__(self, client):
        self.client = client
        self.model_name = MODEL_VERIFICATION

    def verify(self, image_data: bytes, action_description: str) -> str:
        """
        Returns a one-sentence success/failure verdict.
        Intended to be read aloud as TTS feedback.
        """
        prompt = (
            f'Verify if the following action was successful based on the image:\n'
            f'Action: "{action_description}"\n\n'
            "Be critical. If you see failure, explain why. If success, confirm it.\n"
            "Use one sentence maximum as this will be read back to the user as TTS feedback."
        )
        
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
        return response.choices[0].message.content


class Describer:
    """Uses Gemini 2.5 Flash Lite for accessibility-aware scene description."""

    def __init__(self, client):
        self.client = client
        self.model_name = MODEL_DESCRIPTION

    def describe(self, image_data: bytes, question: str) -> str:
        """
        Describe a single image in response to a question.
        Output is formatted for blind users (no bullet points, max 3 sentences).
        """
        accessibility_prompt = (
            "IMPORTANT: This response is for a blind user and will be read aloud. "
            "Output a maximum of 3 sentences with only essential details on the stuff that the user might need based on their question. "
            "Do not describe the background unless you are asked about it. "
            "CRITICAL: Do NOT use bullet points, lists, or multi-paragraph structured text. "
        )
        full_query = f"{accessibility_prompt}\n\nQuestion: {question}"

        base64_img = base64.b64encode(image_data).decode('utf-8')
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
        return response.choices[0].message.content

    def describe_multi(self, labeled_images: List[Tuple[str, bytes]], question: str) -> str:
        """
        Describe multiple labeled images (e.g. front/left/right views).

        Args:
            labeled_images: List of (label, image_bytes) tuples.
            question: The prompt / question for the model.
        """
        content_list = [{"type": "text", "text": question}]
        for label, img_data in labeled_images:
            base64_img = base64.b64encode(img_data).decode('utf-8')
            content_list.append({"type": "text", "text": f"\n[{label.upper()} VIEW]:"})
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content_list}],
            temperature=0.2,
            extra_body={
               "chat_template_kwargs": {"enable_thinking": False}  
            }
        )
        return response.choices[0].message.content
