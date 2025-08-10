import logging
import base64
from typing import Optional
from io import BytesIO
from PIL import Image
from openai import AzureOpenAI

from app.config.settings import settings
from app.services.ai.base import AIService
from app.utils.constants import ImageStyle, STYLE_PROMPTS

logger = logging.getLogger(__name__)

class AzureOpenAIService(AIService):
    def __init__(self):
        self.client = None
        if (settings.use_azure and settings.openai_api_key and 
            settings.azure_endpoint and settings.azure_deployment):
            try:
                self.client = AzureOpenAI(
                    api_key=settings.openai_api_key,
                    azure_endpoint=settings.azure_endpoint,
                    api_version=settings.azure_api_version
                )
                logger.info("Azure OpenAI configured successfully")
            except Exception as e:
                logger.error(f"Failed to configure Azure OpenAI: {e}")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def get_service_name(self) -> str:
        return "Azure OpenAI GPT-4 Vision"
    
    async def process_image(self, image_content: bytes, style: ImageStyle) -> Optional[bytes]:
        """Process image using Azure OpenAI GPT-4 Vision"""
        try:
            if not self.client:
                return None
            
            logger.info("Processing with Azure OpenAI GPT-4 Vision")
            
            # Convert image to PIL Image for processing
            original_img = Image.open(BytesIO(image_content))
            
            # Resize if too large (GPT-4 Vision has size limits)
            max_size = 2048
            if original_img.width > max_size or original_img.height > max_size:
                original_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
            
            # Convert to base64 for OpenAI
            img_byte_arr = BytesIO()
            original_img.save(img_byte_arr, format='JPEG', quality=85)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Get style prompt
            style_prompt = STYLE_PROMPTS.get(style, "Transform this image artistically")
            
            # Create the prompt for image analysis
            analysis_prompt = f"""
            {style_prompt}
            
            Please analyze this image and provide detailed recommendations for transforming it into the requested style. 
            Focus on:
            1. Color adjustments needed (saturation, brightness, contrast)
            2. Texture and filter effects to apply
            3. Lighting modifications required
            4. Specific artistic elements to emphasize
            
            Be specific about technical image processing steps that would achieve this transformation.
            """
            
            # Call Azure OpenAI GPT-4 Vision
            response = self.client.chat.completions.create(
                model=settings.azure_deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            ai_analysis = response.choices[0].message.content
            logger.info(f"Azure OpenAI analysis: {ai_analysis[:100]}...")
            
            # Return the processed image bytes (you would enhance this with actual processing)
            return img_byte_arr.getvalue()
            
        except Exception as e:
            logger.error(f"Azure OpenAI processing failed: {e}")
            return None
