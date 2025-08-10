import logging
from typing import Optional, List
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from app.services.ai.base import AIService
from app.services.ai.azure_service import AzureOpenAIService
from app.services.storage.base import StorageService
from app.services.storage.imgbb_service import ImgBBService
from app.utils.constants import ImageStyle, STYLE_DESCRIPTIONS

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        # Initialize AI services
        self.ai_services: List[AIService] = [
            AzureOpenAIService(),
            # Add other AI services here
        ]
        
        # Initialize storage services
        self.storage_services: List[StorageService] = [
            ImgBBService(),
            # Add other storage services here
        ]
    
    def get_available_ai_service(self) -> Optional[AIService]:
        """Get the first available AI service"""
        for service in self.ai_services:
            if service.is_available():
                return service
        return None
    
    def get_available_storage_service(self) -> Optional[StorageService]:
        """Get the first available storage service"""
        for service in self.storage_services:
            if service.is_available():
                return service
        return None
    
    async def process_image(self, image_content: bytes, style: ImageStyle) -> Optional[bytes]:
        """Process image with available AI service"""
        ai_service = self.get_available_ai_service()
        
        if ai_service:
            logger.info(f"Using AI service: {ai_service.get_service_name()}")
            result = await ai_service.process_image(image_content, style)
            if result:
                return result
        
        # Fallback to basic processing
        logger.warning("Using fallback image processing")
        return self._fallback_processing(image_content, style)
    
    async def upload_image(self, image_content: bytes, filename: str) -> Optional[str]:
        """Upload image using available storage service"""
        storage_service = self.get_available_storage_service()
        
        if storage_service:
            logger.info(f"Using storage service: {storage_service.get_service_name()}")
            return await storage_service.upload_file(image_content, filename)
        
        logger.error("No storage service available")
        return None
    
    def _fallback_processing(self, image_content: bytes, style: ImageStyle) -> bytes:
        """Fallback processing when AI services are not available"""
        try:
            img = Image.open(BytesIO(image_content))
            
            # Apply style-specific transformations
            if style == ImageStyle.CARTOON_3D:
                img = self._apply_cartoon_effect(img)
            elif style == ImageStyle.OIL_PAINT:
                img = self._apply_oil_paint_effect(img)
            elif style == ImageStyle.ANIME_90S:
                img = self._apply_anime_effect(img)
            elif style == ImageStyle.WATERCOLOR:
                img = self._apply_watercolor_effect(img)
            elif style == ImageStyle.VINTAGE_COLORIZED:
                img = self._apply_vintage_effect(img)
            elif style == ImageStyle.ROTOSCOPE:
                img = self._apply_rotoscope_effect(img)
            
            # Add watermark
            img = self._add_style_watermark(img, style)
            
            # Convert back to bytes
            output = BytesIO()
            img.save(output, format='JPEG', quality=85)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            return image_content
    
    def _apply_cartoon_effect(self, img: Image.Image) -> Image.Image:
        """Apply 3D cartoon-like effect"""
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.3)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        return img
    
    def _apply_oil_paint_effect(self, img: Image.Image) -> Image.Image:
        """Apply oil painting effect"""
        img = img.filter(ImageFilter.EDGE_ENHANCE)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        return img
    
    def _apply_anime_effect(self, img: Image.Image) -> Image.Image:
        """Apply anime-style effect"""
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.4)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        return img
    
    def _apply_watercolor_effect(self, img: Image.Image) -> Image.Image:
        """Apply watercolor painting effect"""
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.8)
        return img
    
    def _apply_vintage_effect(self, img: Image.Image) -> Image.Image:
        """Apply vintage colorized effect"""
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.7)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.9)
        return img
    
    def _apply_rotoscope_effect(self, img: Image.Image) -> Image.Image:
        """Apply rotoscope animation effect"""
        img = img.filter(ImageFilter.FIND_EDGES)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        return img
    
    def _add_style_watermark(self, img: Image.Image, style: ImageStyle) -> Image.Image:
        """Add style-specific watermark"""
        draw = ImageDraw.Draw(img)
        watermark_text = f"Chitrify AI - {STYLE_DESCRIPTIONS[style]}"
        text_width = len(watermark_text) * 8
        x = img.width - text_width - 10
        y = img.height - 25
        
        try:
            draw.text((x, y), watermark_text, fill="white", stroke_width=1, stroke_fill="black")
        except:
            draw.text((x, y), watermark_text, fill="white")
        
        return img
