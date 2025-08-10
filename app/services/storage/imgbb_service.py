import logging
import base64
import requests
from typing import Optional

from app.config.settings import settings
from app.services.storage.base import StorageService

logger = logging.getLogger(__name__)

class ImgBBService(StorageService):
    def __init__(self):
        self.api_key = settings.imgbb_api_key
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def get_service_name(self) -> str:
        return "ImgBB"
    
    async def upload_file(self, file_content: bytes, filename: str) -> Optional[str]:
        """Upload processed image to ImgBB and return the public URL"""
        try:
            if not self.api_key:
                logger.error('IMGBB_API_KEY not configured')
                return None
            
            logger.info(f"📤 Uploading {len(file_content)} bytes to ImgBB as {filename}")
            
            # Encode image to base64
            encoded_image = base64.b64encode(file_content).decode('utf-8')
            
            # ImgBB API endpoint
            url = 'https://api.imgbb.com/1/upload'
            
            # Prepare payload
            payload = {
                'key': self.api_key,
                'image': encoded_image,
                'name': filename,
                'expiration': 0  # Never expire
            }
            
            # Upload to ImgBB
            response = requests.post(url, data=payload)
            
            if response.status_code == 200:
                json_response = response.json()
                
                if json_response.get('success'):
                    img_url = json_response['data']['url']
                    logger.info(f"✅ Successfully uploaded to ImgBB: {img_url}")
                    return img_url
                else:
                    logger.error(f"ImgBB API error: {json_response}")
                    return None
            else:
                logger.error(f"ImgBB HTTP error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Exception uploading to ImgBB: {e}")
            return None
