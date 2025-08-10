import re
import io
import logging
import requests
from typing import List, Optional
from PIL import Image
from requests.auth import HTTPBasicAuth

from app.config.settings import settings

logger = logging.getLogger(__name__)

def is_safe_content(image_content: bytes) -> bool:
    """Basic content safety check"""
    try:
        img = Image.open(io.BytesIO(image_content))
        return img.size[0] > 50 and img.size[1] > 50
    except:
        return False

def extract_image_urls_from_message(message: str) -> List[str]:
    """Extract and convert image URLs from message"""
    url_patterns = [
        r'https://ibb\.co/[a-zA-Z0-9]+',
        r'https://i\.ibb\.co/[a-zA-Z0-9/.-]+',
        r'https://i\.imgur\.com/[a-zA-Z0-9]+\.[a-zA-Z]+',
        r'https://imgur\.com/[a-zA-Z0-9]+',
        r'http[s]?://[^\s]+\.(jpg|jpeg|png|gif|bmp|webp)',
    ]
    
    urls = []
    for pattern in url_patterns:
        urls.extend(re.findall(pattern, message, re.IGNORECASE))
    
    return urls

def get_direct_image_url(url: str) -> Optional[str]:
    """Convert any image URL to direct downloadable URL"""
    try:
        if 'ibb.co/' in url and not 'i.ibb.co' in url:
            return convert_imgbb_to_direct_url(url)
        elif 'imgur.com/' in url and not 'i.imgur.com' in url:
            image_id = url.split('/')[-1]
            return f"https://i.imgur.com/{image_id}.jpg"
        else:
            return url
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return url

def convert_imgbb_to_direct_url(imgbb_url: str) -> Optional[str]:
    """Convert ImgBB viewer URL to direct image URL"""
    try:
        if 'ibb.co/' in imgbb_url:
            response = requests.get(imgbb_url)
            if response.status_code == 200:
                pattern = r'https://i\.ibb\.co/[^"\'>\s]+'
                matches = re.findall(pattern, response.text)
                if matches:
                    return matches[0]
        return None
    except Exception as e:
        logger.error(f"Error converting ImgBB URL: {e}")
        return None

def download_image_from_url(image_url: str) -> Optional[bytes]:
    """Download image from public URL"""
    try:
        logger.info(f"Downloading image from: {image_url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(image_url, headers=headers, timeout=30, allow_redirects=True)
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            if 'image' in content_type:
                logger.info(f"Successfully downloaded image: {len(response.content)} bytes, type: {content_type}")
                return response.content
            else:
                logger.error(f"URL doesn't point to an image: {content_type}")
                return None
        else:
            logger.error(f"Failed to download image: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading image from URL: {e}")
        return None
