from abc import ABC, abstractmethod
from typing import Optional

class MessagingService(ABC):
    """Abstract messaging service interface"""
    
    @abstractmethod
    async def send_message(self, to: str, message: str, media_url: Optional[str] = None) -> bool:
        """Send a message to a recipient"""
        pass
    
    @abstractmethod
    def verify_credentials(self) -> bool:
        """Verify service credentials"""
        pass
