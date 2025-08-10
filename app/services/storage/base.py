from abc import ABC, abstractmethod
from typing import Optional

class StorageService(ABC):
    """Abstract storage service interface"""
    
    @abstractmethod
    async def upload_file(self, file_content: bytes, filename: str) -> Optional[str]:
        """Upload file and return public URL"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if storage service is available"""
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """Get storage service name"""
        pass
