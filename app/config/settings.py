import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://username:password@localhost/chitrify"
    
    # AI Providers
    use_azure: bool = False
    openai_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_version: Optional[str] = None
    google_gemini_api_key: Optional[str] = None
    
    # Messaging
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_whatsapp_from: str = "whatsapp:+14155238886"
    
    # Storage
    use_s3: bool = False
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_bucket_name: Optional[str] = None
    imgbb_api_key: Optional[str] = None
    
    # App
    webhook_base_url: str = "http://localhost:8000"
    free_processing: bool = True
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
