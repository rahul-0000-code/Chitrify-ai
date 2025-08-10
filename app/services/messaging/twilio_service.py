import logging
from typing import Optional
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from app.config.settings import settings
from app.services.messaging.base import MessagingService

logger = logging.getLogger(__name__)

class TwilioService(MessagingService):
    def __init__(self):
        if settings.twilio_account_sid and settings.twilio_auth_token:
            self.client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        else:
            self.client = None
            logger.warning("Twilio credentials not configured")
    
    async def send_message(self, to: str, message: str, media_url: Optional[str] = None) -> bool:
        """Send WhatsApp message via Twilio"""
        try:
            if not self.client:
                logger.error("Twilio client not initialized")
                return False
            
            # Ensure the "to" number has the whatsapp: prefix
            if not to.startswith("whatsapp:"):
                to = f"whatsapp:{to.replace('whatsapp:', '')}"
            
            kwargs = {
                'from_': settings.twilio_whatsapp_from,
                'to': to,
                'body': message
            }
            
            if media_url:
                kwargs['media_url'] = media_url
            
            logger.info(f"Sending WhatsApp message from {settings.twilio_whatsapp_from} to {to}")
            
            message_response = self.client.messages.create(**kwargs)
            logger.info(f"Message sent successfully. SID: {message_response.sid}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")
            return False
    
    def verify_credentials(self) -> bool:
        """Verify Twilio credentials"""
        try:
            if not self.client:
                return False
            
            account = self.client.api.accounts(settings.twilio_account_sid).fetch()
            logger.info(f"Twilio account verified: {account.friendly_name}")
            return True
        except Exception as e:
            logger.error(f"Twilio verification failed: {e}")
            return False
    
    def create_response(self) -> MessagingResponse:
        """Create Twilio messaging response object"""
        return MessagingResponse()
