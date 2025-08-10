from typing import Optional
from pydantic import BaseModel
from app.utils.constants import ImageStyle, PaymentStatus

class WhatsAppMessage(BaseModel):
    From: str
    To: str
    Body: str
    MediaUrl0: Optional[str] = None
    MediaContentType0: Optional[str] = None

class StyleSelection(BaseModel):
    style: ImageStyle
    whatsapp_number: str

class PaymentRequest(BaseModel):
    amount: float
    currency: str
    whatsapp_number: str
    style: ImageStyle
