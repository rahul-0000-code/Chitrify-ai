
import os
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, File, Response, UploadFile, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import redis
from celery import Celery
import posthog
import requests
import json
from PIL import Image
import io
import boto3
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Chitrify AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chitrify.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# Celery setup
celery_app = Celery(
    "chitrify",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379")
)

# External service clients
twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
posthog.api_key = os.getenv("POSTHOG_API_KEY")
posthog.host = os.getenv("POSTHOG_HOST", "https://app.posthog.com")

# AWS S3 setup (optional - will fallback to local storage)
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
    USE_S3 = AWS_BUCKET_NAME is not None
except:
    s3_client = None
    AWS_BUCKET_NAME = None
    USE_S3 = False
    logger.info("S3 not configured, will use local storage")

# Enums and Constants
class ImageStyle(str, Enum):
    CARTOON_3D = "3d_cartoon"
    OIL_PAINT = "oil_paint"
    ROTOSCOPE = "rotoscope"
    ANIME_90S = "anime_90s"
    VINTAGE_COLORIZED = "vintage_colorized"
    WATERCOLOR = "watercolor"

class PaymentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

class RenderStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

STYLE_DESCRIPTIONS = {
    ImageStyle.CARTOON_3D: "🎬 3D Cartoon (Pixar-style)",
    ImageStyle.OIL_PAINT: "🎨 Oil-paint Portrait",
    ImageStyle.ROTOSCOPE: "🎵 Rotoscope Outline",
    ImageStyle.ANIME_90S: "📺 90s Anime Cel",
    ImageStyle.VINTAGE_COLORIZED: "📸 Vintage Colourised",
    ImageStyle.WATERCOLOR: "🖌️ Water-colour Sketch"
}

# For testing - all processing is free
FREE_PROCESSING = True

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    whatsapp_number = Column(String, unique=True, index=True)
    country_code = Column(String, default="US")
    free_credits = Column(Integer, default=0)
    total_referrals = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    
    orders = relationship("Order", back_populates="user")
    renders = relationship("ImageRender", back_populates="user")

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    amount = Column(Float)
    currency = Column(String)
    payment_status = Column(String, default=PaymentStatus.PENDING)
    payment_gateway = Column(String)  # stripe or razorpay
    gateway_payment_id = Column(String)
    created_at = Column(DateTime, default=func.now())
    
    user = relationship("User", back_populates="orders")
    renders = relationship("ImageRender", back_populates="order")

class ImageRender(Base):
    __tablename__ = "image_renders"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    order_id = Column(String, ForeignKey("orders.id"), nullable=True)
    original_image_url = Column(String)
    rendered_image_url = Column(String, nullable=True)
    style = Column(String)
    status = Column(String, default=RenderStatus.QUEUED)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(days=30))
    
    user = relationship("User", back_populates="renders")
    order = relationship("Order", back_populates="renders")

class Referral(Base):
    __tablename__ = "referrals"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    referrer_id = Column(String, ForeignKey("users.id"))
    referee_id = Column(String, ForeignKey("users.id"))
    completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
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

# Utility functions
def get_or_create_user(whatsapp_number: str, db: Session) -> User:
    user = db.query(User).filter(User.whatsapp_number == whatsapp_number).first()
    if not user:
        # Determine country based on phone number
        country_code = "IN" if whatsapp_number.startswith("whatsapp:+91") else "US"
        user = User(whatsapp_number=whatsapp_number, country_code=country_code)
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Track new user
        posthog.capture(user.id, 'user_created', {
            'whatsapp_number': whatsapp_number,
            'country': country_code
        })
    
    return user

def upload_to_s3(file_content: bytes, filename: str) -> str:
    """Upload file to S3 or save locally for testing"""
    if USE_S3 and s3_client:
        try:
            key = f"images/{datetime.utcnow().strftime('%Y/%m/%d')}/{filename}"
            s3_client.put_object(
                Bucket=AWS_BUCKET_NAME,
                Key=key,
                Body=file_content,
                ContentType='image/jpeg'
            )
            return f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{key}"
        except:
            logger.error("S3 upload failed, using local storage")
    
    # Fallback to local storage for testing
    os.makedirs("uploads", exist_ok=True)
    local_path = f"uploads/{filename}"
    with open(local_path, "wb") as f:
        f.write(file_content)
    
    # Return a local URL (you'll need to serve this statically in production)
    return f"{os.getenv('WEBHOOK_BASE_URL', 'http://localhost:8000')}/uploads/{filename}"

def send_whatsapp_message(to: str, message: str, media_url: str = None):
    """Send WhatsApp message via Twilio"""
    try:
        kwargs = {
            'from_': os.getenv("TWILIO_WHATSAPP_FROM"),
            'to': to,
            'body': message
        }
        if media_url:
            kwargs['media_url'] = media_url
            
        twilio_client.messages.create(**kwargs)
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message: {e}")

def is_safe_content(image_content: bytes) -> bool:
    """Basic content safety check - you can integrate with AWS Rekognition or similar"""
    # For POC, just check file size and basic image validation
    try:
        img = Image.open(io.BytesIO(image_content))
        return img.size[0] > 50 and img.size[1] > 50  # Basic validation
    except:
        return False

# Celery tasks
@celery_app.task
def process_image_render(render_id: str):
    """Process image rendering using AI model"""
    db = SessionLocal()
    try:
        render = db.query(ImageRender).filter(ImageRender.id == render_id).first()
        if not render:
            return
        
        render.status = RenderStatus.PROCESSING
        db.commit()
        
        # Download original image
        response = requests.get(render.original_image_url)
        if response.status_code != 200:
            raise Exception("Failed to download original image")
        
        # Process with AI model (using Replicate as example)
        # You can replace this with any AI service
        processed_image = process_with_ai_model(response.content, render.style)
        
        # Upload processed image
        filename = f"rendered_{render_id}.jpg"
        rendered_url = upload_to_s3(processed_image, filename)
        
        render.rendered_image_url = rendered_url
        render.status = RenderStatus.COMPLETED
        render.completed_at = func.now()
        db.commit()
        
        # Send result to user
        user = render.user
        message = f"✨ Your {STYLE_DESCRIPTIONS[render.style]} is ready!\n\n🚀 Forward this chat to one friend → get 1 free render when they try Chitrify."
        send_whatsapp_message(user.whatsapp_number, message, rendered_url)
        
        # Track completion
        posthog.capture(user.id, 'image_generated', {
            'style': render.style,
            'render_id': render_id
        })
        
    except Exception as e:
        render.status = RenderStatus.FAILED
        render.error_message = str(e)
        db.commit()
        
        # Refund if payment was made (not applicable in free mode)
        # if render.order_id:
        #     # Implement refund logic here
        #     pass
        
        # Notify user
        send_whatsapp_message(
            render.user.whatsapp_number,
            "Sorry, there was an issue processing your image. Please try again or contact support."
        )
        
        logger.error(f"Image processing failed for {render_id}: {e}")
    
    finally:
        db.close()

def process_with_ai_model(image_content: bytes, style: str) -> bytes:
    """Process image with AI model - simplified for testing"""
    # For testing, just return a processed version (add a watermark or simple effect)
    try:
        img = Image.open(io.BytesIO(image_content))
        
        # Simple processing based on style (for demo)
        if style == ImageStyle.CARTOON_3D:
            # Convert to RGB and resize for demo
            img = img.convert('RGB')
            img = img.resize((min(800, img.width), min(800, img.height)))
        elif style == ImageStyle.OIL_PAINT:
            # Apply a simple filter effect
            img = img.convert('RGB')
        
        # Add a simple watermark for testing
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Try to add text watermark
        try:
            # Use default font
            draw.text((10, 10), f"Chitrify AI - {style}", fill="white")
        except:
            pass  # Skip if font issues
        
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=85)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        # Return original image if processing fails
        return image_content

# API Routes
@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Handle incoming WhatsApp messages"""
    form_data = await request.form()
    
    from_number = form_data.get("From")
    message_body = form_data.get("Body", "").strip()
    media_url = form_data.get("MediaUrl0")
    media_type = form_data.get("MediaContentType0")
    
    # Track message received
    posthog.capture(from_number, 'wa_invoked', {
        'message_body': message_body,
        'has_media': media_url is not None
    })
    
    user = get_or_create_user(from_number, db)
    
    resp = MessagingResponse()
    resp.message(f"Debug:\nMedia URL: {media_url}\nMedia Type: {media_type}")
    
    # Check if message contains /Chitrify command and image
    if "/chitrify" in message_body.lower() and media_url:
        logger.info(f"Incoming from {from_number}: Body={message_body}, MediaUrl={media_url}, MediaType={media_type}")
        # Download and validate image
        try:
            media_response = requests.get(media_url)
            if media_response.status_code != 200:
                resp.message("Sorry, I couldn't download your image. Please try again.")
                return str(resp)
            
            image_content = media_response.content
            
            # Validate image
            if len(image_content) > 10 * 1024 * 1024:  # 10MB limit
                resp.message("Image too large. Please send an image smaller than 10MB.")
                return str(resp)
            
            if not is_safe_content(image_content):
                resp.message("Sorry, I can't process this image. Please send a different photo.")
                return str(resp)
            
            # Upload original image
            filename = f"original_{uuid.uuid4()}.jpg"
            original_url = upload_to_s3(image_content, filename)
            
            # Track image upload
            posthog.capture(user.id, 'image_uploaded', {
                'file_size': len(image_content),
                'media_type': media_type
            })
            
            # Show style selection
            styles_text = "Choose your style:\n\n"
            for i, (style, desc) in enumerate(STYLE_DESCRIPTIONS.items(), 1):
                styles_text += f"{i}️⃣ {desc}\n"
            
            styles_text += f"\n🎉 FREE TESTING MODE - All styles are free!"
            styles_text += "\n\nReply with the number (1-6) to continue."
            
            # Store image URL in Redis for this user
            redis_client.setex(f"pending_image:{user.id}", 300, original_url)  # 5 min expiry
            
            resp.message(styles_text)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            resp.message("Sorry, there was an error processing your image. Please try again.")
    
    elif message_body.isdigit() and 1 <= int(message_body) <= 6:
        # Handle style selection - process immediately without payment
        style_mapping = list(ImageStyle)
        selected_style = style_mapping[int(message_body) - 1]
        
        # Check if user has pending image
        pending_image_url = redis_client.get(f"pending_image:{user.id}")
        if not pending_image_url:
            resp.message("No pending image found. Please send a photo with /Chitrify command first.")
            return str(resp)
        
        pending_image_url = pending_image_url.decode()
        
        # Process immediately (free for testing)
        render = ImageRender(
            user_id=user.id,
            original_image_url=pending_image_url,
            style=selected_style
        )
        db.add(render)
        db.commit()
        
        # Queue processing
        process_image_render.delay(render.id)
        
        resp.message(f"🎉 Processing your {STYLE_DESCRIPTIONS[selected_style]} for FREE! This usually takes under 60 seconds.")
        
        # Clean up pending image
        redis_client.delete(f"pending_image:{user.id}")
    
    else:
        # Default help message
        help_text = """Welcome to Chitrify AI! 🎨

Send any photo with the command /Chitrify to transform it into amazing styles:

🎬 3D Cartoon (Pixar-style)
🎨 Oil-paint Portrait  
🎵 Rotoscope Outline
📺 90s Anime Cel
📸 Vintage Colourised
🖌️ Water-colour Sketch

💰 Just ₹19 per image (or $1 in US)
🎁 Get 1 free render when you refer a friend!

Try it now: Send a photo with /Chitrify"""
        
        resp.message(help_text)
    
    return Response(str(resp), media_type="application/xml")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get basic stats for monitoring"""
    total_users = db.query(User).count()
    total_renders = db.query(ImageRender).count()
    completed_renders = db.query(ImageRender).filter(ImageRender.status == RenderStatus.COMPLETED).count()
    
    return {
        "total_users": total_users,
        "total_renders": total_renders,
        "completed_renders": completed_renders,
        "success_rate": completed_renders / total_renders if total_renders > 0 else 0,
        "mode": "FREE_TESTING"
    }

from fastapi.staticfiles import StaticFiles
if not USE_S3:
    os.makedirs("uploads", exist_ok=True)
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)