import os
import logging
import uuid
import google.generativeai as genai
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum
from io import BytesIO
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi import FastAPI, Depends, Response, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, ENUM
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from PIL import ImageDraw

import redis
from celery import Celery
import requests
from PIL import Image
import io
import boto3
from dotenv import load_dotenv
from PIL import ImageFilter, ImageEnhance

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

# Database setup - PostgreSQL optimized
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost/chitrify")
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False
)
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

GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully")
else:
    logger.warning("Gemini API key not found - image processing will use fallback")

# External service clients
twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

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

# Create PostgreSQL ENUMs
image_style_enum = ENUM(ImageStyle, name='image_style_enum', create_type=False)
payment_status_enum = ENUM(PaymentStatus, name='payment_status_enum', create_type=False)
render_status_enum = ENUM(RenderStatus, name='render_status_enum', create_type=False)

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

# Enhanced style prompts for Gemini
STYLE_PROMPTS = {
    ImageStyle.CARTOON_3D: "Transform this image into a vibrant 3D Pixar-style cartoon with smooth lighting, rounded features, and bright saturated colors. Make it look like a professional animated movie character.",
    ImageStyle.OIL_PAINT: "Convert this image into a classical oil painting with visible brush strokes, rich textures, and artistic lighting. Use the style of Renaissance masters with deep, warm colors.",
    ImageStyle.ROTOSCOPE: "Transform this image into a rotoscoped animation style with bold outlines, flat colors, and traced contours. Make it look like A Scanner Darkly or Waking Life animation.",
    ImageStyle.ANIME_90S: "Convert this image into 90s anime cel animation style with hand-drawn appearance, large expressive eyes, vibrant colors, and clean line art typical of classic Japanese animation.",
    ImageStyle.VINTAGE_COLORIZED: "Transform this image into a vintage colorized photograph from the 1940s-1950s with sepia undertones, soft focus, aged paper texture, and muted color palette.",
    ImageStyle.WATERCOLOR: "Convert this image into a delicate watercolor painting with soft flowing colors, transparent washes, wet-on-wet bleeding effects, and artistic paper texture."
}


# Database Models - PostgreSQL optimized
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    whatsapp_number = Column(String(20), unique=True, index=True, nullable=False)
    country_code = Column(String(3), default="US")
    free_credits = Column(Integer, default=0)
    total_referrals = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    orders = relationship("Order", back_populates="user")
    renders = relationship("ImageRender", back_populates="user")

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)
    payment_status = Column(payment_status_enum, default=PaymentStatus.PENDING, index=True)
    payment_gateway = Column(String(20))  # stripe or razorpay
    gateway_payment_id = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    user = relationship("User", back_populates="orders")
    renders = relationship("ImageRender", back_populates="order")

class ImageRender(Base):
    __tablename__ = "image_renders"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"), nullable=True, index=True)
    original_image_url = Column(Text, nullable=False)
    rendered_image_url = Column(Text, nullable=True)
    style = Column(image_style_enum, nullable=False, index=True)
    status = Column(render_status_enum, default=RenderStatus.QUEUED, index=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), default=lambda: datetime.utcnow() + timedelta(days=30), index=True)
    
    user = relationship("User", back_populates="renders")
    order = relationship("Order", back_populates="renders")

class Referral(Base):
    __tablename__ = "referrals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    referrer_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    referee_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    completed = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)

# Create ENUM types first, then tables
def create_enums_and_tables():
    try:
        # Create ENUMs if they don't exist
        image_style_enum.create(engine, checkfirst=True)
        payment_status_enum.create(engine, checkfirst=True) 
        render_status_enum.create(engine, checkfirst=True)
    except Exception as e:
        logger.info(f"ENUMs may already exist: {e}")
    
    # Create tables
    Base.metadata.create_all(bind=engine)

create_enums_and_tables()

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

def process_with_ai_model(image_content: bytes, style: ImageStyle) -> bytes:
    """Process image with Google Gemini AI model"""
    try:
        if not GEMINI_API_KEY:
            logger.warning("Gemini API not configured, using fallback processing")
            return fallback_image_processing(image_content, style)
        
        # Convert image to PIL Image for processing
        original_img = Image.open(BytesIO(image_content))
        
        # Resize if too large (Gemini has size limits)
        max_size = 1024
        if original_img.width > max_size or original_img.height > max_size:
            original_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        # Convert to bytes for Gemini
        img_byte_arr = BytesIO()
        original_img.save(img_byte_arr, format='JPEG', quality=85)
        processed_image_bytes = img_byte_arr.getvalue()
        
        # Get style prompt
        style_prompt = STYLE_PROMPTS.get(style, "Transform this image artistically")
        
        # Use Gemini Pro Vision for image transformation
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Create the prompt for image transformation
        full_prompt = f"""
        {style_prompt}
        
        Please analyze this image and describe how it would look when transformed into the requested style. 
        Focus on colors, lighting, textures, and artistic elements that would be present in the final result.
        Keep the same composition and subject matter but completely change the artistic style.
        """
        
        # Note: Gemini doesn't directly generate images, so we'll use it for enhanced prompting
        # and fallback to a hybrid approach
        response = model.generate_content([full_prompt, original_img])
        
        logger.info(f"Gemini analysis: {response.text[:100]}...")
        
        # For now, apply enhanced processing based on Gemini's analysis
        return enhanced_image_processing(processed_image_bytes, style, response.text)
        
    except Exception as e:
        logger.error(f"Gemini processing failed: {e}")
        return fallback_image_processing(image_content, style)


def enhanced_image_processing(image_content: bytes, style: ImageStyle, ai_analysis: str) -> bytes:
    """Enhanced image processing with AI insights"""
    try:
        img = Image.open(BytesIO(image_content))
        
        # Apply style-specific transformations
        if style == ImageStyle.CARTOON_3D:
            # Enhanced cartoon processing
            img = apply_cartoon_effect(img)
        elif style == ImageStyle.OIL_PAINT:
            # Enhanced oil paint effect
            img = apply_oil_paint_effect(img)
        elif style == ImageStyle.ANIME_90S:
            # Enhanced anime effect
            img = apply_anime_effect(img)
        elif style == ImageStyle.WATERCOLOR:
            # Enhanced watercolor effect
            img = apply_watercolor_effect(img)
        elif style == ImageStyle.VINTAGE_COLORIZED:
            # Enhanced vintage effect
            img = apply_vintage_effect(img)
        elif style == ImageStyle.ROTOSCOPE:
            # Enhanced rotoscope effect
            img = apply_rotoscope_effect(img)
        
        # Add style-specific watermark
        img = add_style_watermark(img, style)
        
        # Convert back to bytes
        output = BytesIO()
        img.save(output, format='JPEG', quality=90)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Enhanced processing failed: {e}")
        return fallback_image_processing(image_content, style)


def apply_cartoon_effect(img: Image.Image) -> Image.Image:
    """Apply 3D cartoon-like effect"""
    
    
    # Enhance colors and contrast
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.3)
    
    # Add slight blur for smoothness
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Enhance brightness slightly
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    
    return img


def apply_oil_paint_effect(img: Image.Image) -> Image.Image:
    """Apply oil painting effect"""
    # Apply edge enhancement
    img = img.filter(ImageFilter.EDGE_ENHANCE)
    
    # Add texture with slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    
    return img


def apply_anime_effect(img: Image.Image) -> Image.Image:
    """Apply anime-style effect"""
    
    # Increase saturation for vibrant anime colors
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.4)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    return img


def apply_watercolor_effect(img: Image.Image) -> Image.Image:
    """Apply watercolor painting effect"""
    # Soften the image
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # Reduce saturation slightly
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(0.8)
    
    return img


def apply_vintage_effect(img: Image.Image) -> Image.Image:
    """Apply vintage colorized effect"""
    # Add sepia-like tint
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(0.7)
    
    # Reduce brightness slightly
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.9)
    
    return img


def apply_rotoscope_effect(img: Image.Image) -> Image.Image:
    """Apply rotoscope animation effect"""
    # Enhance edges
    img = img.filter(ImageFilter.FIND_EDGES)
    
    # Increase contrast for bold lines
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    return img


def add_style_watermark(img: Image.Image, style: ImageStyle) -> Image.Image:
    """Add style-specific watermark"""
    
    draw = ImageDraw.Draw(img)
    
    # Style-specific watermark text
    watermark_text = f"Chitrify AI - {STYLE_DESCRIPTIONS[style]}"
    
    # Position watermark at bottom right
    text_width = len(watermark_text) * 8  # Approximate text width
    x = img.width - text_width - 10
    y = img.height - 25
    
    # Add text with semi-transparent background
    try:
        draw.text((x, y), watermark_text, fill="white", stroke_width=1, stroke_fill="black")
    except:
        # Fallback if font issues
        draw.text((x, y), watermark_text, fill="white")
    
    return img


def fallback_image_processing(image_content: bytes, style: ImageStyle) -> bytes:
    """Fallback processing when Gemini is not available"""
    try:
        img = Image.open(BytesIO(image_content))
        
        # Apply basic style transformation
        if style == ImageStyle.CARTOON_3D:
            img = apply_cartoon_effect(img)
        # ... other styles
        
        # Add basic watermark
        img = add_style_watermark(img, style)
        
        output = BytesIO()
        img.save(output, format='JPEG', quality=85)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Fallback processing failed: {e}")
        # Return original image as last resort
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
            
            # Show style selection
            styles_text = "Choose your style:\n\n"
            for i, (style, desc) in enumerate(STYLE_DESCRIPTIONS.items(), 1):
                styles_text += f"{i}️⃣ {desc}\n"
            
            styles_text += "\n🎉 FREE TESTING MODE - All styles are free!"
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
        process_image_render.delay(str(render.id))
        
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

if not USE_S3:
    os.makedirs("uploads", exist_ok=True)
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)