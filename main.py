
import os
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
import stripe
import razorpay
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
from fastapi.responses import Response
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
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
razorpay_client = razorpay.Client(
    auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET"))
)
posthog.api_key = os.getenv("POSTHOG_API_KEY")
posthog.host = os.getenv("POSTHOG_HOST", "https://app.posthog.com")

# AWS S3 setup
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

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

PRICING = {
    "US": {"single": 1.00, "pack": 8.00},
    "IN": {"single": 19.0, "pack": 149.0}
}

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
    """Upload file to S3 and return URL"""
    key = f"images/{datetime.utcnow().strftime('%Y/%m/%d')}/{filename}"
    s3_client.put_object(
        Bucket=AWS_BUCKET_NAME,
        Key=key,
        Body=file_content,
        ContentType='image/jpeg'
    )
    return f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{key}"

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
        
        # Refund if payment was made
        if render.order_id:
            # Implement refund logic here
            pass
        
        # Notify user
        send_whatsapp_message(
            render.user.whatsapp_number,
            "Sorry, there was an issue processing your image. Please try again or contact support."
        )
        
        logger.error(f"Image processing failed for {render_id}: {e}")
    
    finally:
        db.close()

def process_with_ai_model(image_content: bytes, style: str) -> bytes:
    """Process image with AI model - replace with your preferred AI service"""
    # Example using Replicate API
    # You can replace this with Stability AI, OpenAI DALL-E, or local models
    
    # For POC, returning original image with a simple filter
    # In production, integrate with actual AI services
    img = Image.open(io.BytesIO(image_content))
    
    # Apply simple processing based on style (for demo)
    if style == ImageStyle.CARTOON_3D:
        # Placeholder processing
        img = img.convert('RGB')
    
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=85)
    return output.getvalue()

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
            
            pricing = PRICING[user.country_code]
            currency = "₹" if user.country_code == "IN" else "$"
            
            styles_text += f"\n💰 Price: {currency}{pricing['single']} per image"
            styles_text += f"\n🎁 Pack of 10: {currency}{pricing['pack']}"
            
            if user.free_credits > 0:
                styles_text += f"\n✨ You have {user.free_credits} free credits!"
            
            styles_text += "\n\nReply with the number (1-6) to continue."
            
            # Store image URL in Redis for this user
            redis_client.setex(f"pending_image:{user.id}", 300, original_url)  # 5 min expiry
            
            resp.message(styles_text)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            resp.message("Sorry, there was an error processing your image. Please try again.")
    
    elif message_body.isdigit() and 1 <= int(message_body) <= 6:
        # Handle style selection
        style_mapping = list(ImageStyle)
        selected_style = style_mapping[int(message_body) - 1]
        
        # Check if user has pending image
        pending_image_url = redis_client.get(f"pending_image:{user.id}")
        if not pending_image_url:
            resp.message("No pending image found. Please send a photo with /Chitrify command first.")
            return str(resp)
        
        pending_image_url = pending_image_url.decode()
        
        # Check if user has free credits
        if user.free_credits > 0:
            # Process immediately with free credit
            user.free_credits -= 1
            db.commit()
            
            render = ImageRender(
                user_id=user.id,
                original_image_url=pending_image_url,
                style=selected_style
            )
            db.add(render)
            db.commit()
            
            # Queue processing
            process_image_render.delay(render.id)
            
            resp.message(f"🎉 Using your free credit! Processing your {STYLE_DESCRIPTIONS[selected_style]}. This usually takes under 60 seconds.")
            
        else:
            # Create payment links
            pricing = PRICING[user.country_code]
            
            if user.country_code == "IN":
                # Create Razorpay payment link
                payment_data = {
                    "amount": int(pricing["single"] * 100),  # Razorpay uses paisa
                    "currency": "INR",
                    "description": f"Chitrify - {STYLE_DESCRIPTIONS[selected_style]}",
                    "customer": {
                        "contact": user.whatsapp_number.replace("whatsapp:", "")
                    }
                }
                
                # Store payment intent
                redis_client.setex(
                    f"payment_intent:{user.id}",
                    600,  # 10 min expiry
                    json.dumps({
                        "style": selected_style,
                        "image_url": pending_image_url,
                        "amount": pricing["single"]
                    })
                )
                
                payment_url = f"{os.getenv('WEBHOOK_BASE_URL')}/payment/razorpay?user_id={user.id}"
                
            else:
                # Create Stripe payment link
                payment_url = f"{os.getenv('WEBHOOK_BASE_URL')}/payment/stripe?user_id={user.id}"
            
            resp.message(f"💳 Pay {PRICING[user.country_code]['single']} to process your {STYLE_DESCRIPTIONS[selected_style]}\n\n🔗 {payment_url}")
    
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
    
    # return str(resp)
    return Response(str(resp), media_type="application/xml")


@app.get("/payment/stripe")
async def stripe_payment(user_id: str, db: Session = Depends(get_db)):
    """Create Stripe payment session"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    payment_intent_data = redis_client.get(f"payment_intent:{user_id}")
    if not payment_intent_data:
        raise HTTPException(status_code=400, detail="Payment intent expired")
    
    intent_data = json.loads(payment_intent_data)
    
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': f"Chitrify - {STYLE_DESCRIPTIONS[intent_data['style']]}",
                    },
                    'unit_amount': int(intent_data['amount'] * 100),
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f"{os.getenv('WEBHOOK_BASE_URL')}/payment/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{os.getenv('WEBHOOK_BASE_URL')}/payment/cancel",
            metadata={
                'user_id': user_id,
                'style': intent_data['style'],
                'image_url': intent_data['image_url']
            }
        )
        
        return {"url": session.url}
    
    except Exception as e:
        logger.error(f"Stripe payment creation failed: {e}")
        raise HTTPException(status_code=500, detail="Payment creation failed")

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Handle Stripe webhooks"""
    payload = await request.body()
    
    try:
        event = stripe.Event.construct_from(
            json.loads(payload), stripe.api_key
        )
        
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            user_id = session['metadata']['user_id']
            style = session['metadata']['style']
            image_url = session['metadata']['image_url']
            
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                # Create order
                order = Order(
                    user_id=user_id,
                    amount=session['amount_total'] / 100,
                    currency=session['currency'].upper(),
                    payment_status=PaymentStatus.COMPLETED,
                    payment_gateway="stripe",
                    gateway_payment_id=session['id']
                )
                db.add(order)
                db.commit()
                
                # Create render
                render = ImageRender(
                    user_id=user_id,
                    order_id=order.id,
                    original_image_url=image_url,
                    style=style
                )
                db.add(render)
                db.commit()
                
                # Queue processing
                process_image_render.delay(render.id)
                
                # Track payment
                posthog.capture(user_id, 'paid', {
                    'amount': order.amount,
                    'currency': order.currency,
                    'gateway': 'stripe'
                })
                
                # Notify user
                send_whatsapp_message(
                    user.whatsapp_number,
                    f"✅ Payment confirmed! Processing your {STYLE_DESCRIPTIONS[style]}. This usually takes under 60 seconds."
                )
        
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Stripe webhook error: {e}")
        return {"status": "error"}

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
        "success_rate": completed_renders / total_renders if total_renders > 0 else 0
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)