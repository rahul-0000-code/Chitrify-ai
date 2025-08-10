import logging
import uuid
from fastapi import APIRouter, Depends, Response, Request, BackgroundTasks
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_or_create_user
from app.models.database import ImageRender
from app.models.schemas import WhatsAppMessage
from app.services.messaging.twilio_service import TwilioService
from app.services.image.processor import ImageProcessor
from app.utils.constants import ImageStyle, RenderStatus, STYLE_DESCRIPTIONS
from app.utils.helpers import (
    extract_image_urls_from_message, 
    get_direct_image_url, 
    download_image_from_url,
    is_safe_content
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
messaging_service = TwilioService()
image_processor = ImageProcessor()

async def process_image_background(render_id: str, image_data: bytes):
    """Background task for processing images"""
    from app.core.dependencies import get_db
    db = next(get_db())
    
    try:
        render = db.query(ImageRender).filter(ImageRender.id == render_id).first()
        if not render:
            logger.error(f"Render {render_id} not found")
            return
        
        render.status = RenderStatus.PROCESSING
        db.commit()
        
        # Process image
        processed_image = await image_processor.process_image(image_data, render.style)
        
        if not processed_image:
            raise Exception("Image processing failed")
        
        # Upload processed image
        filename = f"chitrify_{render.style}_{render_id[:8]}.jpg"
        rendered_url = await image_processor.upload_image(processed_image, filename)
        
        if not rendered_url:
            raise Exception("Failed to upload processed image")
        
        # Update render
        render.rendered_image_url = rendered_url
        render.status = RenderStatus.COMPLETED
        db.commit()
        
        # Send result to user
        style_names = {
            ImageStyle.CARTOON_3D: "Studio Ghibli Animation",
            ImageStyle.OIL_PAINT: "Renaissance Oil Painting", 
            ImageStyle.ROTOSCOPE: "Rotoscope Movie Effect",
            ImageStyle.ANIME_90S: "Classic 90s Anime",
            ImageStyle.VINTAGE_COLORIZED: "Vintage Photograph",
            ImageStyle.WATERCOLOR: "Watercolor Painting"
        }
        
        message = f"""🎉 **Your {style_names[render.style]} is ready!**

✨ **Processing completed successfully!**

🖼️ **View your transformed image:**
{rendered_url}

💡 **Want to try another style?** Send another ImgBB link and choose a different number (1-6)!

🎨 **Chitrify AI** - Transforming memories into art!"""
        
        await messaging_service.send_message(render.user.whatsapp_number, message)
        
    except Exception as e:
        logger.error(f"Processing failed for render {render_id}: {e}")
        
        render.status = RenderStatus.FAILED
        render.error_message = str(e)
        db.commit()
        
        error_message = """😔 **Image processing encountered an error.**

🔄 **Please try:**
1. Send your ImgBB link again
2. Choose a different style (1-6)
3. Or try with a different image

💬 **Need help?** The error has been logged for our team to investigate."""
        
        await messaging_service.send_message(render.user.whatsapp_number, error_message)

@router.post("/whatsapp")
async def whatsapp_webhook(
    request: Request, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    """Handle incoming WhatsApp messages"""
    form_data = await request.form()
    
    from_number = form_data.get("From")
    message_body = form_data.get("Body", "").strip()
    media_url = form_data.get("MediaUrl0")
    media_type = form_data.get("MediaContentType0")
    
    user = get_or_create_user(from_number, db)
    resp = messaging_service.create_response()
    
    # Handle media messages (Twilio direct upload)
    if media_url and media_type and media_type.startswith('image/'):
        # Handle Twilio media (will fail in sandbox, but keep for production)
        fallback_message = """📱 **Unable to access your image directly!**

🔄 **Please use this workaround:**

1️⃣ Go to [[**https://imgbb.com**](https://imgbb.com)](https://imgbb.com)
2️⃣ Upload your image (drag & drop)
3️⃣ Copy the link they give you
4️⃣ Send that link here

**Example:** https://ibb.co/abc123

This is needed for our testing environment. Full version won't need this! 🚀"""
        
        resp.message(fallback_message)
        return Response(str(resp), media_type="application/xml")
    
    # Handle text messages
    elif message_body:
        urls = extract_image_urls_from_message(message_body)
        
        # Process image URLs
        if urls:
            image_url = urls[0]
            logger.info(f"Image URL received from {from_number}: {image_url}")
            
            try:
                direct_url = get_direct_image_url(image_url)
                if not direct_url:
                    resp.message("❌ Couldn't process your image URL. Please try uploading to https://imgbb.com and send the link.")
                    return Response(str(resp), media_type="application/xml")
                
                image_content = download_image_from_url(direct_url)
                if not image_content:
                    resp.message("❌ Couldn't download your image. Please check the URL or try a different image hosting service.")
                    return Response(str(resp), media_type="application/xml")
                
            except Exception as e:
                logger.error(f"Error processing image URL {image_url}: {e}")
                resp.message("❌ Error processing your image URL. Please try again or use a different link.")
                return Response(str(resp), media_type="application/xml")
        
        # Handle style selection
        elif message_body.isdigit() and 1 <= int(message_body) <= 6:
            style_mapping = [
                ImageStyle.CARTOON_3D,
                ImageStyle.OIL_PAINT,
                ImageStyle.ROTOSCOPE,
                ImageStyle.ANIME_90S,
                ImageStyle.VINTAGE_COLORIZED,
                ImageStyle.WATERCOLOR
            ]
            
            selected_style = style_mapping[int(message_body) - 1]
            
            pending_render = db.query(ImageRender).filter(
                ImageRender.user_id == user.id,
                ImageRender.status == RenderStatus.QUEUED
            ).first()
            
            if not pending_render:
                resp.message("❌ No pending image found.\n\n📋 **How to get started:**\n1️⃣ Upload image to https://imgbb.com\n2️⃣ Send the link here\n3️⃣ Choose style (1-6)")
                return Response(str(resp), media_type="application/xml")
            
            pending_render.style = selected_style
            db.commit()
            
            # Get original image data for processing
            try:
                original_image_data = download_image_from_url(pending_render.original_image_url)
                if not original_image_data:
                    raise Exception("Unable to obtain original image data")
                
                background_tasks.add_task(
                    process_image_background, 
                    str(pending_render.id), 
                    original_image_data
                )
                
                style_names = {
                    ImageStyle.CARTOON_3D: "Studio Ghibli style",
                    ImageStyle.OIL_PAINT: "Oil Painting", 
                    ImageStyle.ROTOSCOPE: "Rotoscope Animation",
                    ImageStyle.ANIME_90S: "90s Anime",
                    ImageStyle.VINTAGE_COLORIZED: "Vintage Photo",
                    ImageStyle.WATERCOLOR: "Watercolor Painting"
                }
                
                resp.message(f"🎨 **Creating your {style_names[selected_style]}...**\n\n⏱️ Processing time: ~30-60 seconds\n✨ Your transformed image will arrive shortly!")
                
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                pending_render.status = RenderStatus.FAILED
                pending_render.error_message = str(e)
                db.commit()
                
                resp.message("❌ Error accessing your image data. Please send the ImgBB link again and try selecting a style.")
            
            return Response(str(resp), media_type="application/xml")
        
        else:
            # Welcome message
            help_text = """🎨 **Welcome to Chitrify AI!**

Transform any photo into amazing artistic styles:

🎬 Studio Ghibli animations
🎨 Classical oil paintings  
🎵 Rotoscope movie effects
📺 90s anime characters
📸 Vintage photographs
🖌️ Watercolor artworks

**📋 How to use:**

1️⃣ Upload your image to [[**https://imgbb.com**](https://imgbb.com)](https://imgbb.com)
2️⃣ Copy the link (like: https://ibb.co/abc123)
3️⃣ Send that link here
4️⃣ Choose your style (1-6)
5️⃣ Get your transformed image!

💡 **Just send an ImgBB link to get started!**

*Note: This workaround is needed for our testing environment. Full version will accept images directly!*"""
            
            resp.message(help_text)
            return Response(str(resp), media_type="application/xml")
    
    # Process image if we have image_content
    if 'image_content' in locals():
        try:
            # Validate image
            if len(image_content) > 10 * 1024 * 1024:
                resp.message("Image too large. Please use an image smaller than 10MB.")
                return Response(str(resp), media_type="application/xml")
            
            if not is_safe_content(image_content):
                resp.message("Sorry, I can't process this image. Please try a different one.")
                return Response(str(resp), media_type="application/xml")
            
            # Store the image
            filename = f"original_{uuid.uuid4()}.jpg"
            original_url = await image_processor.upload_image(image_content, filename)
            
            if not original_url:
                resp.message("Sorry, failed to store your image. Please try again.")
                return Response(str(resp), media_type="application/xml")
            
            # Show style selection
            styles_text = "✨ **Choose your transformation style:**\n\n"
            styles_text += "1️⃣ 🎬 **Studio Ghibli** - Magical anime movie style\n"
            styles_text += "2️⃣ 🎨 **Oil Painting** - Classical artistic portrait\n"
            styles_text += "3️⃣ 🎵 **Rotoscope** - Animated movie outline style\n"
            styles_text += "4️⃣ 📺 **90s Anime** - Classic anime character\n"
            styles_text += "5️⃣ 📸 **Vintage Photo** - Retro colorized look\n"
            styles_text += "6️⃣ 🖌️ **Watercolor** - Soft painting effect\n\n"
            styles_text += "🎉 **FREE PREVIEW MODE** - Try any style!\n"
            styles_text += "📱 Reply with a number (1-6) to transform your image!"
            
            # Create pending render
            pending_render = ImageRender(
                user_id=user.id,
                original_image_url=original_url,
                style=ImageStyle.CARTOON_3D,  # Placeholder
                status=RenderStatus.QUEUED
            )
            db.add(pending_render)
            db.commit()
            
            # Clean up old pending renders
            db.query(ImageRender).filter(
                ImageRender.user_id == user.id,
                ImageRender.status == RenderStatus.QUEUED,
                ImageRender.id != pending_render.id
            ).delete()
            db.commit()
            
            resp.message(styles_text)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            resp.message("Sorry, there was an error processing your image. Please try again.")
    
    return Response(str(resp), media_type="application/xml")
