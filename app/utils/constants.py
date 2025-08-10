from enum import Enum

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
    ImageStyle.CARTOON_3D: "🎬 Studio Ghibli Animation",
    ImageStyle.OIL_PAINT: "🎨 Renaissance Oil Painting",
    ImageStyle.ROTOSCOPE: "🎵 Rotoscope Movie Effect", 
    ImageStyle.ANIME_90S: "📺 Classic 90s Anime",
    ImageStyle.VINTAGE_COLORIZED: "📸 Vintage Photograph",
    ImageStyle.WATERCOLOR: "🖌️ Watercolor Painting"
}

STYLE_PROMPTS = {
    ImageStyle.CARTOON_3D: "Transform this image into a vibrant 3D Pixar-style cartoon with smooth lighting, rounded features, and bright saturated colors. Make it look like a professional animated movie character.",
    ImageStyle.OIL_PAINT: "Convert this image into a classical oil painting with visible brush strokes, rich textures, and artistic lighting. Use the style of Renaissance masters with deep, warm colors.",
    ImageStyle.ROTOSCOPE: "Transform this image into a rotoscoped animation style with bold outlines, flat colors, and traced contours. Make it look like A Scanner Darkly or Waking Life animation.",
    ImageStyle.ANIME_90S: "Convert this image into 90s anime cel animation style with hand-drawn appearance, large expressive eyes, vibrant colors, and clean line art typical of classic Japanese animation.",
    ImageStyle.VINTAGE_COLORIZED: "Transform this image into a vintage colorized photograph from the 1940s-1950s with sepia undertones, soft focus, aged paper texture, and muted color palette.",
    ImageStyle.WATERCOLOR: "Convert this image into a delicate watercolor painting with soft flowing colors, transparent washes, wet-on-wet bleeding effects, and artistic paper texture."
}
