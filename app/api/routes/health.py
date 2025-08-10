from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.models.database import User, ImageRender
from app.utils.constants import RenderStatus

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get basic stats for monitoring"""
    total_users = db.query(User).count()
    total_renders = db.query(ImageRender).count()
    completed_renders = db.query(ImageRender).filter(
        ImageRender.status == RenderStatus.COMPLETED
    ).count()
    
    return {
        "total_users": total_users,
        "total_renders": total_renders,
        "completed_renders": completed_renders,
        "success_rate": completed_renders / total_renders if total_renders > 0 else 0,
        "mode": "FREE_TESTING"
    }
