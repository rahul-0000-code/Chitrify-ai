from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.models.database import User

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_or_create_user(whatsapp_number: str, db: Session) -> User:
    """Get or create user"""
    user = db.query(User).filter(User.whatsapp_number == whatsapp_number).first()
    if not user:
        country_code = "IN" if whatsapp_number.startswith("whatsapp:+91") else "US"
        user = User(whatsapp_number=whatsapp_number, country_code=country_code)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user
