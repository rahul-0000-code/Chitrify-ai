import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, ENUM

from app.config.settings import settings
from app.utils.constants import ImageStyle, PaymentStatus, RenderStatus

# Database setup
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Create PostgreSQL ENUMs
image_style_enum = ENUM(ImageStyle, name='image_style_enum', create_type=False)
payment_status_enum = ENUM(PaymentStatus, name='payment_status_enum', create_type=False)
render_status_enum = ENUM(RenderStatus, name='render_status_enum', create_type=False)

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    whatsapp_number = Column(String(50), unique=True, index=True, nullable=False)
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
    payment_gateway = Column(String(20))
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

def create_enums_and_tables():
    """Create database enums and tables"""
    try:
        image_style_enum.create(engine, checkfirst=True)
        payment_status_enum.create(engine, checkfirst=True) 
        render_status_enum.create(engine, checkfirst=True)
    except Exception as e:
        pass  # ENUMs may already exist
    
    Base.metadata.create_all(bind=engine)
