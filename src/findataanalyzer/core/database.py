"""Database connection and session management using SQLAlchemy."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from findataanalyzer.core.config import settings

# Create the SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URI,
    pool_pre_ping=True
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for declarative models
Base = declarative_base() 