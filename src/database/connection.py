"""
Database connection management for MLOps pipeline.

This module provides database engine creation, session management, and connection pooling.
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional
from pathlib import Path

from sqlalchemy import create_engine, Engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.exc import SQLAlchemyError

try:
    from .models import Base
    from ..utils.logging_config import get_logger
except ImportError:
    # For direct script execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from database.models import Base
    from utils.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection string
            echo: Whether to echo SQL statements
        """
        self.database_url = database_url
        self.echo = echo
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        
    def initialize(self) -> None:
        """Initialize database engine and session factory."""
        try:
            # Create engine with appropriate configuration
            if self.database_url.startswith('sqlite'):
                # SQLite configuration
                self.engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 30,
                        "isolation_level": None  # Enables autocommit mode
                    }
                )
                
                # Enable WAL mode for better concurrency
                @event.listens_for(self.engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA cache_size=10000")
                    cursor.execute("PRAGMA temp_store=MEMORY")
                    cursor.close()
                    
            else:
                # PostgreSQL/MySQL configuration
                self.engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    poolclass=QueuePool,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Ensure database directory exists for SQLite
            if self.database_url.startswith('sqlite'):
                db_path = self.database_url.replace('sqlite:///', '')
                db_dir = Path(db_path).parent
                db_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Database engine initialized: {self._get_db_info()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            if not self.engine:
                raise RuntimeError("Database not initialized")
                
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self) -> None:
        """Drop all database tables."""
        try:
            if not self.engine:
                raise RuntimeError("Database not initialized")
                
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
            
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session with automatic cleanup.
        
        Yields:
            SQLAlchemy session
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
            
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session_sync(self) -> Session:
        """
        Get database session for dependency injection.
        
        Returns:
            SQLAlchemy session
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()
    
    def health_check(self) -> bool:
        """
        Check database connectivity.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """
        Get database connection information.
        
        Returns:
            Dictionary with connection details
        """
        if not self.engine:
            return {"status": "not_initialized"}
            
        return {
            "database_url": self._get_db_info(),
            "pool_size": getattr(self.engine.pool, 'size', None),
            "checked_out": getattr(self.engine.pool, 'checkedout', None),
            "overflow": getattr(self.engine.pool, 'overflow', None),
            "checked_in": getattr(self.engine.pool, 'checkedin', None),
        }
    
    def _get_db_info(self) -> str:
        """Get sanitized database URL for logging."""
        if not self.database_url:
            return "unknown"
            
        # Hide password in URL
        if '@' in self.database_url:
            parts = self.database_url.split('@')
            if len(parts) == 2:
                scheme_user = parts[0].split('://')
                if len(scheme_user) == 2:
                    scheme = scheme_user[0]
                    user = scheme_user[1].split(':')[0] if ':' in scheme_user[1] else scheme_user[1]
                    return f"{scheme}://{user}:***@{parts[1]}"
        
        return self.database_url


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def init_database(database_url: str, echo: bool = False) -> DatabaseManager:
    """
    Initialize global database manager.
    
    Args:
        database_url: Database connection string
        echo: Whether to echo SQL statements
        
    Returns:
        Initialized database manager
    """
    global db_manager
    
    db_manager = DatabaseManager(database_url, echo)
    db_manager.initialize()
    db_manager.create_tables()
    
    return db_manager


def get_db_session() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.
    
    Yields:
        SQLAlchemy session
    """
    if not db_manager:
        raise RuntimeError("Database not initialized")
        
    session = db_manager.get_session_sync()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        Database manager instance
    """
    if not db_manager:
        raise RuntimeError("Database not initialized")
    return db_manager 