"""
Firebase Manager for Trading System
Handles all database operations, state management, and real-time logging.
Architectural Choice: Using Firestore for its flexibility, real-time capabilities,
and scalability compared to Realtime Database for trading data.
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
from dataclasses import dataclass, asdict

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud.firestore_v1.client import Client
except ImportError:
    logging.error("firebase-admin not installed. Install via: pip install firebase-admin")
    raise

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Data class for trade records to ensure type safety."""
    trade_id: str
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    price: float
    quantity: float
    timestamp: datetime
    portfolio_value: float
    reason: str
    metadata: Optional[Dict[str, Any]] = None

class FirebaseManager:
    """Manages all Firebase Firestore operations for the trading system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Firebase connection.
        
        Args:
            config_path: Path to Firebase service account JSON file.
                        If None, uses default credentials.
        """
        self._initialized = False
        self._client: Optional[Client] = None
        self.config_path = config_path
        self._initialize_firebase()
        
    def _initialize_firebase(self) -> None:
        """Initialize Firebase Admin SDK with error handling."""
        try:
            if not firebase_admin._apps:
                if self.config_path:
                    cred = credentials.Certificate(self.config_path)
                else:
                    cred = credentials.ApplicationDefault()
                    
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized successfully")
            
            self._client = firestore.client()
            self._initialized = True
            logger.info("Firestore client connected")
            
        except FileNotFoundError as e:
            logger.error(f"Firebase config file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            # Fallback to local logging if Firebase fails
            logger.warning("Using local logging fallback")
            self._initialized = False
    
    def log_trade(self, trade: TradeRecord) -> bool:
        """
        Log a trade to Firestore with comprehensive error handling.
        
        Args:
            trade: TradeRecord object containing trade details
            
        Returns:
            bool: Success status
        """
        if not self._initialized or self._client is None:
            logger.error("Firebase not initialized. Cannot log trade.")
            return False
        
        try:
            trade_dict = asdict(trade)
            # Convert datetime to string for Firestore
            trade_dict['timestamp'] = trade.timestamp.isoformat()
            
            # Add document with trade_id as document ID for easy retrieval
            doc_ref = self._client.collection("trading_trades").document(trade.trade_id)
            doc_ref.set(trade_dict)
            
            logger.info(f"Trade logged successfully: {trade.trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log trade {trade.trade_id}: {e}")
            # Emergency fallback: Log to local file
            self._emergency_local_log(trade_dict)
            return False
    
    def save_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """Save trained model metadata to Firestore."""
        if not self._initialized:
            return False
        
        try:
            metadata['last_updated'] = datetime.utcnow()
            self._client.collection("trained_models").document(model_id).set(metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")
            return False
    
    def get_performance_metrics(self, model_id: str, limit: int = 100) -> List[Dict]:
        """Retrieve performance metrics for a model."""
        if not self._initialized:
            return []
        
        try:
            docs = (self._client.collection("performance_metrics")
                    .where("model_id", "==", model_id)
                    .order_by("timestamp", direction=firestore.Query.DESCENDING)
                    .limit(limit)
                    .stream())
            
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return []
    
    def _emergency_local_log(self, data: Dict[str, Any]) -> None:
        """Emergency local logging when Firebase fails."""
        try:
            with open("emergency_trades.log", "a") as f:
                f.write(json.dumps(data, default=str) + "\n")
            logger.warning("Trade logged to emergency local file")
        except Exception as e:
            logger.critical(f"Failed to write emergency log: {e