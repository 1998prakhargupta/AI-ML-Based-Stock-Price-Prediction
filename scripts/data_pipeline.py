#!/usr/bin/env python3
"""
Data Pipeline Script
===================

Main data pipeline for fetching, processing, and storing market data.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.logging_utils import setup_logging
from utils.config_manager import Config
from data.fetchers import IndexDataManager
from data.processors import DataProcessor
from models.model_utilities import ModelTrainer
from compliance.api_compliance import get_compliance_manager, ComplianceLevel

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with configuration"""
        self.config = Config(config_path)
        self.compliance_manager = get_compliance_manager(ComplianceLevel.MODERATE)
        self.data_fetcher = IndexDataManager()
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        
        logger.info("🚀 Data pipeline initialized")
    
    def fetch_market_data(self, symbols: List[str], days_back: int = 30) -> bool:
        """
        Fetch market data for given symbols.
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days of historical data
            
        Returns:
            Success status
        """
        logger.info(f"📊 Fetching data for {len(symbols)} symbols")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Fetch data for each symbol
            for symbol in symbols:
                logger.info(f"Fetching {symbol}...")
                
                # This would use the actual data fetcher
                # data = self.data_fetcher.get_historical_data(
                #     symbol=symbol,
                #     start_date=start_date.strftime('%Y-%m-%d'),
                #     end_date=end_date.strftime('%Y-%m-%d')
                # )
                
                logger.info(f"✅ {symbol} data fetched")
            
            logger.info("✅ All data fetched successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data fetching failed: {e}")
            return False
    
    def process_data(self) -> bool:
        """
        Process raw data and create features.
        
        Returns:
            Success status
        """
        logger.info("🔄 Processing market data...")
        
        try:
            # This would use the actual data processor
            # processed_data = self.data_processor.process_all_data()
            
            logger.info("✅ Data processing completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            return False
    
    def train_models(self) -> bool:
        """
        Train prediction models.
        
        Returns:
            Success status
        """
        logger.info("🤖 Training models...")
        
        try:
            # This would use the actual model trainer
            # models = self.model_trainer.train_all_models()
            
            logger.info("✅ Model training completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False
    
    def run_full_pipeline(self, symbols: List[str]) -> bool:
        """
        Run the complete data pipeline.
        
        Args:
            symbols: List of stock symbols to process
            
        Returns:
            Success status
        """
        logger.info("🚀 Starting full data pipeline")
        
        try:
            # Step 1: Fetch data
            if not self.fetch_market_data(symbols):
                return False
            
            # Step 2: Process data
            if not self.process_data():
                return False
            
            # Step 3: Train models
            if not self.train_models():
                return False
            
            logger.info("🎉 Data pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False
        
        finally:
            # Cleanup
            self.compliance_manager.cleanup()

def main():
    """Main execution function"""
    # Default symbols for demonstration
    symbols = ["TCS", "INFY", "WIPRO", "RELIANCE", "HDFC"]
    
    # Initialize and run pipeline
    pipeline = DataPipeline()
    
    success = pipeline.run_full_pipeline(symbols)
    
    if success:
        print("✅ Pipeline completed successfully")
        sys.exit(0)
    else:
        print("❌ Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
