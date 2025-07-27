"""
Integration Example: Extending Existing Model Utils with Cost Awareness
=======================================================================

This example shows how to integrate cost-awareness into the existing ML pipeline
by extending the existing model_utils.py with cost capabilities.
"""

import sys
import os

# Add project root to path
sys.path.append('/home/runner/work/AI-ML-Based-Stock-Price-Prediction/AI-ML-Based-Stock-Price-Prediction')

def create_cost_aware_model_manager():
    """
    Example of how to extend the existing ModelManager with cost awareness.
    """
    print("=== Extending ModelManager with Cost Awareness ===")
    
    try:
        # Use existing ModelManager as base
        from src.models.model_utils import ModelManager
        from src.models.training.cost_integration_mixin import CostIntegrationMixin
        from src.models.config.cost_integration import create_basic_cost_config
        
        class CostAwareModelManager(ModelManager, CostIntegrationMixin):
            """Extended ModelManager with cost awareness."""
            
            def __init__(self, config=None):
                # Initialize base ModelManager
                super(ModelManager, self).__init__(config)
                
                # Initialize cost integration
                cost_config = create_basic_cost_config()
                CostIntegrationMixin.__init__(self, cost_config)
                
                print("✓ CostAwareModelManager created successfully")
            
            def train_model_with_costs(self, model_type, X_train, y_train, **kwargs):
                """Train model with cost considerations."""
                print(f"Training {model_type} with cost awareness...")
                
                # Identify cost features
                if hasattr(X_train, 'columns'):
                    cost_features = self._identify_cost_features(X_train.columns)
                    print(f"Found {len(cost_features)} cost features")
                
                # Use existing train_model method
                return self.train_model(model_type, X_train, y_train, **kwargs)
        
        # Create instance
        cost_manager = CostAwareModelManager()
        
        print("✓ Cost-aware ModelManager integration example completed")
        return True
        
    except Exception as e:
        print(f"✗ Cost-aware ModelManager integration failed: {e}")
        return False

def create_cost_aware_evaluator():
    """
    Example of extending the existing ModelEvaluator with cost awareness.
    """
    print("\n=== Extending ModelEvaluator with Cost Awareness ===")
    
    try:
        from src.models.model_utils import ModelEvaluator
        from src.models.evaluation.cost_metrics import CostMetrics
        
        class CostAwareModelEvaluator(ModelEvaluator):
            """Extended ModelEvaluator with cost metrics."""
            
            def __init__(self):
                super().__init__()
                self.cost_metrics = CostMetrics()
                print("✓ CostAwareModelEvaluator created successfully")
            
            def evaluate_model_with_costs(self, y_true, y_pred, X_features=None, 
                                        cost_features=None, model_name="model"):
                """Evaluate model with both standard and cost metrics."""
                print(f"Evaluating {model_name} with cost awareness...")
                
                # Get standard metrics
                standard_metrics = self.evaluate_model(y_true, y_pred, model_name)
                
                # Add cost metrics if cost features available
                cost_enhanced_metrics = {}
                if X_features is not None and cost_features:
                    cost_enhanced_metrics = self.cost_metrics.calculate_cost_enhanced_metrics(
                        y_true, y_pred, X_features, cost_features
                    )
                
                # Combine results
                combined_metrics = {
                    'standard_metrics': standard_metrics,
                    'cost_enhanced_metrics': cost_enhanced_metrics,
                    'model_name': model_name
                }
                
                print(f"Calculated {len(standard_metrics) + len(cost_enhanced_metrics)} total metrics")
                return combined_metrics
        
        # Create instance
        cost_evaluator = CostAwareModelEvaluator()
        
        print("✓ Cost-aware ModelEvaluator integration example completed")
        return True
        
    except Exception as e:
        print(f"✗ Cost-aware ModelEvaluator integration failed: {e}")
        return False

def demonstrate_feature_integration():
    """Show how to integrate cost features with existing data processors."""
    print("\n=== Integrating with Existing Data Processors ===")
    
    try:
        # This would integrate with the existing processors.py
        print("Integration approach with existing data processors:")
        print("""
        from src.data.processors import TechnicalIndicatorProcessor
        from src.models.features.cost_pipeline import CostFeaturePipeline
        
        # Method 1: Wrap existing processor
        class EnhancedProcessor:
            def __init__(self):
                self.tech_processor = TechnicalIndicatorProcessor()
                self.cost_pipeline = CostFeaturePipeline()
            
            def process_all_features(self, df):
                # Process technical indicators first
                tech_result = self.tech_processor.process_all_indicators(df)
                
                # Add cost features
                cost_result = self.cost_pipeline.process_features(tech_result.data)
                
                return cost_result
        
        # Method 2: Extend existing processor
        class CostAwareTechnicalProcessor(TechnicalIndicatorProcessor):
            def __init__(self):
                super().__init__()
                self.cost_pipeline = CostFeaturePipeline()
            
            def process_all_indicators(self, df):
                # Call parent method
                result = super().process_all_indicators(df)
                
                # Add cost features
                return self.cost_pipeline.process_features(result.data)
        """)
        
        print("✓ Data processor integration examples provided")
        return True
        
    except Exception as e:
        print(f"✗ Data processor integration failed: {e}")
        return False

def show_configuration_integration():
    """Show how to integrate with existing configuration management."""
    print("\n=== Configuration Integration ===")
    
    try:
        print("Configuration integration approach:")
        print("""
        # Extend existing Config class with cost settings
        from src.utils.config_manager import Config
        from src.models.config.cost_integration import CostIntegrationConfig
        
        class EnhancedConfig(Config):
            def __init__(self):
                super().__init__()
                self.cost_config = CostIntegrationConfig()
                
                # Update cost config from base config
                self.cost_config.update_from_base_config(self)
            
            def get_cost_config(self):
                return self.cost_config
            
            def is_cost_integration_enabled(self):
                return self.cost_config.is_cost_integration_enabled()
        
        # Usage in existing code:
        config = EnhancedConfig()
        if config.is_cost_integration_enabled():
            # Use cost-aware features
            pass
        else:
            # Use standard features (backward compatibility)
            pass
        """)
        
        print("✓ Configuration integration approach demonstrated")
        return True
        
    except Exception as e:
        print(f"✗ Configuration integration failed: {e}")
        return False

def main():
    """Main integration demonstration."""
    print("Cost-Aware ML Integration with Existing Components")
    print("=" * 60)
    
    results = []
    
    # Run integration examples
    results.append(create_cost_aware_model_manager())
    results.append(create_cost_aware_evaluator())
    results.append(demonstrate_feature_integration())
    results.append(show_configuration_integration())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n" + "=" * 60)
    print(f"Integration Examples: {passed}/{total} successful")
    
    if passed == total:
        print("🎉 All integration examples completed successfully!")
    
    print("\nKey Integration Benefits:")
    print("✓ Extends existing components without breaking changes")
    print("✓ Mixin pattern allows selective cost integration")
    print("✓ Configuration-driven integration levels")
    print("✓ Backward compatibility maintained")
    print("✓ Gradual adoption possible")
    
    print("\nNext Steps for Full Integration:")
    print("1. Install required dependencies (numpy, pandas, scikit-learn)")
    print("2. Create cost data or use synthetic cost generation")
    print("3. Configure integration level (basic/advanced/full)")
    print("4. Test with existing models and datasets")
    print("5. Monitor cost-adjusted performance improvements")

if __name__ == "__main__":
    main()