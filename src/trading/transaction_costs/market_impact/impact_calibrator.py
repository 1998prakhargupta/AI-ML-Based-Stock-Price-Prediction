"""
Impact Calibrator
================

Calibrates market impact model parameters based on historical execution data.
Provides tools for parameter optimization and model validation.
"""

from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging

from .linear_model import LinearImpactModel
from .sqrt_model import SquareRootImpactModel
from .adaptive_model import AdaptiveImpactModel
from ..models import TransactionRequest, MarketConditions

logger = logging.getLogger(__name__)


class CalibrationData:
    """Container for calibration data point."""
    
    def __init__(
        self,
        request: TransactionRequest,
        market_conditions: MarketConditions,
        actual_impact: Decimal,
        execution_timestamp: datetime
    ):
        self.request = request
        self.market_conditions = market_conditions
        self.actual_impact = actual_impact
        self.execution_timestamp = execution_timestamp


class ImpactCalibrator:
    """
    Calibrates market impact model parameters using historical execution data.
    
    Provides functionality to:
    - Calibrate model parameters to minimize prediction errors
    - Validate model performance on out-of-sample data
    - Compare different model types
    - Generate calibration reports
    """
    
    def __init__(self):
        """Initialize the impact calibrator."""
        self.logger = logger.getChild(self.__class__.__name__)
        self.calibration_data: List[CalibrationData] = []
        self.calibration_results: Dict[str, Any] = {}
    
    def add_calibration_data(
        self,
        request: TransactionRequest,
        market_conditions: MarketConditions,
        actual_impact: Decimal,
        execution_timestamp: Optional[datetime] = None
    ) -> None:
        """
        Add calibration data point.
        
        Args:
            request: Transaction request
            market_conditions: Market conditions at execution
            actual_impact: Actual observed market impact
            execution_timestamp: Execution timestamp
        """
        if execution_timestamp is None:
            execution_timestamp = datetime.now()
        
        data_point = CalibrationData(
            request, market_conditions, actual_impact, execution_timestamp
        )
        self.calibration_data.append(data_point)
        
        self.logger.debug(f"Added calibration data point: {request.symbol}, impact={actual_impact}")
    
    def calibrate_linear_model(
        self,
        initial_alpha: Decimal = Decimal('0.1'),
        alpha_range: Tuple[Decimal, Decimal] = (Decimal('0.01'), Decimal('0.5'))
    ) -> Dict[str, Any]:
        """
        Calibrate linear impact model parameters.
        
        Args:
            initial_alpha: Initial alpha parameter
            alpha_range: Range for alpha optimization
            
        Returns:
            Calibration results
        """
        if not self.calibration_data:
            raise ValueError("No calibration data available")
        
        self.logger.info("Calibrating linear impact model")
        
        best_alpha = initial_alpha
        best_error = float('inf')
        
        # Simple grid search optimization
        alpha_step = (alpha_range[1] - alpha_range[0]) / Decimal('20')
        current_alpha = alpha_range[0]
        
        while current_alpha <= alpha_range[1]:
            model = LinearImpactModel(alpha=current_alpha)
            error = self._calculate_model_error(model)
            
            if error < best_error:
                best_error = error
                best_alpha = current_alpha
            
            current_alpha += alpha_step
        
        # Calculate final statistics
        calibrated_model = LinearImpactModel(alpha=best_alpha)
        results = self._generate_calibration_results(calibrated_model, "Linear")
        
        self.logger.info(f"Linear model calibrated: alpha={best_alpha:.4f}, RMSE={best_error:.4f}")
        return results
    
    def calibrate_sqrt_model(
        self,
        initial_alpha: Decimal = Decimal('0.3'),
        alpha_range: Tuple[Decimal, Decimal] = (Decimal('0.1'), Decimal('0.8'))
    ) -> Dict[str, Any]:
        """
        Calibrate square root impact model parameters.
        
        Args:
            initial_alpha: Initial alpha parameter
            alpha_range: Range for alpha optimization
            
        Returns:
            Calibration results
        """
        if not self.calibration_data:
            raise ValueError("No calibration data available")
        
        self.logger.info("Calibrating square root impact model")
        
        best_alpha = initial_alpha
        best_error = float('inf')
        
        # Simple grid search optimization
        alpha_step = (alpha_range[1] - alpha_range[0]) / Decimal('20')
        current_alpha = alpha_range[0]
        
        while current_alpha <= alpha_range[1]:
            model = SquareRootImpactModel(alpha=current_alpha)
            error = self._calculate_model_error(model)
            
            if error < best_error:
                best_error = error
                best_alpha = current_alpha
            
            current_alpha += alpha_step
        
        # Calculate final statistics
        calibrated_model = SquareRootImpactModel(alpha=best_alpha)
        results = self._generate_calibration_results(calibrated_model, "Square Root")
        
        self.logger.info(f"Square root model calibrated: alpha={best_alpha:.4f}, RMSE={best_error:.4f}")
        return results
    
    def calibrate_adaptive_model(
        self,
        initial_alpha: Decimal = Decimal('0.2'),
        alpha_range: Tuple[Decimal, Decimal] = (Decimal('0.1'), Decimal('0.5'))
    ) -> Dict[str, Any]:
        """
        Calibrate adaptive impact model parameters.
        
        Args:
            initial_alpha: Initial base alpha parameter
            alpha_range: Range for alpha optimization
            
        Returns:
            Calibration results
        """
        if not self.calibration_data:
            raise ValueError("No calibration data available")
        
        self.logger.info("Calibrating adaptive impact model")
        
        best_alpha = initial_alpha
        best_error = float('inf')
        
        # Simple grid search optimization for base alpha
        alpha_step = (alpha_range[1] - alpha_range[0]) / Decimal('20')
        current_alpha = alpha_range[0]
        
        while current_alpha <= alpha_range[1]:
            model = AdaptiveImpactModel(base_alpha=current_alpha)
            error = self._calculate_model_error(model)
            
            if error < best_error:
                best_error = error
                best_alpha = current_alpha
            
            current_alpha += alpha_step
        
        # Calculate final statistics
        calibrated_model = AdaptiveImpactModel(base_alpha=best_alpha)
        results = self._generate_calibration_results(calibrated_model, "Adaptive")
        
        self.logger.info(f"Adaptive model calibrated: base_alpha={best_alpha:.4f}, RMSE={best_error:.4f}")
        return results
    
    def _calculate_model_error(self, model) -> float:
        """
        Calculate model prediction error (RMSE).
        
        Args:
            model: Impact model to evaluate
            
        Returns:
            Root mean squared error
        """
        errors = []
        
        for data_point in self.calibration_data:
            try:
                predicted_impact = model.calculate_impact(
                    data_point.request,
                    data_point.market_conditions
                )
                error = float(predicted_impact - data_point.actual_impact)
                errors.append(error ** 2)
            except Exception as e:
                self.logger.warning(f"Error calculating impact for calibration: {e}")
                continue
        
        if not errors:
            return float('inf')
        
        mse = sum(errors) / len(errors)
        rmse = mse ** 0.5
        return rmse
    
    def _generate_calibration_results(self, model, model_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive calibration results.
        
        Args:
            model: Calibrated model
            model_name: Name of the model
            
        Returns:
            Calibration results dictionary
        """
        predictions = []
        actuals = []
        errors = []
        relative_errors = []
        
        for data_point in self.calibration_data:
            try:
                predicted_impact = model.calculate_impact(
                    data_point.request,
                    data_point.market_conditions
                )
                actual_impact = data_point.actual_impact
                
                predictions.append(float(predicted_impact))
                actuals.append(float(actual_impact))
                
                error = float(predicted_impact - actual_impact)
                errors.append(error)
                
                if actual_impact != 0:
                    relative_error = error / float(actual_impact)
                    relative_errors.append(relative_error)
                
            except Exception as e:
                self.logger.warning(f"Error in calibration evaluation: {e}")
                continue
        
        if not predictions:
            return {'error': 'No valid predictions generated'}
        
        # Calculate statistics
        rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
        mae = sum(abs(e) for e in errors) / len(errors)
        mean_error = sum(errors) / len(errors)
        
        if relative_errors:
            mean_relative_error = sum(relative_errors) / len(relative_errors)
            rmse_relative = (sum(e**2 for e in relative_errors) / len(relative_errors)) ** 0.5
        else:
            mean_relative_error = None
            rmse_relative = None
        
        results = {
            'model_name': model_name,
            'model_parameters': model.get_model_parameters(),
            'calibration_timestamp': datetime.now(),
            'data_points': len(self.calibration_data),
            'valid_predictions': len(predictions),
            'statistics': {
                'rmse': rmse,
                'mae': mae,
                'mean_error': mean_error,
                'mean_relative_error': mean_relative_error,
                'rmse_relative': rmse_relative
            },
            'performance_metrics': {
                'accuracy_90_pct': sum(1 for e in relative_errors if abs(e) < 0.9) / len(relative_errors) if relative_errors else None,
                'accuracy_50_pct': sum(1 for e in relative_errors if abs(e) < 0.5) / len(relative_errors) if relative_errors else None,
                'accuracy_25_pct': sum(1 for e in relative_errors if abs(e) < 0.25) / len(relative_errors) if relative_errors else None
            }
        }
        
        return results
    
    def compare_models(self) -> Dict[str, Any]:
        """
        Compare different impact models on the calibration data.
        
        Returns:
            Model comparison results
        """
        if not self.calibration_data:
            raise ValueError("No calibration data available")
        
        self.logger.info("Comparing impact models")
        
        # Calibrate each model
        linear_results = self.calibrate_linear_model()
        sqrt_results = self.calibrate_sqrt_model()
        adaptive_results = self.calibrate_adaptive_model()
        
        # Compare results
        comparison = {
            'comparison_timestamp': datetime.now(),
            'models': {
                'linear': linear_results,
                'square_root': sqrt_results,
                'adaptive': adaptive_results
            },
            'best_model': self._determine_best_model(linear_results, sqrt_results, adaptive_results)
        }
        
        self.logger.info(f"Model comparison completed. Best model: {comparison['best_model']}")
        return comparison
    
    def _determine_best_model(self, *model_results) -> str:
        """
        Determine the best model based on RMSE.
        
        Args:
            *model_results: Model calibration results
            
        Returns:
            Name of the best model
        """
        best_rmse = float('inf')
        best_model = None
        
        for result in model_results:
            if 'statistics' in result and 'rmse' in result['statistics']:
                rmse = result['statistics']['rmse']
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = result['model_name']
        
        return best_model or 'unknown'
    
    def clear_calibration_data(self) -> None:
        """Clear all calibration data."""
        self.calibration_data.clear()
        self.calibration_results.clear()
        self.logger.info("Calibration data cleared")
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """
        Get summary of calibration data.
        
        Returns:
            Calibration data summary
        """
        if not self.calibration_data:
            return {'message': 'No calibration data available'}
        
        symbols = set(data.request.symbol for data in self.calibration_data)
        impacts = [float(data.actual_impact) for data in self.calibration_data]
        
        return {
            'total_data_points': len(self.calibration_data),
            'unique_symbols': len(symbols),
            'symbols': sorted(list(symbols)),
            'impact_statistics': {
                'min': min(impacts),
                'max': max(impacts),
                'mean': sum(impacts) / len(impacts),
                'median': sorted(impacts)[len(impacts) // 2]
            },
            'date_range': {
                'earliest': min(data.execution_timestamp for data in self.calibration_data),
                'latest': max(data.execution_timestamp for data in self.calibration_data)
            }
        }