"""
Time Series Forecasting Models Module
Implements ARIMA, SARIMA, and LSTM models for financial time series
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Data Preprocessing
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
import os


class TimeSeriesForecaster:
    """Base class for time series forecasting models"""
    
    def __init__(self, model_type: str = 'arima'):
        """
        Initialize the forecaster
        
        Args:
            model_type: Type of model ('arima', 'sarima', 'lstm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        self.model_params = {}
        
    def prepare_data(self, data: pd.Series, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM modeling
        
        Args:
            data: Time series data
            sequence_length: Number of time steps to use for prediction
            
        Returns:
            X, y arrays for training
        """
        data_array = data.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data_array)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        if self.model_type in ['arima', 'sarima']:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
        elif self.model_type == 'lstm':
            self.model.save(filepath)
        
        # Save scaler
        scaler_path = filepath.replace('.pkl', '_scaler.pkl').replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # Save parameters
        params_path = filepath.replace('.pkl', '_params.json').replace('.h5', '_params.json')
        with open(params_path, 'w') as f:
            json.dump(self.model_params, f)
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        if self.model_type in ['arima', 'sarima']:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
        elif self.model_type == 'lstm':
            self.model = tf.keras.models.load_model(filepath)
            
        # Load scaler
        scaler_path = filepath.replace('.pkl', '_scaler.pkl').replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.is_fitted = True
        return self


class ARIMAModel(TimeSeriesForecaster):
    """ARIMA/SARIMA Model Implementation"""
    
    def __init__(self, seasonal: bool = False, m: int = 12):
        """
        Initialize ARIMA model
        
        Args:
            seasonal: Whether to use seasonal ARIMA (SARIMA)
            m: Seasonal period for SARIMA
        """
        model_type = 'sarima' if seasonal else 'arima'
        super().__init__(model_type)
        self.seasonal = seasonal
        self.m = m
        self.order = None
        self.seasonal_order = None
        
    def find_optimal_parameters(self, train_data: pd.Series, **kwargs) -> Dict:
        """
        Find optimal ARIMA parameters using auto_arima
        
        Args:
            train_data: Training time series
            **kwargs: Additional parameters for auto_arima
            
        Returns:
            Dictionary with optimal parameters
        """
        print("Finding optimal ARIMA parameters...")
        
        # Default parameters
        default_kwargs = {
            'start_p': 0, 'd': None, 'start_q': 0,
            'max_p': 5, 'max_d': 2, 'max_q': 5,
            'seasonal': self.seasonal,
            'm': self.m if self.seasonal else 1,
            'start_P': 0, 'D': None, 'start_Q': 0,
            'max_P': 2, 'max_D': 1, 'max_Q': 2,
            'stepwise': True,
            'trace': True,
            'error_action': 'ignore',
            'suppress_warnings': True,
            'information_criterion': 'aic'
        }
        
        # Update with user parameters
        default_kwargs.update(kwargs)
        
        # Run auto_arima
        auto_model = auto_arima(
            train_data,
            **default_kwargs
        )
        
        # Extract optimal parameters
        self.order = auto_model.order
        self.seasonal_order = auto_model.seasonal_order
        
        optimal_params = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': auto_model.aic(),
            'bic': auto_model.bic()
        }
        
        print(f"Optimal order: {self.order}")
        if self.seasonal:
            print(f"Optimal seasonal order: {self.seasonal_order}")
        
        return optimal_params
    
    def fit(self, train_data: pd.Series, **kwargs):
        """
        Fit ARIMA/SARIMA model
        
        Args:
            train_data: Training time series
            **kwargs: Parameters for ARIMA/SARIMAX
        """
        # Find optimal parameters if not provided
        if self.order is None:
            self.find_optimal_parameters(train_data, **kwargs)
        
        print(f"Fitting {self.model_type.upper()} model...")
        
        if self.seasonal:
            # SARIMA model
            self.model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                **kwargs
            )
        else:
            # ARIMA model
            self.model = ARIMA(
                train_data,
                order=self.order,
                **kwargs
            )
        
        self.model_fit = self.model.fit()
        self.is_fitted = True
        
        # Store fitted values
        self.fitted_values = self.model_fit.fittedvalues
        
        print(f"Model fitting complete. AIC: {self.model_fit.aic:.2f}")
        return self
    
    def predict(self, steps: int, return_conf_int: bool = False, alpha: float = 0.05):
        """
        Generate predictions
        
        Args:
            steps: Number of steps to forecast
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            Predictions (and optionally confidence intervals)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.model_fit.get_forecast(steps=steps)
        predictions = forecast.predicted_mean
        
        if return_conf_int:
            conf_int = forecast.conf_int(alpha=alpha)
            return predictions, conf_int
        
        return predictions
    
    def summary(self):
        """Print model summary"""
        if self.is_fitted:
            return self.model_fit.summary()
        else:
            print("Model not fitted yet.")


class LSTMModel(TimeSeriesForecaster):
    """LSTM Neural Network for Time Series Forecasting"""
    
    def __init__(self, sequence_length: int = 60):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Number of time steps to use for prediction
        """
        super().__init__('lstm')
        self.sequence_length = sequence_length
        self.history = None
        
    def build_model(self, input_shape: Tuple, **kwargs):
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data
            **kwargs: Model architecture parameters
        """
        # Default architecture
        default_params = {
            'units_layer1': 50,
            'units_layer2': 25,
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }
        default_params.update(kwargs)
        
        self.model_params = default_params
        
        # Build model
        self.model = Sequential([
            LSTM(
                units=default_params['units_layer1'],
                return_sequences=True,
                input_shape=input_shape
            ),
            Dropout(default_params['dropout_rate']),
            
            LSTM(
                units=default_params['units_layer2'],
                return_sequences=False
            ),
            Dropout(default_params['dropout_rate']),
            
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=default_params['learning_rate'])
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print("LSTM model built successfully.")
        print(self.model.summary())
        
        return self
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            **kwargs):
        """
        Train LSTM model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            **kwargs: Training parameters
        """
        # Default training parameters
        default_kwargs = {
            'epochs': 50,
            'batch_size': 32,
            'verbose': 1,
            'callbacks': []
        }
        default_kwargs.update(kwargs)
        
        # Add callbacks if not provided
        if not default_kwargs['callbacks']:
            default_kwargs['callbacks'] = [
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                )
            ]
        
        # Reshape for LSTM [samples, time steps, features]
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        # Train model
        print("Training LSTM model...")
        
        if X_val is not None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                **default_kwargs
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                **default_kwargs
            )
        
        self.is_fitted = True
        print("LSTM training complete.")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Reshape if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def predict_future(self, last_sequence: np.ndarray, steps: int) -> np.ndarray:
        """
        Predict multiple future steps
        
        Args:
            last_sequence: Last known sequence of data
            steps: Number of future steps to predict
            
        Returns:
            Future predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            current_sequence_reshaped = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next step
            next_step = self.model.predict(current_sequence_reshaped, verbose=0)
            next_step_original = self.scaler.inverse_transform(next_step)[0, 0]
            
            predictions.append(next_step_original)
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = self.scaler.transform([[next_step_original]])[0, 0]
        
        return np.array(predictions)
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()