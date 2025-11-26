"""
LSTM model for time series forecasting
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import joblib
import config

class LSTMForecaster:
    """LSTM model for forecasting"""
    
    def __init__(self, sequence_length=60, lstm_units=64, dropout=0.2):
        self.name = "LSTM"
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = None
        self.history = None
    
    def _create_sequences(self, X, y=None):
        """Create sequences for LSTM"""
        X_seq = []
        y_seq = []
        
        # Group by store and family
        if 'store_nbr' in X.columns and 'family' in X.columns:
            for (store, family), group in X.groupby(['store_nbr', 'family']):
                group = group.sort_values(config.DATE_COL)
                values = group[self.feature_names].values
                
                for i in range(len(values) - self.sequence_length):
                    X_seq.append(values[i:i+self.sequence_length])
                    if y is not None:
                        y_seq.append(y.iloc[group.index[i+self.sequence_length]])
        else:
            # Global sequences
            values = X[self.feature_names].values
            for i in range(len(values) - self.sequence_length):
                X_seq.append(values[i:i+self.sequence_length])
                if y is not None:
                    y_seq.append(y.iloc[i+self.sequence_length])
        
        X_seq = np.array(X_seq)
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        return X_seq
    
    def _build_model(self, input_shape):
        """Build LSTM model"""
        model = keras.Sequential([
            layers.LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(self.dropout),
            layers.LSTM(self.lstm_units, return_sequences=False),
            layers.Dropout(self.dropout),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LSTM_PARAMS['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X, y, validation_split=0.1):
        """Train LSTM model"""
        print(f"Training {self.name}...")
        
        # Select numerical features only
        self.feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_numeric)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Create sequences
        print("Creating sequences...")
        X_seq, y_seq = self._create_sequences(X_scaled_df, y)
        
        print(f"Sequence shape: {X_seq.shape}")
        print(f"Target shape: {y_seq.shape}")
        
        # Build model
        self.model = self._build_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))
        
        # Train model
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_seq, y_seq,
            epochs=config.LSTM_PARAMS['epochs'],
            batch_size=config.LSTM_PARAMS['batch_size'],
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print(f"{self.name} training complete!")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Select and scale features
        X_numeric = X[self.feature_names]
        X_scaled = self.scaler.transform(X_numeric)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Create sequences
        X_seq = self._create_sequences(X_scaled_df)
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0).flatten()
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        # Note: This returns predictions only for samples that could form sequences
        # In practice, you'd need to handle the first sequence_length samples differently
        return predictions
    
    def save(self, filepath):
        """Save model"""
        # Save Keras model
        self.model.save(str(filepath).replace('.pkl', '.h5'))
        
        # Save other attributes
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        # Load Keras model
        self.model = keras.models.load_model(str(filepath).replace('.pkl', '.h5'))
        
        # Load other attributes
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.sequence_length = data['sequence_length']
        self.lstm_units = data['lstm_units']
        self.dropout = data['dropout']
        print(f"Model loaded from {filepath}")
        return self

if __name__ == "__main__":
    print("LSTM model module loaded successfully")
