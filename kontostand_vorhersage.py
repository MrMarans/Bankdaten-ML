import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import warnings

# TensorFlow Warnungen filtern
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*The name tf.executing_eagerly_outside_functions is deprecated.*')

def process_and_predict(file, start_balance=937, cutoff_value=5000, days_to_predict=180, 
                     date_column='Buchungstag', amount_column='Betrag', progress_callback=None):
    def update_progress(progress, text):
        if progress_callback:
            progress_callback(progress, text)

    # Daten einlesen und vorbereiten
    update_progress(0.01, "Daten werden eingelesen...")
    df = pd.read_csv(file, dayfirst=True, decimal=',', encoding='ISO-8859-1', sep=';')
    df = df.rename(columns={date_column: 'Buchungstag', amount_column: 'Betrag'})
    df['Buchungstag'] = pd.to_datetime(df['Buchungstag'], format='%d.%m.%y')
    df["Betrag"] = pd.to_numeric(df["Betrag"], errors='coerce')
    
    # Speichere Originalanzahl der Transaktionen
    original_length = len(df)
    df = df[df['Betrag'].abs() <= cutoff_value]
    filtered_length = len(df)
    
    # Tägliche Änderungen berechnen
    update_progress(0.02, "Tägliche Änderungen werden berechnet...")
    df = df.sort_values('Buchungstag')
    daily_changes = df.groupby('Buchungstag')['Betrag'].sum().reset_index()
    daily_changes = daily_changes.set_index('Buchungstag')
    
    # Vollständigen Datumsbereich erstellen
    date_range = pd.date_range(start=daily_changes.index.min(), end=daily_changes.index.max(), freq='D')
    full_daily_df = pd.DataFrame(index=date_range)
    full_daily_df = full_daily_df.join(daily_changes)
    full_daily_df['Betrag'] = full_daily_df['Betrag'].fillna(0)
    
    # Kontostand berechnen
    full_daily_df['Tatsächlicher_Betrag'] = start_balance + full_daily_df['Betrag'].cumsum()
    
    # Monatliche Muster analysieren
    update_progress(0.03, "Monatliche Muster werden analysiert...")
    df['Monat'] = df['Buchungstag'].dt.month
    df['Tag'] = df['Buchungstag'].dt.day
    monthly_patterns = df.groupby(['Monat', 'Tag'])['Betrag'].mean().reset_index()
    
    # Einfaches Feature-Engineering
    update_progress(0.04, "Features werden erstellt...")
    X = np.column_stack([
        full_daily_df.index.day,
        full_daily_df.index.month,
        full_daily_df['Tatsächlicher_Betrag'].values
    ])
    
    # Separate Normalisierung für verschiedene Features
    scaler_date = MinMaxScaler()
    scaler_month = MinMaxScaler()
    scaler_balance = MinMaxScaler()
    
    X_scaled = np.column_stack([
        scaler_date.fit_transform(X[:, 0].reshape(-1, 1)),
        scaler_month.fit_transform(X[:, 1].reshape(-1, 1)),
        scaler_balance.fit_transform(X[:, 2].reshape(-1, 1))
    ])
    
    # Sequenzen erstellen (30 Tage)
    update_progress(0.05, "Sequenzen werden erstellt...")
    seq_length = 30
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i:(i + seq_length)])
        y_seq.append(X_scaled[i + seq_length, 2])  # Nur Kontostand vorhersagen
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Daten aufteilen
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    
    # Verbessertes Modell mit Regularisierung
    update_progress(0.06, "Modell wird erstellt...")
    model = Sequential([
        LSTM(32, input_shape=(seq_length, X_scaled.shape[1]), 
             return_sequences=True, 
             kernel_regularizer=tf.keras.regularizers.l2(0.01),
             recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(16, 
             kernel_regularizer=tf.keras.regularizers.l2(0.01),
             recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(8, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        
        Dense(1, activation='linear')
    ])
    
    # Training mit verbessertem Setup
    update_progress(0.07, "Modell wird trainiert...")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                 loss='huber',  # Robuster gegenüber Ausreißern
                 metrics=['mae'])
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        min_delta=1e-4
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )
    
    # Custom Callback für Trainingsfortschritt
    class TrainingProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:  # Update alle 5 Epochen
                progress = 0.07 + min((epoch/200) * 0.8, 0.8)  # Training von 7% bis 87%
                update_progress(progress, 
                              f"Training Epoch {epoch+1}/200 - Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")

    # Kleinere Batch-Size und mehr Epochen
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, TrainingProgressCallback()],
        verbose=0
    )
    
    # Vorhersage
    update_progress(0.87, "Vorhersage wird erstellt...")
    last_sequence = X_scaled[-seq_length:]
    future_scaled = []
    
    for _ in range(days_to_predict):
        next_pred = model.predict(last_sequence.reshape(1, seq_length, X_scaled.shape[1]), verbose=0)
        
        # Nächstes Datum
        last_date = full_daily_df.index[-1] + pd.Timedelta(days=len(future_scaled) + 1)
        next_day = last_date.day
        next_month = last_date.month
        
        # Neue Sequenz
        next_features = np.array([[next_day, next_month, next_pred[0][0]]])
        last_sequence = np.vstack([last_sequence[1:], next_features])
        future_scaled.append(next_pred[0][0])
    
    # Transformation zurück
    update_progress(0.93, "Ergebnisse werden aufbereitet...")
    future_predictions = []
    last_value = full_daily_df['Tatsächlicher_Betrag'].iloc[-1]
    
    for pred in future_scaled:
        # Berechne monatlichen Durchschnitt für den aktuellen Monat
        current_month = (full_daily_df.index[-1] + pd.Timedelta(days=len(future_predictions) + 1)).month
        monthly_avg_change = df[df['Monat'] == current_month]['Betrag'].mean()
        
        # Begrenze die Änderung auf realistischer Basis
        max_change = monthly_avg_change * 2  # Maximale Änderung
        pred_change = pred - last_value
        clipped_change = np.clip(pred_change, -abs(max_change), abs(max_change))
        new_value = last_value + clipped_change
        
        future_predictions.append(new_value)
        last_value = new_value
    
    future_predictions = np.array(future_predictions)
    
    # Datum-Arrays für die Visualisierung
    future_dates = pd.date_range(
        start=full_daily_df.index[-1] + pd.Timedelta(days=1),
        periods=len(future_predictions),
        freq='D'
    )
    
    update_progress(1.0, "Fertig!")
    # DataFrame für Trainingsverlauf erstellen
    training_df = pd.DataFrame({
        'Epoch': range(1, len(history.history['loss']) + 1),
        'Training Loss': history.history['loss'],
        'Validation Loss': history.history['val_loss'],
        'Training MAE': history.history['mae'],
        'Validation MAE': history.history['val_mae']
    })
    
    return {
        'dates': full_daily_df.index,
        'actual_values': full_daily_df['Tatsächlicher_Betrag'].values,
        'future_dates': future_dates,
        'future_prediction': future_predictions.reshape(-1, 1),
        'training_history': history.history,
        'training_df': training_df,  # Neues DataFrame für Trainingsverlauf
        'raw_data': full_daily_df,   # Rohdaten für die Anzeige
        'statistics': {
            'total_transactions': original_length,
            'min_amount': df['Betrag'].min(),
            'max_amount': df['Betrag'].max(),
            'mean_amount': df['Betrag'].mean(),
            'median_amount': df['Betrag'].median()
        },
        'filtering': {
            'removed_transactions': original_length - filtered_length,
            'remaining_transactions': filtered_length
        }
    }