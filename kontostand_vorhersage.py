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
                     date_column='Buchungstag', amount_column='Betrag', 
                     min_gehalt=1000, min_gehalt_vorkommen=1,
                     min_fixkosten=50, min_fixkosten_vorkommen=2,
                     max_varianz=0.3, progress_callback=None):
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
    
    # Verbesserte Analyse der Zahlungsmuster
    def analyze_transactions(df):
        # Nur Tage mit tatsächlichen Transaktionen betrachten
        income_df = df[df['Betrag'] > min_gehalt].copy()
        expense_df = df[df['Betrag'] < -min_fixkosten].copy()
        
        # Gruppiere nach Tag und berechne Statistiken nur für Tage mit Transaktionen
        income_by_day = (income_df.groupby(['Monat', 'Tag'])['Betrag']
                        .agg(['count', 'mean', 'std'])
                        .reset_index())
        
        expense_by_day = (expense_df.groupby(['Monat', 'Tag'])['Betrag']
                         .agg(['count', 'mean', 'std'])
                         .reset_index())
        
        # Identifiziere echte Gehaltseingänge (nur wenn tatsächlich Geld eingegangen ist)
        income_by_day['ist_gehalt'] = (
            (income_by_day['mean'] > min_gehalt) &  # Mindestbetrag für Gehalt
            (income_by_day['count'] >= min_gehalt_vorkommen) &  # Mindestens n Vorkommen
            (income_by_day['std'] / income_by_day['mean'] < 0.5)  # Maximal 50% Abweichung
        )
        
        # Identifiziere echte Fixkosten (nur wenn tatsächlich Abbuchungen erfolgten)
        expense_by_day['ist_fixkosten'] = (
            (expense_by_day['mean'] < -min_fixkosten) &  # Mindestbetrag für Fixkosten
            (expense_by_day['count'] >= min_fixkosten_vorkommen) &  # Mindestens n Vorkommen
            (expense_by_day['std'] / expense_by_day['mean'].abs() < max_varianz)  # Maximale Varianz
        )
        
        # Gewichte die Gehaltseingänge nach ihrer Häufigkeit und Höhe
        income_by_day['gewichtung'] = (
            (income_by_day['count'] / income_by_day['count'].max()) * 
            (income_by_day['mean'] / income_by_day['mean'].max())
        )
        
        return income_by_day, expense_by_day

    # Feature-Engineering
    update_progress(0.04, "Features werden erstellt...")
    
    # Analysiere Transaktionsmuster
    income_patterns, expense_patterns = analyze_transactions(df)
    
    def create_daily_features(date, patterns, is_income=True):
        day = date.day
        month = date.month
        features = np.zeros(2)  # [ist_zahlungstag, tage_bis_zahlung]
        
        # Finde Muster für diesen Tag und Monat
        matching_pattern = patterns[
            (patterns['Tag'] == day) & 
            (patterns['Monat'] == month)
        ]
        
        if not matching_pattern.empty:
            if is_income and matching_pattern['ist_gehalt'].iloc[0]:
                features[0] = 1  # Gehaltszahlungstag
                if 'gewichtung' in matching_pattern.columns:
                    features[0] *= matching_pattern['gewichtung'].iloc[0]
            elif not is_income and matching_pattern['ist_fixkosten'].iloc[0]:
                features[0] = 1  # Fixkostentag
        
        # Finde nächsten relevanten Zahlungstag im gleichen Monat
        if is_income:
            next_days = patterns[
                (patterns['Monat'] == month) & 
                (patterns['Tag'] > day) & 
                patterns['ist_gehalt']
            ]['Tag']
        else:
            next_days = patterns[
                (patterns['Monat'] == month) & 
                (patterns['Tag'] > day) & 
                patterns['ist_fixkosten']
            ]['Tag']
        
        if len(next_days) > 0:
            next_day = next_days.iloc[0]
            features[1] = (next_day - day) / 31.0
        else:
            # Suche im nächsten Monat
            next_month = (month % 12) + 1
            if is_income:
                next_days = patterns[
                    (patterns['Monat'] == next_month) & 
                    patterns['ist_gehalt']
                ]['Tag']
            else:
                next_days = patterns[
                    (patterns['Monat'] == next_month) & 
                    patterns['ist_fixkosten']
                ]['Tag']
            
            if len(next_days) > 0:
                next_day = next_days.iloc[0]
                features[1] = (31 - day + next_day) / 31.0
            else:
                features[1] = 1.0  # Kein nächster Zahlungstag gefunden
        
        return features

    # Erstelle Features für alle Tage
    daily_features = []
    for date in full_daily_df.index:
        income_feats = create_daily_features(date, income_patterns, True)
        expense_feats = create_daily_features(date, expense_patterns, False)
        
        # Kombiniere Features
        day_features = np.concatenate([
            [date.day / 31.0, date.month / 12.0],  # Normalisierte Zeit-Features (2)
            [full_daily_df.loc[date, 'Tatsächlicher_Betrag']],  # Kontostand (1)
            income_feats,  # Eingangs-Features (2) - nur Zahlungstag und Tage bis zur nächsten Zahlung
            expense_feats,  # Ausgangs-Features (2) - nur Zahlungstag und Tage bis zur nächsten Zahlung
            [np.sin(2 * np.pi * date.day / 31.0), np.cos(2 * np.pi * date.day / 31.0)],  # Zyklische Tages-Features (2)
            [np.sin(2 * np.pi * date.month / 12.0), np.cos(2 * np.pi * date.month / 12.0)]  # Zyklische Monats-Features (2)
        ])
        daily_features.append(day_features)

    X = np.array(daily_features)  # Shape: (n_days, 11) - 11 Features total
    
    # Separate Normalisierung für verschiedene Features
    scaler_date = MinMaxScaler()
    scaler_month = MinMaxScaler()
    scaler_balance = MinMaxScaler()
    
    # Erstelle erweiterte Features mit Zahlungsmustern
    X_scaled = np.column_stack([
        scaler_date.fit_transform(X[:, 0].reshape(-1, 1)),      # Tag (1)
        scaler_month.fit_transform(X[:, 1].reshape(-1, 1)),     # Monat (1)
        scaler_balance.fit_transform(X[:, 2].reshape(-1, 1)),   # Kontostand (1)
        X[:, 3:5],   # Income Features (2) - Zahlungstag und Tage bis zur nächsten Zahlung
        X[:, 5:7],   # Expense Features (2) - Zahlungstag und Tage bis zur nächsten Zahlung
        X[:, 7:]     # Zyklische Features (4)
    ])  # Gesamtform: (n_days, 11)

    # Sequenzen erstellen (30 Tage) mit allen Features
    update_progress(0.05, "Sequenzen werden erstellt...")
    seq_length = 30
    X_seq, y_seq = [], []
    
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i:(i + seq_length)])
        y_seq.append(X_scaled[i + seq_length, 2])  # Vorhersage des Kontostands
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Daten aufteilen
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    
    # Verbessertes Modell mit mehr Features
    update_progress(0.06, "Modell wird erstellt...")
    model = Sequential([
        LSTM(64, input_shape=(seq_length, X_scaled.shape[1]), 
             return_sequences=True, 
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, 
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16, activation='relu',
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
    last_sequence = X_scaled[-seq_length:]  # Shape: (seq_length, 15)
    future_scaled = []
    
    for _ in range(days_to_predict):
        next_pred = model.predict(last_sequence.reshape(1, seq_length, X_scaled.shape[1]), verbose=0)
        
        # Nächstes Datum
        last_date = full_daily_df.index[-1] + pd.Timedelta(days=len(future_scaled) + 1)
        
        # Feature-Erstellung für das neue Datum
        next_day = last_date.day / 31.0
        next_month = last_date.month / 12.0
        
        # Zahlungsmuster-Features
        next_income = create_daily_features(last_date, income_patterns, True)
        next_expense = create_daily_features(last_date, expense_patterns, False)
        
        # Zyklische Features
        next_sin_day = np.sin(2 * np.pi * last_date.day / 31.0)
        next_cos_day = np.cos(2 * np.pi * last_date.day / 31.0)
        next_sin_month = np.sin(2 * np.pi * last_date.month / 12.0)
        next_cos_month = np.cos(2 * np.pi * last_date.month / 12.0)
        
        # Kombiniere alle Features in der gleichen Reihenfolge wie beim Training
        next_features = np.array([
            next_day,                    # Tag (1)
            next_month,                  # Monat (1)
            next_pred[0][0],            # Vorhergesagter Kontostand (1)
            *next_income,               # Income Features (2)
            *next_expense,              # Expense Features (2)
            next_sin_day, next_cos_day,  # Zyklische Tages-Features (2)
            next_sin_month, next_cos_month  # Zyklische Monats-Features (2)
        ]).reshape(1, -1)  # Shape: (1, 11)
        
        # Update sequence
        last_sequence = np.vstack([last_sequence[1:], next_features])
        future_scaled.append(next_pred[0][0])
    
    future_scaled = np.array(future_scaled)
    
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
    
    # Erstelle ein DataFrame mit den ursprünglichen Features
    feature_names = [
        'Tag', 'Monat', 'Kontostand',
        'Eingang_Zahlungstag', 'Eingang_Tage_bis_Zahlung',
        'Ausgang_Zahlungstag', 'Ausgang_Tage_bis_Zahlung',
        'Sin_Tag', 'Cos_Tag', 'Sin_Monat', 'Cos_Monat'
    ]
    
    X_original_df = pd.DataFrame(X, columns=feature_names, index=full_daily_df.index)

    return {
        'dates': full_daily_df.index,
        'actual_values': full_daily_df['Tatsächlicher_Betrag'].values,
        'future_dates': future_dates,
        'future_prediction': future_predictions.reshape(-1, 1),
        'training_history': history.history,
        'training_df': training_df,
        'raw_data': full_daily_df,
        'income_patterns': income_patterns,
        'expense_patterns': expense_patterns,
        'X_train': X_train,
        'y_train': y_train,
        'X_original': X_original_df,  # Neu: Ursprüngliche Features als DataFrame
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