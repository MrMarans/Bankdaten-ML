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


    df = pd.read_csv(file, dayfirst=True, decimal=',', encoding='ISO-8859-1', sep=';')
    df = df.rename(columns={date_column: 'Buchungstag', amount_column: 'Betrag'})
    df['Buchungstag'] = pd.to_datetime(df['Buchungstag'], format='%d.%m.%y')
    df["Betrag"] = pd.to_numeric(df["Betrag"], errors='coerce')
    
    # Speichere Originalanzahl der Transaktionen
    original_length = len(df)
    df = df[df['Betrag'].abs() <= cutoff_value]
    filtered_length = len(df)

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

    
    # Analysiere Transaktionsmuster
    income_patterns, expense_patterns = analyze_transactions(df)
    
    # Verbesserte Funktion für Feature-Erstellung
    def create_daily_features(date, patterns, is_income=True):
        day = date.day
        month = date.month
        # Mehr Features für detailliertere Muster [ist_zahlungstag, erwarteter_betrag, tage_bis_zahlung]
        features = np.zeros(3)
        
        # Finde Muster für diesen Tag und Monat
        matching_pattern = patterns[
            (patterns['Tag'] == day) & 
            (patterns['ist_gehalt' if is_income else 'ist_fixkosten'])
        ]
        
        if not matching_pattern.empty:
            features[0] = 1  # Ist ein Zahlungstag
            # Erwarteter Betrag (normalisiert)
            features[1] = matching_pattern['mean'].iloc[0] / (5000 if is_income else 2000)
            if 'gewichtung' in matching_pattern.columns and is_income:
                features[0] *= matching_pattern['gewichtung'].iloc[0]
        
        # Finde nächsten relevanten Zahlungstag
        relevant_patterns = patterns[patterns['ist_gehalt' if is_income else 'ist_fixkosten']]
        
        # Finde nächsten Zahlungstag im gleichen Monat
        next_days = relevant_patterns[
            (relevant_patterns['Tag'] > day)
        ]['Tag']
        
        if len(next_days) > 0:
            next_day = next_days.min()
            features[2] = (next_day - day) / 31.0
        else:
            # Suche den ersten Zahlungstag
            if len(relevant_patterns) > 0:
                next_day = relevant_patterns['Tag'].min()
                features[2] = (31 - day + next_day) / 31.0
            else:
                features[2] = 1.0  # Kein nächster Zahlungstag gefunden
        
        return features

    # Erstelle Features für alle Tage
    daily_features = []
    for date in full_daily_df.index:
        income_feats = create_daily_features(date, income_patterns, True)
        expense_feats = create_daily_features(date, expense_patterns, False)
        
        # Extrahiere Gehaltsbetrag für diesen Tag wenn es ein Gehaltszahlungstag ist
        gehalt_heute = 0
        if any(income_patterns[
            (income_patterns['Tag'] == date.day) & 
            income_patterns['ist_gehalt']
        ].index):
            gehalt_muster = income_patterns[
                (income_patterns['Tag'] == date.day) & 
                income_patterns['ist_gehalt']
            ]
            gehalt_heute = gehalt_muster['mean'].iloc[0] if not gehalt_muster.empty else 0
        
        # Extrahiere Fixkostenbetrag für diesen Tag wenn es ein Fixkostentag ist
        fixkosten_heute = 0
        if any(expense_patterns[
            (expense_patterns['Tag'] == date.day) & 
            expense_patterns['ist_fixkosten']
        ].index):
            fixkosten_muster = expense_patterns[
                (expense_patterns['Tag'] == date.day) & 
                expense_patterns['ist_fixkosten']
            ]
            fixkosten_heute = fixkosten_muster['mean'].iloc[0] if not fixkosten_muster.empty else 0
        
        # Kombiniere Features
        day_features = np.concatenate([
            [date.day / 31.0, date.month / 12.0],  # Normalisierte Zeit-Features (2)
            [full_daily_df.loc[date, 'Tatsächlicher_Betrag']],  # Kontostand (1)
            income_feats,  # Eingangs-Features (3)
            expense_feats,  # Ausgangs-Features (3)
            [np.sin(2 * np.pi * date.day / 31.0), np.cos(2 * np.pi * date.day / 31.0)],  # Zyklische Tages-Features (2)
            [np.sin(2 * np.pi * date.month / 12.0), np.cos(2 * np.pi * date.month / 12.0)],  # Zyklische Monats-Features (2)
            [gehalt_heute / 5000.0],  # Normalisierter Gehaltsbetrag heute (1)
            [abs(fixkosten_heute) / 2000.0],  # Normalisierter Fixkostenbetrag heute (1)
            [1 if date.day <= 7 else 0],  # Ist Anfang des Monats (1)
            [1 if date.day >= 25 else 0],  # Ist Ende des Monats (1)
        ])
        daily_features.append(day_features)

    X = np.array(daily_features)  # Shape: (n_days, 17) - 17 Features total
    
    # Separate Normalisierung für verschiedene Features
    scaler_date = MinMaxScaler()
    scaler_month = MinMaxScaler()
    scaler_balance = MinMaxScaler()
    
    # Erstelle erweiterte Features mit Zahlungsmustern
    X_scaled = np.column_stack([
        scaler_date.fit_transform(X[:, 0].reshape(-1, 1)),      # Tag (1)
        scaler_month.fit_transform(X[:, 1].reshape(-1, 1)),     # Monat (1)
        scaler_balance.fit_transform(X[:, 2].reshape(-1, 1)),   # Kontostand (1)
        X[:, 3:]   # Alle anderen Features (14)
    ])  # Gesamtform: (n_days, 17)

    # Verbesserte Sequenzgenerierung für LSTM
    seq_length = 30
    X_seq, y_seq = [], []
    
    # Ursprüngliche Sequenzen
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i:(i + seq_length)])
        y_seq.append(X_scaled[i + seq_length, 2])  # Vorhersage des Kontostands
    
    # Datenaugmentierung durch leicht veränderte Sequenzen
    augmentation_count = min(500, len(X_scaled) - seq_length)  # Maximal 500 zusätzliche Sequenzen
    for i in range(augmentation_count):
        # Zufälligen Startpunkt wählen
        start_idx = np.random.randint(0, len(X_scaled) - seq_length)
        
        # Sequenz kopieren und leicht modifizieren
        seq = X_scaled[start_idx:(start_idx + seq_length)].copy()
        
        # Kleine zufällige Änderungen hinzufügen (außer beim Kontostand)
        noise = np.random.normal(0, 0.03, seq.shape)  # 3% Rauschen
        noise[:, 2] = 0  # Kein Rauschen beim Kontostand
        seq += noise
        
        X_seq.append(seq)
        y_seq.append(X_scaled[start_idx + seq_length, 2])  # Vorhersage des Kontostands
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Daten aufteilen mit ausgewogener Verteilung
    indices = np.random.permutation(len(X_seq))
    train_size = int(len(X_seq) * 0.8)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]
    

    model = Sequential([
        LSTM(128, input_shape=(seq_length, X_scaled.shape[1]), 
             return_sequences=True, 
             kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        BatchNormalization(),
        Dropout(0.2),
         
        LSTM(64, return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(32, 
             kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        BatchNormalization(),
        
        Dense(1, activation='linear')
    ])
    
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                 loss='huber',  # Robuster gegenüber Ausreißern
                 metrics=['mae'])
    
    # Callbacks mit angepassten Parametern
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,  # Mehr Geduld für bessere Konvergenz
        restore_best_weights=True,
        min_delta=1e-5
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,  # Mehr Geduld bei Lernraten-Anpassung
        min_lr=1e-6,
        verbose=0
    )
    
    # Custom Callback für Trainingsfortschritt
    class TrainingProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:  # Update alle 5 Epochen
                progress = min((epoch/max(300, 10000)) * 1, 0.8) 
                update_progress(progress, 
                              f"Training Epoch {epoch+1}/10000 - Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")

    # Anpassung der Batch-Size und Epochen
    history = model.fit(
        X_train, y_train,
        epochs=10000,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, TrainingProgressCallback()],
        verbose=0
    )
    
    # Vorhersage
    update_progress(0.87, "Vorhersage wird erstellt...")
    last_sequence = X_scaled[-seq_length:]  # Letzte Sequenz
    future_scaled = []
    future_days = []  # Speichere die erzeugten Tage für die Vorhersage
    
    for i in range(days_to_predict):
        next_pred = model.predict(last_sequence.reshape(1, seq_length, X_scaled.shape[1]), verbose=0)
        
        # Nächstes Datum
        last_date = full_daily_df.index[-1] + pd.Timedelta(days=i + 1)
        future_days.append(last_date)
        
        # Feature-Erstellung für das neue Datum
        next_day = last_date.day / 31.0
        next_month = last_date.month / 12.0
        
        # Zahlungsmuster-Features
        next_income = create_daily_features(last_date, income_patterns, True)
        next_expense = create_daily_features(last_date, expense_patterns, False)
        
        # Extrahiere Gehaltsbetrag für diesen Tag wenn es ein Gehaltszahlungstag ist
        gehalt_heute = 0
        if any(income_patterns[
            (income_patterns['Tag'] == last_date.day) & 
            income_patterns['ist_gehalt']
        ].index):
            gehalt_muster = income_patterns[
                (income_patterns['Tag'] == last_date.day) & 
                income_patterns['ist_gehalt']
            ]
            gehalt_heute = gehalt_muster['mean'].iloc[0] if not gehalt_muster.empty else 0
        
        # Extrahiere Fixkostenbetrag für diesen Tag wenn es ein Fixkostentag ist
        fixkosten_heute = 0
        if any(expense_patterns[
            (expense_patterns['Tag'] == last_date.day) & 
            expense_patterns['ist_fixkosten']
        ].index):
            fixkosten_muster = expense_patterns[
                (expense_patterns['Tag'] == last_date.day) & 
                expense_patterns['ist_fixkosten']
            ]
            fixkosten_heute = fixkosten_muster['mean'].iloc[0] if not fixkosten_muster.empty else 0
        
        # Zyklische Features
        next_sin_day = np.sin(2 * np.pi * last_date.day / 31.0)
        next_cos_day = np.cos(2 * np.pi * last_date.day / 31.0)
        next_sin_month = np.sin(2 * np.pi * last_date.month / 12.0)
        next_cos_month = np.cos(2 * np.pi * last_date.month / 12.0)
        
        # Kombiniere alle Features in der gleichen Reihenfolge wie beim Training
        next_features = np.array([
            next_day,                    # Tag (1)
            next_month,                  # Monat (1)
            next_pred[0][0],             # Vorhergesagter Kontostand (1)
            *next_income,                # Income Features (3)
            *next_expense,               # Expense Features (3)
            next_sin_day, next_cos_day,  # Zyklische Tages-Features (2)
            next_sin_month, next_cos_month,  # Zyklische Monats-Features (2)
            gehalt_heute / 5000.0,       # Normalisierter Gehaltsbetrag heute (1)
            abs(fixkosten_heute) / 2000.0,  # Normalisierter Fixkostenbetrag heute (1)
            1 if last_date.day <= 7 else 0,  # Ist Anfang des Monats (1)
            1 if last_date.day >= 25 else 0  # Ist Ende des Monats (1)
        ]).reshape(1, -1)  # Shape: (1, 17)
        
        # Update sequence
        last_sequence = np.vstack([last_sequence[1:], next_features])
        future_scaled.append(next_pred[0][0])
    
    future_scaled = np.array(future_scaled)
    
    # Transformation zurück
    update_progress(0.93, "Ergebnisse werden aufbereitet...")
    future_predictions = []
    last_value = full_daily_df['Tatsächlicher_Betrag'].iloc[-1]
    
    # Berechne Statistiken für die Gehaltszahlungen und Fixkosten
    salary_days = income_patterns[income_patterns['ist_gehalt']]['Tag'].values if not income_patterns[income_patterns['ist_gehalt']].empty else []
    salary_amounts = income_patterns[income_patterns['ist_gehalt']]['mean'].values if not income_patterns[income_patterns['ist_gehalt']].empty else []
    
    fixed_cost_days = expense_patterns[expense_patterns['ist_fixkosten']]['Tag'].values if not expense_patterns[expense_patterns['ist_fixkosten']].empty else []
    fixed_cost_amounts = expense_patterns[expense_patterns['ist_fixkosten']]['mean'].values if not expense_patterns[expense_patterns['ist_fixkosten']].empty else []
    
    # Berechne durchschnittliche monatliche Änderungen ohne Gehalt und Fixkosten
    monthly_pattern = df.copy()
    # Entferne Gehaltszahlungen
    for day in salary_days:
        salary_dates = monthly_pattern[monthly_pattern['Tag'] == day].index
        monthly_pattern.loc[salary_dates, 'Betrag'] = 0
    
    # Entferne Fixkosten
    for day in fixed_cost_days:
        fixed_cost_dates = monthly_pattern[monthly_pattern['Tag'] == day].index
        monthly_pattern.loc[fixed_cost_dates, 'Betrag'] = 0
    
    # Berechne den täglichen Durchschnitt der verbleibenden Änderungen
    avg_daily_change = monthly_pattern['Betrag'].mean()
    
    # Modell-basierte Vorhersage mit Korrektur durch die erkannten Muster
    for i, pred_scaled in enumerate(future_scaled):
        current_date = future_days[i]
        
        # Konvertiere den skalierten Wert zurück
        pred_value = scaler_balance.inverse_transform([[pred_scaled]])[0][0]
        
        # Korrigiere die Vorhersage basierend auf bekannten Mustern
        # 1. Gehaltszahlungen
        for j, day in enumerate(salary_days):
            if current_date.day == day:
                # Füge Gehalt hinzu
                salary_effect = salary_amounts[j] * 0.8  # Leicht gedämpft
                pred_value += salary_effect
        
        # 2. Fixkosten
        for j, day in enumerate(fixed_cost_days):
            if current_date.day == day:
                # Ziehe Fixkosten ab
                fixed_cost_effect = fixed_cost_amounts[j] * 0.8  # Leicht gedämpft
                pred_value += fixed_cost_effect  # Beachte: Fixkosten sind bereits negativ
        
        # 3. Tägliche Änderung für sonstige Ausgaben/Einnahmen
        pred_value += avg_daily_change
        
        # Begrenze die tägliche Änderung auf realistische Werte
        max_change = max(np.abs(salary_amounts).max() if len(salary_amounts) > 0 else 1000, 
                         np.abs(fixed_cost_amounts).max() if len(fixed_cost_amounts) > 0 else 500) * 1.5
        
        # Begrenze die Änderung zum vorherigen Wert
        change = pred_value - last_value
        if abs(change) > max_change and i > 0:  # Nicht den ersten Wert begrenzen
            change = np.clip(change, -max_change, max_change)
            pred_value = last_value + change
        
        # Stelle sicher, dass der Wert nicht unter 0 fällt
        pred_value = max(0, pred_value)
        
        future_predictions.append(pred_value)
        last_value = pred_value
    
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
        'Eingang_Zahlungstag', 'Eingang_Erwarteter_Betrag', 'Eingang_Tage_bis_Zahlung',
        'Ausgang_Zahlungstag', 'Ausgang_Erwarteter_Betrag', 'Ausgang_Tage_bis_Zahlung',
        'Sin_Tag', 'Cos_Tag', 'Sin_Monat', 'Cos_Monat',
        'Gehaltsbetrag_heute', 'Fixkostenbetrag_heute',
        'Ist_Anfang_des_Monats', 'Ist_Ende_des_Monats'
    ]
    
    # Stelle sicher, dass die Anzahl der Feature-Namen mit der tatsächlichen Anzahl übereinstimmt
    if len(feature_names) != X.shape[1]:
        print(f"Warnung: Anzahl Feature-Namen ({len(feature_names)}) stimmt nicht mit Feature-Dimensionen ({X.shape[1]}) überein")
        # Korrigiere die Namen falls nötig
        if len(feature_names) < X.shape[1]:
            feature_names.extend([f'Feature_{i+1}' for i in range(len(feature_names), X.shape[1])])
        else:
            feature_names = feature_names[:X.shape[1]]
    
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