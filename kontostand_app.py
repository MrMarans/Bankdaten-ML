import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from kontostand_vorhersage import process_and_predict
import numpy as np

def main():
    st.title('Kontostand Vorhersage')
    
    # Sidebar für Upload und Parameter
    with st.sidebar:
        st.header('Daten und Parameter')
        
        # File Upload
        uploaded_file = st.file_uploader("CSV-Datei hochladen", type=['csv'])
        
        if uploaded_file is not None:
            # Lese die CSV-Datei für die Spaltenauswahl
            try:
                # Versuche verschiedene Encodings
                encodings = ['ISO-8859-1', 'utf-8', 'cp1252']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, nrows=1, encoding=encoding, sep=';')
                        break
                    except:
                        continue
                
                if df is not None:
                    # Spaltenauswahl
                    st.subheader('Spaltenzuordnung')
                    
                    # Bereinige die Spaltennamen
                    columns = [col.strip('"\' ') for col in df.columns]
                    
                    date_column = st.selectbox(
                        'Spalte mit Datum auswählen',
                        options=columns,
                        index=columns.index('Buchungstag') if 'Buchungstag' in columns else 0
                    )
                    amount_column = st.selectbox(
                        'Spalte mit Betrag auswählen',
                        options=columns,
                        index=columns.index('Betrag') if 'Betrag' in columns else 0
                    )
                    
                    # Grundlegende Parameter
                    st.subheader('Grundeinstellungen')
                    start_balance = st.number_input('Startbetrag (€)', value=937)
                    cutoff_value = st.number_input('Cutoff-Wert für Beträge (€)', value=5000)
                    prediction_days = st.number_input('Anzahl Tage für Vorhersage', 
                                                    min_value=30, max_value=365, value=60)
                    
                    # Muster-Erkennungs-Parameter
                    st.subheader('Muster-Erkennung')
                    
                    # Gehaltserkennung
                    st.write("Gehaltserkennung:")
                    min_gehalt = st.number_input(
                        'Mindestbetrag für Gehalt (€)',
                        min_value=0,
                        value=1000,
                        help='Eingänge über diesem Betrag werden als potentielles Gehalt erkannt'
                    )
                    min_gehalt_vorkommen = st.number_input(
                        'Minimale Anzahl Vorkommen für Gehalt',
                        min_value=1,
                        value=1,
                        help='Wie oft muss ein Gehalt mindestens vorkommen?'
                    )
                    
                    # Fixkosten-Erkennung
                    st.write("Fixkosten-Erkennung:")
                    min_fixkosten = st.number_input(
                        'Mindestbetrag für Fixkosten (€)',
                        min_value=0,
                        value=50,
                        help='Ausgänge über diesem Betrag werden als potentielle Fixkosten erkannt'
                    )
                    min_fixkosten_vorkommen = st.number_input(
                        'Minimale Anzahl Vorkommen für Fixkosten',
                        min_value=1,
                        value=2,
                        help='Wie oft muss eine Fixkostenzahlung mindestens vorkommen?'
                    )
                    max_varianz = st.slider(
                        'Maximale Varianz für Fixkosten (%)',
                        min_value=0,
                        max_value=100,
                        value=30,
                        help='Maximale prozentuale Abweichung für Fixkosten'
                    )
            except Exception as e:
                st.error(f"Fehler beim Lesen der CSV-Datei: {str(e)}")
                return
    
    if uploaded_file is not None and 'df' in locals():
        # Daten verarbeiten und Vorhersage durchführen
        if st.sidebar.button('Training starten'):
            # Fortschrittsanzeige
            progress_text = "Operation in progress. Please wait."
            progress_bar = st.progress(0, text=progress_text)
            
            try:
                # Position des Datei-Cursors zurücksetzen
                uploaded_file.seek(0)
                
                # Fortschritt: Daten laden und vorbereiten (25%)
                progress_bar.progress(0.25, text="Daten werden geladen und vorbereitet...")
                
                results = process_and_predict(
                    uploaded_file, 
                    start_balance=start_balance,
                    cutoff_value=cutoff_value,
                    days_to_predict=prediction_days,
                    date_column=date_column,
                    amount_column=amount_column,
                    min_gehalt=min_gehalt,
                    min_gehalt_vorkommen=min_gehalt_vorkommen,
                    min_fixkosten=min_fixkosten,
                    min_fixkosten_vorkommen=min_fixkosten_vorkommen,
                    max_varianz=max_varianz/100.0,  # Konvertiere Prozent zu Dezimal
                    progress_callback=lambda p, t: progress_bar.progress(0.25 + p * 0.75, text=t)
                )
                
                # Daten für Statistiken
                stats = results['statistics']
                filtering = results['filtering']
                
                # Haupt-Visualisierung
                fig = go.Figure()
                
                # Trainingsdaten (Hauptlinie)
                actual_values = np.array(results['actual_values'])
                if len(actual_values.shape) > 1:
                    actual_values = actual_values.flatten()
                
                fig.add_trace(go.Scatter(
                    x=results['dates'],
                    y=actual_values,
                    name='Historische Daten',
                    line=dict(color='blue')
                ))
                
                # Gehaltseingänge als Marker
                if 'income_patterns' in results:
                    income_df = results['income_patterns']
                    gehalt_tage = income_df[income_df['ist_gehalt']]['Tag'].values
                    gehalt_betraege = income_df[income_df['ist_gehalt']]['mean'].values
                    
                    # Finde alle Gehaltsdaten
                    gehalt_dates = []
                    gehalt_values = []
                    for date in results['dates']:
                        if date.day in gehalt_tage:
                            gehalt_dates.append(date)
                            idx = np.where(gehalt_tage == date.day)[0][0]
                            gehalt_values.append(actual_values[results['dates'].get_loc(date)])
                    
                    fig.add_trace(go.Scatter(
                        x=gehalt_dates,
                        y=gehalt_values,
                        mode='markers',
                        name='Gehaltseingänge',
                        marker=dict(
                            size=12,
                            symbol='star',
                            color='green',
                            line=dict(width=2, color='darkgreen')
                        ),
                        hovertemplate="Gehalt am %{x}<br>Betrag: %{y:.2f}€"
                    ))
                
                # Fixkosten als Marker
                if 'expense_patterns' in results:
                    expense_df = results['expense_patterns']
                    fixkosten_tage = expense_df[expense_df['ist_fixkosten']]['Tag'].values
                    fixkosten_betraege = expense_df[expense_df['ist_fixkosten']]['mean'].values
                    
                    # Finde alle Fixkosten-Daten
                    fixkosten_dates = []
                    fixkosten_values = []
                    for date in results['dates']:
                        if date.day in fixkosten_tage:
                            fixkosten_dates.append(date)
                            idx = np.where(fixkosten_tage == date.day)[0][0]
                            fixkosten_values.append(actual_values[results['dates'].get_loc(date)])
                    
                    fig.add_trace(go.Scatter(
                        x=fixkosten_dates,
                        y=fixkosten_values,
                        mode='markers',
                        name='Fixkosten',
                        marker=dict(
                            size=10,
                            symbol='x',
                            color='red',
                            line=dict(width=2, color='darkred')
                        ),
                        hovertemplate="Fixkosten am %{x}<br>Betrag: %{y:.2f}€"
                    ))
                
                # Vorhersage
                try:
                    future_predictions = np.array(results['future_prediction'])
                    if len(future_predictions.shape) > 1:
                        future_predictions = future_predictions.flatten()
                    
                    fig.add_trace(go.Scatter(
                        x=results['future_dates'],
                        y=future_predictions,
                        name='Vorhersage',
                        line=dict(color='red', dash='dot')
                    ))
                except Exception as e:
                    st.warning(f"Fehler bei der Vorhersage-Visualisierung: {str(e)}")
                
                # Layout anpassen
                fig.update_layout(
                    title='Kontostand: Historie und Vorhersage',
                    xaxis_title='Datum',
                    yaxis_title='Betrag (€)',
                    hovermode='closest',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                # Plot anzeigen
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiken in einem aufklappbaren Bereich
                with st.expander("Statistiken der Daten"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Beträge vor Filterung")
                        st.write(f"Anzahl Transaktionen: {stats['total_transactions']}")
                        st.write(f"Minimum: {stats['min_amount']:.2f}€")
                        st.write(f"Maximum: {stats['max_amount']:.2f}€")
                        st.write(f"Durchschnitt: {stats['mean_amount']:.2f}€")
                        st.write(f"Median: {stats['median_amount']:.2f}€")
                    
                    with col2:
                        st.subheader("Filterung")
                        st.write(f"Entfernte Transaktionen: {filtering['removed_transactions']}")
                        st.write(f"Verbleibende Transaktionen: {filtering['remaining_transactions']}")
                
                # Trainingsverlauf in aufklappbarem Bereich
                with st.expander("Trainingsverlauf"):
                    # Zeige den Trainingsverlauf als Plot
                    fig_loss = go.Figure()
                    
                    # Loss Werte
                    fig_loss.add_trace(go.Scatter(
                        x=results['training_df']['Epoch'],
                        y=results['training_df']['Training Loss'],
                        name='Training Loss',
                        line=dict(color='blue')
                    ))
                    fig_loss.add_trace(go.Scatter(
                        x=results['training_df']['Epoch'],
                        y=results['training_df']['Validation Loss'],
                        name='Validation Loss',
                        line=dict(color='red')
                    ))
                    
                    # Layout anpassen
                    fig_loss.update_layout(
                        title='Trainingsverlauf - Loss',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        showlegend=True
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                    
                    # MAE Plot
                    fig_mae = go.Figure()
                    fig_mae.add_trace(go.Scatter(
                        x=results['training_df']['Epoch'],
                        y=results['training_df']['Training MAE'],
                        name='Training MAE',
                        line=dict(color='green')
                    ))
                    fig_mae.add_trace(go.Scatter(
                        x=results['training_df']['Epoch'],
                        y=results['training_df']['Validation MAE'],
                        name='Validation MAE',
                        line=dict(color='orange')
                    ))
                    
                    fig_mae.update_layout(
                        title='Trainingsverlauf - MAE',
                        xaxis_title='Epoch',
                        yaxis_title='Mean Absolute Error',
                        showlegend=True
                    )
                    st.plotly_chart(fig_mae, use_container_width=True)
                    
                    # Rohdaten anzeigen
                    st.subheader("Rohdaten und Muster")
                    
                    # Original Transaktionen
                    st.write("Alle Transaktionen:")
                    df_display = results['raw_data'].copy()
                    df_display = df_display.reset_index()
                    df_display = df_display.rename(columns={'index': 'Datum'})
                    
                    # Formatierung der Spalten
                    df_display['Datum'] = pd.to_datetime(df_display['Datum']).dt.strftime('%Y-%m-%d')
                    for col in ['Betrag', 'Tatsächlicher_Betrag']:
                        if col in df_display.columns:
                            df_display[col] = df_display[col].round(2).apply(lambda x: f"{x:,.2f} €")
                    
                    st.dataframe(df_display)
                    
                    # Erkannte Muster
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Regelmäßige Eingänge:")
                        if 'income_patterns' in results:
                            income_df = results['income_patterns'].copy()
                            income_df['mean'] = income_df['mean'].round(2).apply(lambda x: f"{x:,.2f} €")
                            income_df['std'] = income_df['std'].round(2).apply(lambda x: f"{x:,.2f} €")
                            st.dataframe(income_df)
                        else:
                            st.write("Keine regelmäßigen Eingänge erkannt")
                    
                    with col2:
                        st.write("Regelmäßige Ausgänge:")
                        if 'expense_patterns' in results:
                            expense_df = results['expense_patterns'].copy()
                            expense_df['mean'] = expense_df['mean'].round(2).apply(lambda x: f"{x:,.2f} €")
                            expense_df['std'] = expense_df['std'].round(2).apply(lambda x: f"{x:,.2f} €")
                            st.dataframe(expense_df)
                        else:
                            st.write("Keine regelmäßigen Ausgänge erkannt")
                    
                    # Trainingsdaten anzeigen
                    st.subheader("Trainings-Sequenzen")
                    
                    # X_train anzeigen
                    st.write("Eingabe-Sequenzen (X_train):")
                    if 'X_train' in results:
                        # Erstelle ein DataFrame aus den Trainingsdaten
                        x_train_df = pd.DataFrame(
                            results['X_train'].reshape(-1, results['X_train'].shape[-1]),
                            columns=[f'Feature_{i+1}' for i in range(results['X_train'].shape[-1])]
                        )
                        st.dataframe(x_train_df.head(100))  # Zeige die ersten 100 Zeilen
                        st.write(f"Shape X_train: {results['X_train'].shape}")
                    
                    # y_train anzeigen
                    st.write("Zielwerte (y_train):")
                    if 'y_train' in results:
                        y_train_df = pd.DataFrame(results['y_train'], columns=['Zielwert'])
                        st.dataframe(y_train_df.head(100))  # Zeige die ersten 100 Zeilen
                        st.write(f"Shape y_train: {results['y_train'].shape}")
                    
                    # Ursprüngliche Features anzeigen
                    st.subheader("Ursprüngliche Features (vor Skalierung)")
                    if 'X_original' in results:
                        st.write("Feature-Werte vor der Skalierung:")
                        x_orig_display = results['X_original'].copy()
                        x_orig_display = x_orig_display.reset_index()
                        x_orig_display = x_orig_display.rename(columns={'index': 'Datum'})
                        
                        # Formatierung der Spalten
                        x_orig_display['Datum'] = pd.to_datetime(x_orig_display['Datum']).dt.strftime('%Y-%m-%d')
                        
                        # Formatiere numerische Spalten
                        for col in x_orig_display.columns:
                            if 'Betrag' in col or 'Kontostand' in col:
                                x_orig_display[col] = x_orig_display[col].round(2).apply(lambda x: f"{x:,.2f} €")
                            elif col != 'Datum':
                                x_orig_display[col] = x_orig_display[col].round(4)
                        
                        st.dataframe(x_orig_display.head(100))  # Zeige die ersten 100 Zeilen
                        st.write(f"Shape: {results['X_original'].shape}")
                
            except Exception as e:
                st.error(f"Fehler bei der Verarbeitung: {str(e)}")
                progress_bar.empty()

if __name__ == '__main__':
    main()