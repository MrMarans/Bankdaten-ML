import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from kontostand_vorhersage import process_and_predict

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
                    
                    # Parameter
                    st.subheader('Parameter')
                    start_balance = st.number_input('Startbetrag (€)', value=937)
                    cutoff_value = st.number_input('Cutoff-Wert für Beträge (€)', value=5000)
                    prediction_days = st.number_input('Anzahl Tage für Vorhersage', 
                                                    min_value=30, max_value=365, value=60)
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
                    progress_callback=lambda p, t: progress_bar.progress(0.25 + p * 0.75, text=t)
                )
                
                # Daten für Statistiken
                stats = results['statistics']
                filtering = results['filtering']
                
                # Haupt-Visualisierung
                fig = go.Figure()
                
                # Trainingsdaten
                fig.add_trace(go.Scatter(
                    x=results['dates'],
                    y=results['actual_values'],
                    name='Historische Daten',
                    line=dict(color='blue')
                ))
                
                # Vorhersage
                fig.add_trace(go.Scatter(
                    x=results['future_dates'],
                    y=results['future_prediction'].flatten(),
                    name='Vorhersage',
                    line=dict(color='red', dash='dot')
                ))
                
                # Layout anpassen
                fig.update_layout(
                    title='Kontostand: Historie und Vorhersage',
                    xaxis_title='Datum',
                    yaxis_title='Betrag (€)',
                    hovermode='x unified',
                    showlegend=True
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
                    st.subheader("Rohdaten")
                    df_display = results['raw_data'].copy()
                    
                    # Formatierung der Spalten
                    df_display['Datum'] = pd.to_datetime(df_display['Datum']).dt.strftime('%Y-%m-%d')
                    for col in ['Betrag', 'Tatsächlicher_Betrag']:
                        if col in df_display.columns:
                            df_display[col] = df_display[col].round(2).apply(lambda x: f"{x:,.2f} €")
                    
                    print(df_display)
                
            except Exception as e:
                st.error(f"Fehler bei der Verarbeitung: {str(e)}")
                progress_bar.empty()

if __name__ == '__main__':
    main()