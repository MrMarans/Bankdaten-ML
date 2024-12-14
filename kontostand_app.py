import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from kontostand_vorhersage import process_and_predict
import numpy as np

def main():
    st.title('Balance Prediction')
    
    # Sidebar for upload and parameters
    with st.sidebar:
        st.header('Data and Parameters')
        
        # File Upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            # Read CSV file for column selection
            try:
                # Try different encodings
                encodings = ['ISO-8859-1', 'utf-8', 'cp1252']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, nrows=1, encoding=encoding, sep=';')
                        break
                    except:
                        continue
                
                if df is not None:
                    # Column selection
                    st.subheader('Column Mapping')
                    
                    # Clean column names
                    columns = [col.strip('"\' ') for col in df.columns]
                    
                    date_column = st.selectbox(
                        'Select date column',
                        options=columns,
                        index=columns.index('Buchungstag') if 'Buchungstag' in columns else 0
                    )
                    amount_column = st.selectbox(
                        'Select amount column',
                        options=columns,
                        index=columns.index('Betrag') if 'Betrag' in columns else 0
                    )
                    
                    # Basic parameters
                    st.subheader('Basic Settings')
                    start_balance = st.number_input('Starting amount (€)', value=0, help="Starting amount of your account")
                    cutoff_value = st.number_input('Cutoff value for amounts (€)', value=5000, help="Amounts above this value will be ignored")
                    prediction_days = st.number_input('Number of days for prediction', 
                                                    min_value=30, max_value=365, value=60, help="Number of days to predict")
                    
                    # Pattern recognition parameters
                    st.subheader('Pattern Recognition')
                    
                    # Salary detection
                    st.write("Salary Detection:")
                    min_gehalt = st.number_input(
                        'Minimum amount for salary (€)',
                        min_value=0,
                        value=2100,
                        help='Incoming transfers above this amount will be recognized as potential salary'
                    )
                    min_gehalt_vorkommen = st.number_input(
                        'Minimum occurrences for salary',
                        min_value=1,
                        value=1,
                        help='How many times must a salary occur at least?'
                    )
                    
                    # Fixed costs detection
                    st.write("Fixed Costs Detection:")
                    min_fixkosten = st.number_input(
                        'Minimum amount for fixed costs (€)',
                        min_value=0,
                        value=400,
                        help='Outgoing transfers above this amount will be recognized as potential fixed costs'
                    )
                    min_fixkosten_vorkommen = st.number_input(
                        'Minimum occurrences for fixed costs',
                        min_value=1,
                        value=1,
                        help='How many times must a fixed cost payment occur at least?'
                    )
                    max_varianz = st.slider(
                        'Maximum variance for fixed costs (%)',
                        min_value=0,
                        max_value=100,
                        value=30,
                        help='Maximum percentage deviation for fixed costs'
                    )
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return
    
    if uploaded_file is not None and 'df' in locals():
        # Process data and make prediction
        if st.sidebar.button('Start Training'):
            # Progress indicator
            progress_text = "Operation in progress. Please wait."
            progress_bar = st.progress(0.15, text=progress_text)
            
            try:
                # Reset file cursor position
                uploaded_file.seek(0)
                
                # Progress: Load and prepare data (25%)
                progress_bar.progress(0.0, text="Loading and preparing data...")
                
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
                    max_varianz=max_varianz/100.0,  # Convert percent to decimal
                    progress_callback=lambda p, t: progress_bar.progress(0.25 + p * 0.75, text=t)
                )
                
                # Data for statistics
                stats = results['statistics']
                filtering = results['filtering']
                
                # Main visualization
                fig = go.Figure()
                
                # Training data (main line)
                actual_values = np.array(results['actual_values'])
                if len(actual_values.shape) > 1:
                    actual_values = actual_values.flatten()
                
                fig.add_trace(go.Scatter(
                    x=results['dates'],
                    y=actual_values,
                    name='Historical Data',
                    line=dict(color='blue')
                ))
                
                # Salary income as markers
                if 'income_patterns' in results:
                    income_df = results['income_patterns']
                    gehalt_tage = income_df[income_df['ist_gehalt']]['Tag'].values
                    gehalt_betraege = income_df[income_df['ist_gehalt']]['mean'].values
                    
                    # Find all salary data
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
                        name='Salary Incomes',
                        marker=dict(
                            size=12,
                            symbol='star',
                            color='green',
                            line=dict(width=2, color='darkgreen')
                        ),
                        hovertemplate="Salary on %{x}<br>Amount: %{y:.2f}€"
                    ))
                
                # Fixed costs as markers
                if 'expense_patterns' in results:
                    expense_df = results['expense_patterns']
                    fixkosten_tage = expense_df[expense_df['ist_fixkosten']]['Tag'].values
                    fixkosten_betraege = expense_df[expense_df['ist_fixkosten']]['mean'].values
                    
                    # Find all fixed cost data
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
                        name='Fixed Costs',
                        marker=dict(
                            size=10,
                            symbol='x',
                            color='red',
                            line=dict(width=2, color='darkred')
                        ),
                        hovertemplate="Fixed Costs on %{x}<br>Amount: %{y:.2f}€"
                    ))
                
                # Prediction
                try:
                    future_predictions = np.array(results['future_prediction'])
                    if len(future_predictions.shape) > 1:
                        future_predictions = future_predictions.flatten()
                    
                    fig.add_trace(go.Scatter(
                        x=results['future_dates'],
                        y=future_predictions,
                        name='Prediction',
                        line=dict(color='red', dash='dot')
                    ))
                except Exception as e:
                    st.warning(f"Error in prediction visualization: {str(e)}")
                
                # Adjust layout
                fig.update_layout(
                    title='Account Balance: History and Prediction',
                    xaxis_title='Date',
                    yaxis_title='Amount (€)',
                    hovermode='closest',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                # Display plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics in an expandable section
                with st.expander("Data Statistics"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Amounts Before Filtering")
                        st.write(f"Number of Transactions: {stats['total_transactions']}")
                        st.write(f"Minimum: {stats['min_amount']:.2f}€")
                        st.write(f"Maximum: {stats['max_amount']:.2f}€")
                        st.write(f"Average: {stats['mean_amount']:.2f}€")
                        st.write(f"Median: {stats['median_amount']:.2f}€")
                    
                    with col2:
                        st.subheader("Filtering")
                        st.write(f"Removed Transactions: {filtering['removed_transactions']}")
                        st.write(f"Remaining Transactions: {filtering['remaining_transactions']}")
                
                # Training progress in an expandable section
                with st.expander("Training Progress"):
                    # Show training progress as a plot
                    fig_loss = go.Figure()
                    
                    # Loss values
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
                    
                    # Adjust layout
                    fig_loss.update_layout(
                        title='Training Progress - Loss',
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
                        title='Training Progress - MAE',
                        xaxis_title='Epoch',
                        yaxis_title='Mean Absolute Error',
                        showlegend=True
                    )
                    st.plotly_chart(fig_mae, use_container_width=True)
                    
                    # Show raw data
                    st.subheader("Raw Data and Patterns")
                    
                    # Original transactions
                    st.write("All Transactions:")
                    df_display = results['raw_data'].copy()
                    df_display = df_display.reset_index()
                    df_display = df_display.rename(columns={'index': 'Date'})
                    
                    # Column formatting
                    df_display['Date'] = pd.to_datetime(df_display['Date']).dt.strftime('%Y-%m-%d')
                    for col in ['Amount', 'Actual_Amount']:
                        if col in df_display.columns:
                            df_display[col] = df_display[col].round(2).apply(lambda x: f"{x:,.2f} €")
                    
                    st.dataframe(df_display)
                    
                    # Recognized patterns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Regular Incomes:")
                        if 'income_patterns' in results:
                            income_df = results['income_patterns'].copy()
                            income_df['mean'] = income_df['mean'].round(2).apply(lambda x: f"{x:,.2f} €")
                            income_df['std'] = income_df['std'].round(2).apply(lambda x: f"{x:,.2f} €")
                            st.dataframe(income_df)
                        else:
                            st.write("No regular incomes detected")
                    
                    with col2:
                        st.write("Regular Expenses:")
                        if 'expense_patterns' in results:
                            expense_df = results['expense_patterns'].copy()
                            expense_df['mean'] = expense_df['mean'].round(2).apply(lambda x: f"{x:,.2f} €")
                            expense_df['std'] = expense_df['std'].round(2).apply(lambda x: f"{x:,.2f} €")
                            st.dataframe(expense_df)
                        else:
                            st.write("No regular expenses detected")
                    
                    # Training data display
                    st.subheader("Training Sequences")
                    
                    # Display X_train
                    st.write("Input Sequences (X_train):")
                    if 'X_train' in results:
                        # Create DataFrame from training data
                        x_train_df = pd.DataFrame(
                            results['X_train'].reshape(-1, results['X_train'].shape[-1]),
                            columns=[f'Feature_{i+1}' for i in range(results['X_train'].shape[-1])]
                        )
                        st.dataframe(x_train_df.head(100))  # Show first 100 rows
                        st.write(f"Shape X_train: {results['X_train'].shape}")
                    
                    # Display y_train   
                    st.write("Target Values (y_train):")
                    if 'y_train' in results:
                        y_train_df = pd.DataFrame(results['y_train'], columns=['Target Value'])
                        st.dataframe(y_train_df.head(100))  # Show first 100 rows
                        st.write(f"Shape y_train: {results['y_train'].shape}")
                    
                    # Display original features
                    st.subheader("Original Features (before scaling)")
                    if 'X_original' in results:
                        st.write("Feature values before scaling:")
                        x_orig_display = results['X_original'].copy()
                        x_orig_display = x_orig_display.reset_index()
                        x_orig_display = x_orig_display.rename(columns={'index': 'Date'})
                        
                        # Column formatting
                        x_orig_display['Date'] = pd.to_datetime(x_orig_display['Date']).dt.strftime('%Y-%m-%d')
                        
                        # Format numerical columns
                        for col in x_orig_display.columns:
                            if 'Amount' in col or 'Balance' in col:
                                x_orig_display[col] = x_orig_display[col].round(2).apply(lambda x: f"{x:,.2f} €")
                            elif col != 'Date':
                                x_orig_display[col] = x_orig_display[col].round(4)
                        
                        st.dataframe(x_orig_display.head(100))  # Show first 100 rows
                        st.write(f"Shape: {results['X_original'].shape}")
                
            except Exception as e:
                st.error(f"Error processing: {str(e)}")
                progress_bar.empty()

if __name__ == '__main__':
    main()