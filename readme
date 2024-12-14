# Balance Prediction Streamlit Application

Available as docker container: mrmarans/bank_prediction_app:latest

## Overview
This Streamlit application provides an advanced financial analysis and prediction tool for personal account balances. It leverages machine learning techniques to analyze transaction data, identify patterns, and forecast future account balances.

## Key Features

### Data Analysis
- Upload CSV files with financial transaction data
- Flexible column mapping for date and amount columns
- Detailed statistical analysis of transactions

### Pattern Recognition
- Automatic detection of:
  - Regular salary/income patterns
  - Fixed cost/recurring expense patterns
- Configurable thresholds for pattern identification
  - Minimum amount for salary detection
  - Minimum occurrences for pattern recognition
  - Variance tolerance for fixed costs

### Prediction Capabilities
- Balance prediction for a specified number of days
- Machine learning model to forecast future account balance
- Visualization of:
  - Historical transaction data
  - Predicted future balance
  - Salary income markers
  - Fixed cost markers

### Visualization
- Interactive Plotly charts showing:
  - Account balance history
  - Prediction line
  - Salary and fixed cost markers
- Training progress visualization
  - Loss curves
  - Mean Absolute Error (MAE) curves

## How to Use

1. **Upload Data**
   - Click "Upload CSV file" in the sidebar
   - Select appropriate date and amount columns
   - Ensure CSV is semicolon-separated

2. **Configure Parameters**
   - Set starting balance
   - Define cutoff value for transactions
   - Select prediction period
   - Configure salary and fixed cost detection parameters

3. **Start Training**
   - Click "Start Training" button
   - Wait for processing and visualization

## Detailed Outputs

### Statistics
- Transaction count
- Amount statistics (min, max, average, median)
- Filtering information

### Patterns
- Identified regular income patterns
- Recognized fixed cost patterns

### Training Insights
- Training and validation loss curves
- Mean Absolute Error progression
- Raw transaction data
- Training sequences and features

## Technical Details
- Built with Streamlit
- Uses Pandas for data manipulation
- Plotly for interactive visualizations
- Machine learning prediction model
- Supports multiple CSV encodings

## Requirements
- Python 3.7+
- Streamlit
- Pandas
- Plotly
- Custom prediction module (kontostand_vorhersage)

## Potential Use Cases
- Personal finance tracking
- Budget forecasting
- Expense pattern analysis
- Financial planning

## Limitations
- Requires clean, structured CSV data
- Prediction accuracy depends on data quality and patterns
- Works best with regular, predictable income and expenses

## Future Improvements
- Support for more data formats
- Enhanced machine learning models
- More sophisticated pattern recognition
- Additional visualization options