# 📊 Company 10K Financial Analyzer

A powerful Streamlit application that analyzes financial data from SEC 10-K filings using AI-powered insights and advanced data visualization.

## ✨ Features

- **📥 Automated Filing Downloads**: Download SEC 10-K filings for any public company
- **🔍 Advanced Financial Extraction**: Extract key financial metrics using enhanced regex patterns
- **🤖 AI-Powered Analysis**: Generate insights using Google's Gemini AI
- **📈 Interactive Visualizations**: Beautiful charts and graphs for financial trends
- **❓ Interactive Q&A**: Ask questions about financial data and get AI-powered answers
- **📊 Data Export**: View financial data in organized tables
- **⚙️ Configurable Settings**: Easy configuration through environment variables

## 🚀 Key Improvements

### Code Structure
- **Object-Oriented Design**: Clean class-based architecture
- **Type Hints**: Full type annotation for better code maintainability
- **Error Handling**: Comprehensive error handling and logging
- **Configuration Management**: Centralized configuration using dataclasses

### Enhanced Functionality
- **Better Financial Extraction**: Improved regex patterns for more accurate data extraction
- **Advanced Visualizations**: Multiple chart types with better styling
- **AI Integration**: Context-aware question answering
- **Data Validation**: Input validation for ticker symbols and dates
- **Progress Indicators**: User-friendly loading states and progress bars

### User Experience
- **Modern UI**: Clean, responsive interface with emojis and better styling
- **Sidebar Navigation**: Organized settings and controls
- **Real-time Feedback**: Success/error messages and status updates
- **Interactive Elements**: Date pickers, dropdowns, and dynamic content

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Company-10K-Analyzer-main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key_here"
   export USER_AGENT="Your-App-Name"
   export USER_EMAIL="your_email@example.com"
   ```

## 🎯 Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Enter company information**:
   - Enter a valid ticker symbol (e.g., AAPL, MSFT, GOOGL)
   - Select date range for analysis
   - Click "Analyze Financial Data"

3. **View results**:
   - Financial data table with extracted metrics
   - Interactive visualizations
   - AI-generated insights
   - Ask custom questions about the data

## 📋 Supported Financial Metrics

- **Revenue**: Total revenue figures
- **Net Income**: Net profit/loss
- **Total Assets**: Company's total assets
- **Total Liabilities**: Company's total liabilities
- **Cash Flow**: Operating cash flow
- **Debt**: Total debt obligations

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini AI API key | Required |
| `USER_AGENT` | User agent for SEC downloads | "Company-10K-Analyzer" |
| `USER_EMAIL` | Email for SEC downloads | "akshatnahata05@gmail.com" |

### Date Range
- **Minimum**: 1990-01-01
- **Maximum**: Current date
- **Format**: YYYY-MM-DD

## 📊 Data Processing

### Financial Data Extraction
The application uses enhanced regex patterns to extract financial information from 10-K filings:

```python
# Example patterns
revenue_patterns = [
    r'(?:total\s+)?revenue(?:\s+in\s+millions?)?\s*[:\-]?\s*\$?([\d,]+\.?\d*)',
    r'revenue\s*[:\-]?\s*\$?([\d,]+\.?\d*)',
    r'net\s+revenue\s*[:\-]?\s*\$?([\d,]+\.?\d*)'
]
```

### AI Analysis
The application uses Google's Gemini AI to:
- Analyze financial trends
- Generate insights
- Answer user questions
- Provide recommendations

## 🎨 Visualizations

The application creates multiple types of charts:

1. **Line Charts**: Show trends over time
2. **Scatter Plots**: Display data distribution
3. **Bar Charts**: Compare metrics across years
4. **Financial Metrics Dashboard**: Comprehensive overview

## 🔍 Error Handling

The application includes comprehensive error handling:

- **API Key Validation**: Checks for required API keys
- **Ticker Validation**: Validates ticker symbol format
- **Date Validation**: Ensures valid date ranges
- **File Processing**: Handles corrupted or missing files
- **Network Errors**: Manages download failures

## 📈 Performance Optimizations

- **Lazy Loading**: Load data only when needed
- **Caching**: Cache processed financial data
- **Parallel Processing**: Process multiple files concurrently
- **Memory Management**: Efficient data structures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GEMINI_API_KEY` is set correctly
2. **Download Failures**: Check internet connection and SEC server status
3. **No Data Found**: Verify ticker symbol and date range
4. **Memory Issues**: Reduce date range for large companies

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the error logs
- Open an issue on GitHub

---

**Note**: This application requires an active internet connection and a valid Google Gemini AI API key to function properly.
