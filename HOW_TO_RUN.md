# 🚀 How to Run the Company 10K Analyzer

Follow these steps to set up and run the application on your local machine.

---

## 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for downloading filings and using AI features)

---

## 2. Clone the Repository

```bash
git clone <repository-url>
cd Company-10K-Analyzer-main
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Set Up Environment Variables

The application requires a Google Gemini AI API key and your email for SEC downloads.

On macOS/Linux:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export USER_AGENT="Your-App-Name"
export USER_EMAIL="your_email@example.com"
```

On Windows (Command Prompt):
```cmd
set GEMINI_API_KEY=your_gemini_api_key_here
set USER_AGENT=Your-App-Name
set USER_EMAIL=your_email@example.com
```

---

## 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501).

---

## 6. Using the App

1. **Enter the company ticker symbol** (e.g., AAPL, MSFT, GOOGL).
2. **Select the date range** for the filings you want to analyze.
3. Click **"Analyze Financial Data"**.
4. View:
   - Download status and progress
   - Extracted financial data in table form
   - Interactive visualizations
   - AI-generated insights and answers to your questions
5. Use the **Q&A section** to ask custom questions about the financial data (e.g., "What are the revenue trends?", "How has net income changed?").

---

## 7. Example Queries

- "What are the revenue trends for AAPL between 2018 and 2023?"
- "How has net income changed for MSFT over the last 5 years?"
- "Is there any year where total liabilities exceeded total assets for GOOGL?"
- "What are the key financial risks for TSLA based on the 10-K filings?"

---

## 8. Troubleshooting

- **API Key Error:** Ensure `GEMINI_API_KEY` is set and valid.
- **No Data Found:** Check ticker symbol and date range.
- **Download Failures:** Verify your internet connection and SEC server status.
- **Other Issues:** See logs in the terminal or refer to the README for more help.

---

**Enjoy exploring company financials with AI-powered insights!** 