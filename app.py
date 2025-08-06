import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
import google.generativeai as genai
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration class
@dataclass
class Config:
    """Configuration settings for the application."""
    api_key: str
    user_agent: str = "Company-10K-Analyzer"
    user_email: str = "akshatnahata05@gmail.com"
    base_dir: str = os.getcwd()
    filings_dir: str = "sec-edgar-filings"
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            user_agent=os.getenv("USER_AGENT", "Company-10K-Analyzer"),
            user_email=os.getenv("USER_EMAIL", "akshatnahata05@gmail.com")
        )

# Financial data structure
@dataclass
class FinancialData:
    """Structure for financial data."""
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    cash_flow: Optional[float] = None
    debt: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

class FinancialAnalyzer:
    """Main class for financial analysis operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_gemini()
        self.downloader = Downloader(config.user_agent, config.user_email)
        
    def setup_gemini(self):
        """Setup Gemini AI configuration."""
        if not self.config.api_key:
            st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
            st.stop()
        genai.configure(api_key=self.config.api_key)
    
    def validate_date_format(self, date_str: str) -> bool:
        """Validate date format YYYY-MM-DD."""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate ticker symbol."""
        if not ticker or len(ticker.strip()) == 0:
            return False
        return bool(re.match(r'^[A-Z]{1,5}$', ticker.upper()))
    
    def download_10k_filings(self, ticker: str, after_date: str, before_date: str) -> bool:
        """Download 10-K filings with improved error handling."""
        try:
            with st.spinner(f"📥 Downloading 10-K filings for {ticker}..."):
                self.downloader.get("10-K", ticker, after=after_date, before=before_date)
                st.success(f"✅ Successfully downloaded 10-K filings for {ticker}")
                return True
        except Exception as e:
            st.error(f"❌ Error downloading 10-K filings for {ticker}: {str(e)}")
            logger.error(f"Download error for {ticker}: {e}")
            return False
    
    def clean_text(self, text: str) -> str:
        """Clean text content with improved error handling."""
        try:
            # Try html.parser first
            cleaned_text = BeautifulSoup(text, 'html.parser').get_text()
        except Exception:
            try:
                # Fallback to lxml
                cleaned_text = BeautifulSoup(text, 'lxml').get_text()
            except Exception:
                # Last resort: basic cleaning
                cleaned_text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep important punctuation
        cleaned_text = re.sub(r'[^\w\s.,;:()$%]', '', cleaned_text)
        return cleaned_text.strip()
    
    def extract_financial_info(self, text: str) -> FinancialData:
        """Extract financial information with enhanced patterns."""
        financial_data = FinancialData()
        
        # Enhanced regex patterns for financial metrics
        patterns = {
            'revenue': [
                r'(?:total\s+)?revenue(?:\s+in\s+millions?)?\s*[:\-]?\s*\$?([\d,]+\.?\d*)',
                r'revenue\s*[:\-]?\s*\$?([\d,]+\.?\d*)',
                r'net\s+revenue\s*[:\-]?\s*\$?([\d,]+\.?\d*)'
            ],
            'net_income': [
                r'net\s+income(?:\s+in\s+millions?)?\s*[:\-]?\s*\$?([\d,]+\.?\d*)',
                r'net\s+earnings\s*[:\-]?\s*\$?([\d,]+\.?\d*)',
                r'net\s+profit\s*[:\-]?\s*\$?([\d,]+\.?\d*)'
            ],
            'total_assets': [
                r'total\s+assets(?:\s+in\s+millions?)?\s*[:\-]?\s*\$?([\d,]+\.?\d*)',
                r'total\s+assets\s*[:\-]?\s*\$?([\d,]+\.?\d*)'
            ],
            'total_liabilities': [
                r'total\s+liabilities(?:\s+in\s+millions?)?\s*[:\-]?\s*\$?([\d,]+\.?\d*)',
                r'total\s+debt\s*[:\-]?\s*\$?([\d,]+\.?\d*)'
            ],
            'cash_flow': [
                r'operating\s+cash\s+flow\s*[:\-]?\s*\$?([\d,]+\.?\d*)',
                r'cash\s+flow\s+from\s+operations\s*[:\-]?\s*\$?([\d,]+\.?\d*)'
            ],
            'debt': [
                r'total\s+debt\s*[:\-]?\s*\$?([\d,]+\.?\d*)',
                r'long\s+term\s+debt\s*[:\-]?\s*\$?([\d,]+\.?\d*)'
            ]
        }
        
        cleaned_text = self.clean_text(text)
        
        for metric, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1).replace(',', ''))
                        setattr(financial_data, metric, value)
                        break
                    except ValueError:
                        continue
        
        return financial_data
    
    def analyze_financial_data(self, financial_data: FinancialData, ticker: str) -> str:
        """Analyze financial data using Gemini AI."""
        try:
            prompt = f"""
            Analyze the following financial data for {ticker}:
            {json.dumps(financial_data.to_dict(), indent=2)}
            
            Provide insights on:
            1. Key financial trends
            2. Strengths and weaknesses
            3. Potential risks or opportunities
            4. Overall financial health assessment
            
            Format your response in a clear, structured manner.
            """
            
            response = genai.generate_text(prompt=prompt)
            return response.result
        except Exception as e:
            logger.error(f"Error in financial analysis: {e}")
            return f"Analysis error: {str(e)}"
    
    def process_filing_data(self, ticker: str) -> Dict[int, Dict]:
        """Process filing data with improved structure."""
        processed_data = {}
        base_path = Path(self.config.base_dir) / self.config.filings_dir / ticker / "10-K"
        
        if not base_path.exists():
            st.warning(f"⚠️ No filing data found for {ticker}")
            return processed_data
        
        with st.spinner("🔍 Processing filing data..."):
            for company_dir in base_path.iterdir():
                if not company_dir.is_dir():
                    continue
                
                # Extract year from directory name
                year_match = re.search(r'-(\d{2})-', company_dir.name)
                if not year_match:
                    continue
                
                year_suffix = year_match.group(1)
                year = 2000 + int(year_suffix) if int(year_suffix) <= 23 else 1900 + int(year_suffix)
                
                # Find and process text files
                for file_path in company_dir.glob("*.txt"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                        
                        financial_info = self.extract_financial_info(text_content)
                        insights = self.analyze_financial_data(financial_info, ticker)
                        
                        processed_data[year] = {
                            'financial_info': financial_info,
                            'insights': insights,
                            'file_path': str(file_path)
                        }
                        break  # Process only first text file
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        continue
        
        return processed_data
    
    def create_financial_table(self, processed_data: Dict[int, Dict], ticker: str) -> pd.DataFrame:
        """Create a formatted financial data table."""
        if not processed_data:
            return pd.DataFrame()
        
        table_data = []
        for year, data in sorted(processed_data.items()):
            financial_info = data['financial_info']
            row = {
                'Year': year,
                'Revenue ($M)': financial_info.revenue,
                'Net Income ($M)': financial_info.net_income,
                'Total Assets ($M)': financial_info.total_assets,
                'Total Liabilities ($M)': financial_info.total_liabilities,
                'Cash Flow ($M)': financial_info.cash_flow,
                'Debt ($M)': financial_info.debt,
                'Insights': data['insights'][:200] + "..." if len(data['insights']) > 200 else data['insights']
            }
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def visualize_financial_metrics(self, processed_data: Dict[int, Dict], ticker: str):
        """Create enhanced financial visualizations."""
        if not processed_data:
            st.warning(f"⚠️ No data available for visualization for {ticker}")
            return
        
        years = sorted(processed_data.keys())
        
        # Create metrics for visualization
        metrics_data = {
            'Revenue': [processed_data[year]['financial_info'].revenue for year in years],
            'Net Income': [processed_data[year]['financial_info'].net_income for year in years],
            'Total Assets': [processed_data[year]['financial_info'].total_assets for year in years],
            'Total Liabilities': [processed_data[year]['financial_info'].total_liabilities for year in years]
        }
        
        # Filter out None values for plotting
        valid_years = []
        valid_metrics = {k: [] for k in metrics_data.keys()}
        
        for i, year in enumerate(years):
            has_valid_data = any(metrics_data[metric][i] is not None for metric in metrics_data.keys())
            if has_valid_data:
                valid_years.append(year)
                for metric in metrics_data.keys():
                    valid_metrics[metric].append(metrics_data[metric][i])
        
        if not valid_years:
            st.warning("⚠️ No valid financial data found for visualization")
            return
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Financial Analysis for {ticker}", fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Line plots for each metric
        for i, (metric, values) in enumerate(valid_metrics.items()):
            row, col = i // 2, i % 2
            axes[row, col].plot(valid_years, values, marker='o', color=colors[i], linewidth=2, markersize=6)
            axes[row, col].set_title(f"{metric} Over Time", fontweight='bold')
            axes[row, col].set_xlabel("Year")
            axes[row, col].set_ylabel(f"{metric} ($M)")
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additional insights
        self.display_financial_insights(processed_data, ticker)
    
    def display_financial_insights(self, processed_data: Dict[int, Dict], ticker: str):
        """Display financial insights and analysis."""
        st.subheader("📊 Financial Insights")
        
        # Calculate trends
        years = sorted(processed_data.keys())
        if len(years) >= 2:
            first_year = years[0]
            last_year = years[-1]
            
            first_data = processed_data[first_year]['financial_info']
            last_data = processed_data[last_year]['financial_info']
            
            if first_data.revenue and last_data.revenue:
                revenue_growth = ((last_data.revenue - first_data.revenue) / first_data.revenue) * 100
                st.metric("Revenue Growth", f"{revenue_growth:.1f}%")
            
            if first_data.net_income and last_data.net_income:
                income_growth = ((last_data.net_income - first_data.net_income) / first_data.net_income) * 100
                st.metric("Net Income Growth", f"{income_growth:.1f}%")
    
    def answer_question(self, question: str, ticker: str, processed_data: Dict[int, Dict]) -> str:
        """Answer user questions using Gemini AI with context."""
        try:
            # Create context from processed data
            context = f"Financial data for {ticker}:\n"
            for year, data in sorted(processed_data.items()):
                context += f"Year {year}: {json.dumps(data['financial_info'].to_dict())}\n"
            
            prompt = f"""
            Context: {context}
            
            Question: {question}
            
            Please provide a detailed, accurate answer based on the financial data provided.
            Focus on specific numbers and trends when relevant.
            """
            
            response = genai.generate_text(prompt=prompt)
            return response.result
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error processing question: {str(e)}"

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Company 10K Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize configuration
    config = Config.from_env()
    analyzer = FinancialAnalyzer(config)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    
    # Main content
    st.title("📊 Company 10K Financial Analyzer")
    st.markdown("Analyze financial data from SEC 10-K filings using AI-powered insights.")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input(
            "🏢 Company Ticker Symbol",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            help="Enter the stock ticker symbol (1-5 characters)"
        ).upper()
    
    with col2:
        st.markdown("📅 Date Range")
        after_date = st.date_input(
            "From Date",
            value=date(2018, 1, 1),
            min_value=date(1990, 1, 1),
            max_value=date.today()
        )
        before_date = st.date_input(
            "To Date",
            value=date(2023, 12, 31),
            min_value=date(1990, 1, 1),
            max_value=date.today()
        )
    
    # Validation
    if not analyzer.validate_ticker(ticker):
        st.error("❌ Please enter a valid ticker symbol (1-5 letters)")
        return
    
    if after_date >= before_date:
        st.error("❌ Start date must be before end date")
        return
    
    # Process button
    if st.button("🚀 Analyze Financial Data", type="primary"):
        if not ticker:
            st.error("❌ Please enter a ticker symbol")
            return
        
        # Download filings
        if analyzer.download_10k_filings(ticker, after_date.strftime("%Y-%m-%d"), before_date.strftime("%Y-%m-%d")):
            # Process data
            processed_data = analyzer.process_filing_data(ticker)
            
            if processed_data:
                # Display results
                st.success(f"✅ Analysis complete for {ticker}")
                
                # Create and display table
                df = analyzer.create_financial_table(processed_data, ticker)
                if not df.empty:
                    st.subheader("📋 Financial Data Summary")
                    st.dataframe(df, use_container_width=True)
                    
                    # Visualizations
                    st.subheader("📈 Financial Visualizations")
                    analyzer.visualize_financial_metrics(processed_data, ticker)
                    
                    # AI Analysis
                    st.subheader("🤖 AI-Powered Analysis")
                    overall_analysis = analyzer.analyze_financial_data(
                        FinancialData(), ticker
                    )
                    st.write(overall_analysis)
                    
                    # Interactive Q&A
                    st.subheader("❓ Ask Questions")
                    question = st.text_input(
                        "Ask a question about the financial data:",
                        placeholder="e.g., What are the revenue trends? How has profitability changed?"
                    )
                    
                    if question:
                        with st.spinner("🤔 Analyzing your question..."):
                            answer = analyzer.answer_question(question, ticker, processed_data)
                            st.write("**Answer:**")
                            st.write(answer)
            else:
                st.warning(f"⚠️ No financial data found for {ticker} in the specified date range")
        else:
            st.error(f"❌ Failed to download filings for {ticker}")

if __name__ == "__main__":
    main()
