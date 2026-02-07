"""
Stock Fundamental Analysis Tool - Modernized 2026 Version

Fetches fundamental financial data from Yahoo Finance using yfinance library
and filters stocks based on user-defined criteria.

Original author: Ferhat (2017)
Modernized: February 2026

Usage:
    python fundamental_analysis_filter.py

Configuration:
    Modify the country_code in main() function to select input file.
    Modify StockFilter parameters to adjust screening criteria.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class Config:
    """Application configuration with paths and settings."""
    
    country_code: str
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    max_workers: int = 5  # Parallel fetch workers
    max_retries: int = 3  # Retry attempts per ticker
    batch_delay: float = 1.0  # Seconds between batches
    
    country_files: Dict[str, str] = field(default_factory=lambda: {
        'USA': 'US_Stocks.csv',
        'UK': 'UK_Stocks.csv',
        'UK_TEST': 'UK_Stocks_test.csv',  # Small test file
        'UK_PREFIX': 'UK_Stocks_prefix_test.csv',  # Test ticker prefix cleaning
        'JAP': 'JAP_Stocks.csv',
        'GER': 'Germany_Stocks.csv',
        'FRA': 'FRA_Stocks.csv',
        'ITA': 'ITA_Stocks.csv',
        'ITA2': 'ITA_Stocks2.csv',
        'HK': 'HK_Stocks.csv'
    })
    
    # Exchange suffix mapping for international tickers
    exchange_suffixes: Dict[str, str] = field(default_factory=lambda: {
        'UK': '.L',      # London Stock Exchange
        'UK_TEST': '.L', # London Stock Exchange (test)
        'UK_PREFIX': '.L',  # London Stock Exchange (prefix test)
        'JAP': '.T',     # Tokyo Stock Exchange
        'GER': '.DE',    # Deutsche Börse
        'FRA': '.PA',    # Euronext Paris
        'ITA': '.MI',    # Borsa Italiana
        'HK': '.HK',     # Hong Kong Stock Exchange
        'USA': ''        # No suffix for US stocks
    })
    
    def __post_init__(self):
        """Ensure directories exist and paths are Path objects."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def input_file(self) -> Path:
        """Get input file path for configured country."""
        filename = self.country_files.get(self.country_code)
        if not filename:
            raise ValueError(f"Unknown country code: {self.country_code}")
        return self.input_dir / filename
    
    def output_file(self, suffix: str = "") -> Path:
        """Get output file path with optional suffix."""
        filename = f"{self.country_code}_df{suffix}.csv"
        return self.output_dir / filename


@dataclass
class StockFilter:
    """Stock screening criteria configuration."""
    
    max_debt_to_equity: float = 1.0
    min_current_ratio: float = 1.0
    min_net_income_growth_5y: float = 20.0
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filter criteria to dataframe.
        
        Args:
            df: DataFrame with stock fundamental data
            
        Returns:
            Filtered DataFrame meeting all criteria
        """
        mask = (
            (df['Debt/Equity'] < self.max_debt_to_equity) &
            (df['Current Ratio'] > self.min_current_ratio) &
            (df['5-Year Net Income %'] > self.min_net_income_growth_5y)
        )
        return df.loc[mask]


class FetchError(Enum):
    """Enumeration of possible data fetch errors."""
    NETWORK = "Network error"
    RATE_LIMIT = "Rate limit exceeded"
    INVALID_TICKER = "Invalid ticker symbol"
    NO_DATA = "No data available"
    PARSING_ERROR = "Data parsing error"


@dataclass
class FetchResult:
    """Result of fetching stock data for a single ticker."""
    ticker: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[FetchError] = None
    error_message: Optional[str] = None


# ============================================================================
# Data Fetching Functions
# ============================================================================

def safe_get(data: Any, keys: List[str], default: Any = None) -> Optional[float]:
    """
    Safely extract numeric value from dict or DataFrame with fallback keys.
    
    Args:
        data: Dictionary or pandas DataFrame
        keys: List of possible keys to try (in order of preference)
        default: Default value if all keys fail
        
    Returns:
        Extracted float value or default
    """
    for key in keys:
        try:
            if isinstance(data, dict):
                if key in data:
                    value = data[key]
                    if pd.notna(value) and value is not None:
                        return float(value)
            elif isinstance(data, pd.DataFrame):
                if key in data.index:
                    value = data.loc[key].iloc[0]
                    if pd.notna(value):
                        return float(value)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            continue
    return default


def fetch_stock_data(ticker: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Fetch fundamental data for a single ticker from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        logger: Logger instance
        
    Returns:
        Dictionary containing info, financials, balance_sheet, cashflow
        or None if fetch failed
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Validate that we got meaningful data
        if not info or len(info) < 5 or 'symbol' not in info:
            logger.warning(f"{ticker}: No data returned from Yahoo Finance")
            return None
        
        return {
            'info': info,
            'financials': stock.financials,
            'balance_sheet': stock.balance_sheet,
            'cashflow': stock.cashflow
        }
    except Exception as e:
        logger.error(f"{ticker}: Error fetching data - {str(e)}")
        return None


def extract_metrics(ticker: str, stock_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all fundamental metrics from yfinance data.
    
    Args:
        ticker: Stock ticker symbol
        stock_data: Dictionary from fetch_stock_data
        
    Returns:
        Dictionary with all extracted metrics
    """
    info = stock_data['info']
    financials = stock_data['financials']
    balance_sheet = stock_data['balance_sheet']
    cashflow = stock_data['cashflow']
    
    metrics = {'Ticker': ticker}
    
    # Company name
    metrics['Company Name'] = info.get('longName', info.get('shortName', ''))
    
    # Direct from info dictionary
    metrics['EPS'] = safe_get(info, ['trailingEps', 'forwardEps'])
    metrics['Dividends'] = safe_get(info, ['dividendRate'])
    metrics['Payout Ratio'] = safe_get(info, ['payoutRatio'])
    metrics['Shares'] = safe_get(info, ['sharesOutstanding'])
    metrics['Book Value/Share'] = safe_get(info, ['bookValue'])
    metrics['Current Ratio'] = safe_get(info, ['currentRatio'])
    metrics['Quick Ratio'] = safe_get(info, ['quickRatio'])
    metrics['Debt/Equity'] = safe_get(info, ['debtToEquity'])
    metrics['Return on Assets'] = safe_get(info, ['returnOnAssets'])
    metrics['Return on Equity'] = safe_get(info, ['returnOnEquity'])
    metrics['Gross Margin'] = safe_get(info, ['grossMargins'])
    metrics['Operating Margin'] = safe_get(info, ['operatingMargins'])
    metrics['Net Margin'] = safe_get(info, ['profitMargins'])
    
    # From financial statements
    metrics['Net Income'] = safe_get(financials, ['Net Income'])
    metrics['Revenue'] = safe_get(financials, ['Total Revenue', 'Revenue'])
    metrics['Total Assets'] = safe_get(balance_sheet, ['Total Assets'])
    
    # Calculated metrics
    free_cash_flow = safe_get(cashflow, ['Free Cash Flow'])
    shares = metrics['Shares']
    if free_cash_flow is not None and shares is not None and shares > 0:
        metrics['Free Cashflow/Share'] = free_cash_flow / shares
    else:
        metrics['Free Cashflow/Share'] = None
    
    # EBT Margin (Earnings Before Tax / Revenue)
    pretax_income = safe_get(financials, ['Pretax Income', 'Income Before Tax'])
    revenue = metrics['Revenue']
    if pretax_income is not None and revenue is not None and revenue > 0:
        metrics['EBT Margin'] = pretax_income / revenue
    else:
        metrics['EBT Margin'] = None
    
    # Interest Coverage (EBIT / Interest Expense)
    ebit = safe_get(financials, ['EBIT', 'Operating Income'])
    interest_expense = safe_get(financials, ['Interest Expense'])
    if ebit is not None and interest_expense is not None and interest_expense != 0:
        metrics['Interest Coverage'] = abs(ebit / interest_expense)
    else:
        metrics['Interest Coverage'] = None
    
    # Return on Invested Capital (approximation)
    net_income = metrics['Net Income']
    total_assets = metrics['Total Assets']
    current_liabilities = safe_get(balance_sheet, ['Current Liabilities'])
    if net_income and total_assets and current_liabilities:
        invested_capital = total_assets - current_liabilities
        if invested_capital > 0:
            metrics['Return on Invested Capital'] = net_income / invested_capital
        else:
            metrics['Return on Invested Capital'] = None
    else:
        metrics['Return on Invested Capital'] = None
    
    # Growth metrics (require historical data)
    metrics['EPS Year on Year'] = None
    metrics['5-Year Net Income %'] = None
    
    if not financials.empty and len(financials.columns) >= 2:
        # EPS Year on Year growth
        for eps_key in ['Basic EPS', 'Diluted EPS']:
            if eps_key in financials.index:
                try:
                    eps_current = financials.loc[eps_key].iloc[0]
                    eps_prior = financials.loc[eps_key].iloc[1]
                    if pd.notna(eps_current) and pd.notna(eps_prior) and eps_prior != 0:
                        metrics['EPS Year on Year'] = ((eps_current - eps_prior) / abs(eps_prior)) * 100
                        break
                except (IndexError, KeyError):
                    pass
    
    if not financials.empty and len(financials.columns) >= 5:
        # 5-Year Net Income CAGR
        if 'Net Income' in financials.index:
            try:
                ni_current = financials.loc['Net Income'].iloc[0]
                ni_5y = financials.loc['Net Income'].iloc[4]
                if pd.notna(ni_current) and pd.notna(ni_5y) and ni_5y > 0:
                    metrics['5-Year Net Income %'] = (((ni_current / ni_5y) ** (1/5)) - 1) * 100
            except (IndexError, KeyError):
                pass
    
    return metrics


def fetch_with_retry(
    ticker: str,
    config: Config,
    logger: logging.Logger
) -> FetchResult:
    """
    Fetch stock data with retry logic and exponential backoff.
    
    Args:
        ticker: Stock ticker symbol
        config: Application configuration
        logger: Logger instance
        
    Returns:
        FetchResult with success status and data or error info
    """
    for attempt in range(config.max_retries):
        try:
            stock_data = fetch_stock_data(ticker, logger)
            
            if stock_data is None:
                if attempt < config.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.debug(f"{ticker}: Retry {attempt + 1} after {wait_time}s")
                    time.sleep(wait_time)
                    continue
                return FetchResult(
                    ticker=ticker,
                    success=False,
                    error=FetchError.NO_DATA,
                    error_message="No data available after retries"
                )
            
            # Extract metrics
            metrics = extract_metrics(ticker, stock_data)
            return FetchResult(ticker=ticker, success=True, data=metrics)
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for specific error types
            if '404' in error_str or 'not found' in error_str:
                return FetchResult(
                    ticker=ticker,
                    success=False,
                    error=FetchError.INVALID_TICKER,
                    error_message=f"Ticker not found: {str(e)}"
                )
            elif 'rate limit' in error_str or '429' in error_str:
                wait_time = 2 ** (attempt + 2)
                logger.warning(f"{ticker}: Rate limited, waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                if attempt < config.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.error(f"{ticker}: Error on attempt {attempt + 1}: {e}")
                    time.sleep(wait_time)
                else:
                    return FetchResult(
                        ticker=ticker,
                        success=False,
                        error=FetchError.NETWORK,
                        error_message=str(e)
                    )
    
    return FetchResult(
        ticker=ticker,
        success=False,
        error=FetchError.NETWORK,
        error_message="Max retries exceeded"
    )


def process_tickers_parallel(
    tickers: List[str],
    config: Config,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Process multiple tickers in parallel with rate limiting.
    
    Args:
        tickers: List of ticker symbols
        config: Application configuration
        logger: Logger instance
        
    Returns:
        DataFrame with fundamental data for all successfully fetched tickers
    """
    results = []
    failed_tickers = []
    
    logger.info(f"Starting parallel fetch for {len(tickers)} tickers with {config.max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(fetch_with_retry, ticker, config, logger): ticker
            for ticker in tickers
        }
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_ticker), 1):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result.success and result.data:
                    results.append(result.data)
                    logger.info(f"[OK] {ticker} ({i}/{len(tickers)})")
                else:
                    failed_tickers.append(ticker)
                    error_info = f"{result.error.value}: {result.error_message}" if result.error else "Unknown error"
                    logger.warning(f"[FAIL] {ticker} ({i}/{len(tickers)}) - {error_info}")
            except Exception as e:
                failed_tickers.append(ticker)
                logger.error(f"[FAIL] {ticker} ({i}/{len(tickers)}) - Exception: {e}")
            
            # Rate limiting between batches
            if i % config.max_workers == 0 and i < len(tickers):
                time.sleep(config.batch_delay)
    
    logger.info(f"Fetch complete: {len(results)} succeeded, {len(failed_tickers)} failed")
    if failed_tickers:
        logger.info(f"Failed tickers: {', '.join(failed_tickers[:10])}" + 
                   (f" ... and {len(failed_tickers) - 10} more" if len(failed_tickers) > 10 else ""))
    
    if not results:
        logger.error("No data fetched successfully!")
        return pd.DataFrame()
    
    return pd.DataFrame(results)


# ============================================================================
# Main Application Class
# ============================================================================

class StockAnalyzer:
    """Main application class for stock fundamental analysis."""
    
    def __init__(self, config: Config):
        """
        Initialize the analyzer.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run(self, filter_criteria: Optional[StockFilter] = None) -> pd.DataFrame:
        """
        Execute full analysis pipeline.
        
        Args:
            filter_criteria: Optional filter to apply to results
            
        Returns:
            DataFrame with all fetched stock data
        """
        # Load tickers
        tickers = self._load_tickers()
        self.logger.info(f"Loaded {len(tickers)} tickers from {self.config.country_code}")
        
        # Fetch data
        df = self._fetch_all_data(tickers)
        
        if df.empty:
            self.logger.error("No data to process!")
            return df
        
        self.logger.info(f"Successfully fetched data for {len(df)} stocks")
        
        # Calculate additional metrics
        df = self._calculate_additional_metrics(df)
        
        # Save raw data
        raw_output = self.config.output_file()
        df.to_csv(raw_output, index=False)
        self.logger.info(f"Saved raw data to {raw_output}")
        
        # Apply filters if provided
        if filter_criteria:
            df_filtered = filter_criteria.apply(df)
            filtered_output = self.config.output_file("_condition")
            df_filtered.to_csv(filtered_output, index=False)
            self.logger.info(f"Saved filtered data ({len(df_filtered)} stocks) to {filtered_output}")
            
            if len(df_filtered) > 0:
                self.logger.info(f"Filter results: {len(df_filtered)}/{len(df)} stocks passed criteria")
            else:
                self.logger.warning("No stocks passed the filter criteria!")
        
        return df
    
    def _load_tickers(self) -> List[str]:
        """
        Load ticker symbols from input CSV.
        
        Returns:
            List of ticker symbols
        """
        input_file = self.config.input_file
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        self.logger.info(f"Loading tickers from {input_file}")
        df = pd.read_csv(input_file, encoding="ISO-8859-1")
        
        if 'Ticker' not in df.columns:
            raise ValueError(f"Input file must have 'Ticker' column. Found: {df.columns.tolist()}")
        
        tickers = df['Ticker'].dropna().astype(str).tolist()
        
        # Clean tickers: remove exchange prefixes (e.g., "XLON:", "NYSE:", etc.)
        cleaned_tickers = []
        for ticker in tickers:
            # Remove exchange prefix if present (format: EXCHANGE:SYMBOL)
            if ':' in ticker:
                ticker = ticker.split(':', 1)[1]  # Take everything after first colon
            cleaned_tickers.append(ticker)
        
        # Apply exchange suffix if needed
        suffix = self.config.exchange_suffixes.get(self.config.country_code, '')
        if suffix:
            final_tickers = []
            for ticker in cleaned_tickers:
                # Add suffix if not already present
                if not ticker.endswith(suffix):
                    ticker = f"{ticker}{suffix}"
                final_tickers.append(ticker)
            self.logger.debug(f"Applied exchange suffix '{suffix}' to tickers")
            return final_tickers
        
        return cleaned_tickers
    
    def _fetch_all_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch fundamental data for all tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with fundamental data
        """
        return process_tickers_parallel(tickers, self.config, self.logger)
    
    def _calculate_additional_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional derived metrics.
        
        Args:
            df: DataFrame with fundamental data
            
        Returns:
            DataFrame with additional calculated columns
        """
        # Dividend Cover (Net Income / Dividends) - only if column doesn't exist
        if 'Dividend Cover' not in df.columns:
            df['Dividend Cover'] = None
            mask = (df['Net Income'].notna()) & (df['Dividends'].notna()) & (df['Dividends'] > 0)
            df.loc[mask, 'Dividend Cover'] = df.loc[mask, 'Net Income'] / df.loc[mask, 'Dividends']
        
        return df


# ============================================================================
# Main Entry Point
# ============================================================================

def setup_logging(config: Config) -> None:
    """
    Configure logging for the application.
    
    Args:
        config: Application configuration
    """
    log_file = config.output_dir / f"stock_analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")


def main():
    """Main entry point for the application."""
    print("=" * 70)
    print("Stock Fundamental Analysis Tool - Modernized 2026 Version")
    print("=\" * 70)
    print()
    
    # Configure application
    config = Config(country_code='UK')  # Change to: USA, UK, UK_TEST, JAP, GER, FRA, ITA, ITA2, HK
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    try:
        # Create analyzer
        analyzer = StockAnalyzer(config)
        
        # Define filter criteria
        filter_criteria = StockFilter(
            max_debt_to_equity=1.0,
            min_current_ratio=1.0,
            min_net_income_growth_5y=20.0
        )
        
        logger.info("Starting analysis...")
        logger.info(f"Country: {config.country_code}")
        logger.info(f"Filter: D/E < {filter_criteria.max_debt_to_equity}, "
                   f"Current Ratio > {filter_criteria.min_current_ratio}, "
                   f"5Y NI Growth > {filter_criteria.min_net_income_growth_5y}%")
        
        # Run analysis
        start_time = time.time()
        results = analyzer.run(filter_criteria)
        elapsed = time.time() - start_time
        
        # Summary
        print()
        print("=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Total stocks processed: {len(results)}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Output directory: {config.output_dir.absolute()}")
        print(f"Output files: {config.country_code}_df.csv (raw), {config.country_code}_df_condition.csv (filtered)")
        print()
        
        logger.info(f"Analysis complete in {elapsed:.1f}s")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n[ERROR] File not found: {e}")
        print(f"Please ensure the input file exists at: {config.input_file}")
    except Exception as e:
        logger.exception("Fatal error during analysis")
        print(f"\n[ERROR] Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()