# Fundamental Analysis Stocks

**Modernized 2026 Version** - Completely refactored with modern Python practices!

This tool fetches fundamental financial data from Yahoo Finance and filters stocks based on customizable screening criteria. It processes multiple tickers in parallel, providing comprehensive fundamental analysis metrics for stock screening.

Originally created in 2017 using the Good Morning library, this modernized version uses [yfinance](https://github.com/ranaroussi/yfinance) for reliable, up-to-date financial data from Yahoo Finance.

## Features

✨ **Modern Python 3.8+ codebase** with type hints, dataclasses, and async processing  
📊 **30+ fundamental metrics** including ratios, margins, growth rates, and cash flow  
🚀 **Parallel processing** with configurable workers and automatic retry logic  
📝 **Comprehensive logging** with detailed progress tracking and error reporting  
🔧 **Flexible filtering** with customizable screening criteria  
🌍 **International support** for US, UK, Japan, Germany, France, Italy, and Hong Kong markets

## Requirements

- Python 3.8 or higher
- pandas >= 2.0.0
- yfinance >= 0.2.30

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. **Select your country**: Modify the `country_code` in `main()` function:
   ```python
   config = Config(country_code='UK')  # Options: USA, UK, JAP, GER, FRA, ITA, ITA2, HK
   ```

2. **Customize filter criteria** (optional):
   ```python
   filter_criteria = StockFilter(
       max_debt_to_equity=1.0,         # Maximum debt-to-equity ratio
       min_current_ratio=1.0,          # Minimum current ratio
       min_net_income_growth_5y=20.0   # Minimum 5-year net income growth %
   )
   ```

3. **Run the script**:
   ```bash
   python fundamental_analysis_filter.py
   ```

### Output Files

The script generates two CSV files in the `output/` directory:

- **`{COUNTRY}_df.csv`** - Complete dataset with all fetched stocks and metrics
- **`{COUNTRY}_df_condition.csv`** - Filtered stocks meeting your screening criteria

Plus a detailed log file: `stock_analysis_YYYYMMDD_HHMMSS.log`

### Example Output

The tool fetches 30+ fundamental metrics including:

**Profitability**: Net Income, EPS, Revenue, Gross/Operating/Net Margins  
**Financial Health**: Current Ratio, Quick Ratio, Debt/Equity  
**Returns**: ROA, ROE, Return on Invested Capital  
**Growth**: EPS YoY, 5-Year Net Income Growth  
**Cash Flow**: Free Cash Flow per Share, Dividends, Payout Ratio  
**Valuation**: Book Value per Share, Shares Outstanding

## Configuration

### Adjusting Parallel Workers

For faster processing (or to reduce rate limiting):
```python
config = Config(
    country_code='UK',
    max_workers=5,        # Number of parallel threads (default: 5)
    max_retries=3,        # Retry attempts per ticker (default: 3)
    batch_delay=1.0       # Delay between batches in seconds (default: 1.0)
)
```

### Adding Custom Countries

Edit the `country_files` dictionary in the `Config` dataclass:
```python
country_files: Dict[str, str] = field(default_factory=lambda: {
    'USA': 'US_Stocks.csv',
    'UK': 'UK_Stocks.csv',
    # Add your custom country here
})
```

## What's New in 2026 Version

🔄 **Migration from Good Morning to yfinance** - More reliable and actively maintained  
🏗️ **Complete refactor** - Object-oriented design with proper separation of concerns  
⚡ **Parallel processing** - 5x faster with concurrent ticker fetching  
🛡️ **Robust error handling** - Exponential backoff, retry logic, detailed error reporting  
📊 **Better data validation** - Safe extraction with fallback keys for missing data  
🎯 **Type safety** - Full type hints for better IDE support and fewer bugs  
📝 **Production logging** - Structured logging with file output and progress tracking  
🔧 **Configurable** - Dataclass-based configuration for easy customization  
📦 **Modern Python** - Uses pathlib, f-strings, dataclasses, enums (Python 3.8+)

## Architecture

```
StockAnalyzer (main class)
    ↓
Config (dataclass) - Paths, settings, country mappings
    ↓
process_tickers_parallel() - Parallel fetching with ThreadPoolExecutor
    ↓
fetch_with_retry() - Retry logic with exponential backoff
    ↓
fetch_stock_data() - yfinance API calls
    ↓
extract_metrics() - Extract & calculate 30+ metrics
    ↓
StockFilter (dataclass) - Apply screening criteria
```

## Troubleshooting

**Import Error**: Make sure dependencies are installed: `pip install -r requirements.txt`

**No data returned**: Some tickers may not have financial data available on Yahoo Finance. Check the log file for details.

**Rate limiting**: If you see many rate limit errors, reduce `max_workers` or increase `batch_delay` in the Config.

**Missing 5-Year metrics**: yfinance typically provides 4-5 years of data. Some stocks may not have enough history.

**International tickers**: The script automatically adds exchange suffixes (e.g., `.L` for London). Verify your input CSV has the base ticker symbols without suffixes.

## Credits

**Original Author**: Ferhat (2017) - Initial concept using Good Morning library  
**Modernization**: 2026 - Complete refactor for modern Python with yfinance

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Issues and pull requests welcome! This project aims to help retail investors with stock screening using fundamental analysis.
