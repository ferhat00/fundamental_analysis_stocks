Run a stock fundamental analysis for a given country/market.

## Instructions

The user wants to run the stock fundamental analysis tool. Follow these steps:

1. **Determine the country code** from the user's message (if provided as $ARGUMENTS). Valid codes are:
   - `USA` — US stocks
   - `UK` — London Stock Exchange
   - `JAP` — Tokyo Stock Exchange
   - `GER` — Deutsche Börse
   - `FRA` — Euronext Paris
   - `ITA` / `ITA2` — Borsa Italiana
   - `HK` — Hong Kong Stock Exchange

   If no code was given, ask the user which market they want to analyze before proceeding.

2. **Update the country code** in `fundamental_analysis_filter.py` at the `Config(country_code=...)` line inside `main()` to match the requested market.

3. **Run the analysis** using the virtual environment:
   ```
   .venv/Scripts/python.exe fundamental_analysis_filter.py
   ```

4. **Report the results** by reading the generated output CSV files in the `output/` directory:
   - `<COUNTRY>_df.csv` — all fetched stocks
   - `<COUNTRY>_df_condition.csv` — stocks that passed the filter

   Summarize:
   - How many stocks were fetched successfully
   - How many passed the filter criteria (D/E < 1.0, Current Ratio > 1.0, 5Y Net Income Growth > 20%)
   - List the top passing stocks (Ticker, Company Name, key metrics)
   - Note any patterns or standout names worth reviewing

5. **Restore the country code** back to its original value after the run if you changed it.
