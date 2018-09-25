# Fundamental Analysis Stocks

This script uses the [Good Morning](https://github.com/petercerno/good-morning) repo by Peter Cerno. This code parses the Morning Star website for a given ticker symbol using the `gm.KeyRatiosDownloader()` function and returns the specified fundamental indicators and ratios for that company. My code runs this in a loop parsing through all the companies in a country specific CSV file with the ticker symbols of all the companies for that country. It then outputs all that into a dataframe.

## Requirements
Python 3, Good Morning, Pandas

Here is the raw output:
![capture](https://user-images.githubusercontent.com/30912225/46042636-ccd44880-c10d-11e8-8582-066328018cf7.PNG)

 Some boolean logic on the dataframes, or creating your own columns through calculations of other ratios can then be undertaken as a filter to narrow down the list of companies. This is then output to anoth 'condition' dataframe. The code outputs the raw and condition dataframes to a csv file. 
 
 Condition output:
 ![capture2](https://user-images.githubusercontent.com/30912225/46042152-9a761b80-c10c-11e8-92e7-e4e30f683672.PNG)
 
 Currently there is a limited number of sequential requests to the Morning Star website using Good Morning, so the code was modified to ignore Value Error and retry with the same ticker symbol.
 
 I welcome others to fork and continue this in the future.
 
 ## Usage
 Modify `country_code = 'UK'` at the beginning of the script before you run it, picking one of the countries in the dictionary. Then run in IDE. Takes about 1-2 mins depending on internet connection and number of tickers in input file.
 
 ## Future tasks
- [x] Initial concept script
- [ ] Make script object oriented function
- [ ] Enable automatic creation, up to date list of ticker symbols, perhaps parsing some online source
- [ ] Expand to other country ticker symbols
