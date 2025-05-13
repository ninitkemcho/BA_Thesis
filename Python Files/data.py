import pandas as pd
import time
from tiingo import TiingoClient
import warnings
warnings.filterwarnings("ignore")

# Load S&P 500 tickers and industries
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500 = pd.read_html(sp500_url)[0]
sp500 = sp500[sp500['Date added'] <= '2010-01-01']
sp500 = sp500.rename(columns={"Symbol": "Ticker", "GICS Sector": "Industry"})
sp500 = sp500.reset_index()

# sp500.to_excel('sp500.xlsx')


tickers  = list(sp500['Ticker'])

config = {
    'session': True,
    'api_key': ''
}
client = TiingoClient(config)

daily_data = []
failed_tickers = []

for ticker in tickers:
    try:
        df = client.get_dataframe(
            ticker,
            startDate='2010-01-01',
            endDate='2025-04-30',
            frequency='daily'
        )
        df["ticker"] = ticker
        daily_data.append(df)
        time.sleep(1.5)  # be polite to the API
    except Exception as e:
        failed_tickers.append(ticker)
        print(f"Failed for {ticker}: {e}")

all_data = pd.concat(daily_data).reset_index()
all_data['date'] = all_data['date'].dt.tz_localize(None)
all_data.to_excel('data_daily.xlsx')
