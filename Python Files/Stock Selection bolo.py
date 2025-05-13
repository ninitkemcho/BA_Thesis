import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# Step 1: Fetch and filter S&P 500 companies added before or on 2010-01-01
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
table = soup.find("table", {"id": "constituents"})
df = pd.read_html(str(table))[0]

df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce')
df = df[df['Date added'] <= '2010-01-01']
df = df[['Symbol', 'GICS Sector', 'Date added']]
df.rename(columns={'Symbol': 'ticker', 'GICS Sector': 'Sector'}, inplace=True)

stocks_data = df.copy()

# Step 2: Load daily data from CSV
data_path = r"C:/Users/nino.tkemaladze/Desktop/BA_Thesis(desktop)/Data/data_daily.csv"
daily_data = pd.read_csv(data_path, parse_dates=['date'])

GOOG = pd.read_csv(
    'C:/Users/nino.tkemaladze/Desktop/BA_Thesis(desktop)/Data/GOOG_unadjusted.csv')
GOOG['date'] = pd.to_datetime(GOOG['date'], format='%m/%d/%Y', errors='coerce')

LIN = pd.read_csv(
    'C:/Users/nino.tkemaladze/Desktop/BA_Thesis(desktop)/Data/LAN_unadjusted.csv')
LIN['date'] = pd.to_datetime(LIN['date'], format='%m/%d/%Y', errors='coerce')

VTRS = pd.read_csv(
    'C:/Users/nino.tkemaladze/Desktop/BA_Thesis(desktop)/Data/VTRS_unadjusted.csv')
VTRS['date'] = pd.to_datetime(VTRS['date'], format='%m/%d/%Y', errors='coerce')


daily_data = daily_data[~daily_data['ticker'].isin(['GOOG', 'LIN', 'VTRS'])]

daily_data = pd.concat([daily_data, GOOG], ignore_index=True)
daily_data['ticker'] = daily_data['ticker'].fillna('GOOG')

daily_data = pd.concat([daily_data, LIN], ignore_index=True)
daily_data['ticker'] = daily_data['ticker'].fillna('LIN')

daily_data = pd.concat([daily_data, VTRS], ignore_index=True)
daily_data['ticker'] = daily_data['ticker'].fillna('VTRS')


daily_data = daily_data.merge(
    df[['ticker', 'Sector']], how='left', on='ticker')

#daily_data.to_csv('daily_data.csv')

# Step 3: Filter historical data for tickers in our list
#daily_data = daily_data[daily_data['ticker'].isin(stocks_data['ticker'])]

# TRAIN-TEST SPLITTING

daily_data['date'] = pd.to_datetime(daily_data['date'], dayfirst=True, errors='coerce')

daily_data.drop('Unnamed: 0', axis = 1, inplace = True)

# Drop rows where conversion failed, if any
# Split the data

train = daily_data[daily_data['date'] <= '2021-03-31']
test = daily_data[daily_data['date'] > '2021-03-31']

# first_dates = daily_data.groupby('ticker')['date'].min().reset_index()
# last_dates = daily_data.groupby('ticker')['date'].max().reset_index()

train.to_csv('train.csv')
test.to_csv('test.csv')

# Step 4: Calculate average volume and volatility for each ticker
volume_threshold = 1_000_000
volatility_threshold = 0.015

filtered_tickers = []

for ticker in stocks_data['ticker']:
    stock_df = train[train['ticker'] == ticker]
    if stock_df.empty:
        continue
    avg_volume = stock_df['volume'].mean()
    returns = stock_df['close'].pct_change().dropna()
    volatility = returns.std()

    if not np.isnan(avg_volume) and not np.isnan(volatility):
        if avg_volume >= volume_threshold and volatility >= volatility_threshold:
            sector = stocks_data[stocks_data['ticker']
                                 == ticker]['Sector'].values[0]
            filtered_tickers.append({
                'ticker': ticker,
                'Sector': sector,
                'Average Volume': avg_volume,
                'Volatility': volatility
            })

filtered_stocks_df = pd.DataFrame(filtered_tickers)

# Step 5: Function to select 4 random stocks per sector


def select_random_stocks_by_sector(data, n=4, seed=42):
    random_stocks = []
    sectors = data['Sector'].unique()
    for sector in sectors:
        sector_stocks = data[data['Sector'] == sector]
        if len(sector_stocks) >= n:
            selected = sector_stocks.sample(n=n, random_state=seed)
            random_stocks.append(selected)
    return pd.concat(random_stocks)


# Selection A: With volume and volatility filter
selected_stocks_A = select_random_stocks_by_sector(filtered_stocks_df, seed=42)
tickers_A = selected_stocks_A['ticker'].tolist()

# Selection B: Without any filter
selected_stocks_B = select_random_stocks_by_sector(stocks_data, seed=42)
tickers_B = selected_stocks_B['ticker'].tolist()

# Step 6: Filter price data for both selections
price_data_A = train[train['ticker'].isin(tickers_A)].copy()
price_data_B = train[train['ticker'].isin(tickers_B)].copy()

price_data_A_test = test[test['ticker'].isin(tickers_A)].copy()
price_data_B_test = test[test['ticker'].isin(tickers_B)].copy()

# Step 7: Save results
#selected_stocks_A.to_csv("train_with_VV.csv", index=False)
price_data_A.to_csv("train_with_VV.csv", index=False)
price_data_A_test.to_csv("test_with_VV.csv", index=False)

#selected_stocks_B.to_csv("selected_stocks_without_VV.csv", index=False)
price_data_B.to_csv("train_without_VV.csv", index=False)
price_data_B_test.to_csv("test_without_VV.csv", index=False)

print("Selection A (with volume & volatility filter):")
print(selected_stocks_A[['ticker', 'Sector']])

print("\nSelection B (random only):")
print(selected_stocks_B[['ticker', 'Sector']])
