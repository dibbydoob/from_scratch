import pytz
import yfinance
import requests
import threading
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import random
from utils import Alpha
from trend import Momentum
import numpy as np
import matplotlib.pyplot as plt

from utils import save_pickle, load_pickle
import numpy as np
import matplotlib.pyplot as plt



def get_sp500_tickers():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'html')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))
    tickers = list(df[0].Symbol)
    return tickers

def get_history(ticker, period_start, period_end, granularity="1d", tries=0):
    try:
        df = yfinance.Ticker(ticker).history(
            start=period_start,
            end=period_end,
            interval=granularity,
            auto_adjust=True
        ).reset_index()
    except Exception as err:
        if tries < 5:
            return get_history(ticker, period_start, period_end, granularity, tries+1)
        return pd.DataFrame()
    
    df = df.rename(columns={
        "Date":"datetime",
        "Open":"open",
        "High":"high",
        "Low":"low",
        "Close":"close",
        "Volume":"volume"
    })
    if df.empty:
        return pd.DataFrame()
    
    df["datetime"] = df["datetime"].dt.tz_localize(pytz.utc)
    df = df.drop(columns=["Dividends", "Stock Splits"])
    df = df.set_index("datetime",drop=True)
    return df

def get_histories(tickers, period_starts,period_ends, granularity="1d"):
    dfs = [None]*len(tickers)
    def _helper(i):
        print(tickers[i])
        df = get_history(
            tickers[i],
            period_starts[i], 
            period_ends[i], 
            granularity=granularity
        )
        dfs[i] = df
    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    #for i in range(len(tickers)): _helper(i) #can replace the 3 preceding lines for sequential polling 
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    dfs = [df for df in dfs if not df.empty]
    return tickers, dfs

def get_ticker_dfs(start,end):

    try:
        tickers, ticker_dfs = load_pickle("dataset.obj")
    except Exception as err:
        tickers = get_sp500_tickers()
        starts=[start]*len(tickers)
        ends=[end]*len(tickers)
        tickers,dfs = get_histories(tickers,starts,ends,granularity="1d")
        ticker_dfs = {ticker:df for ticker,df in zip(tickers,dfs)}
        save_pickle("dataset.obj", (tickers,ticker_dfs))
    return tickers, ticker_dfs 


period_start = datetime(2010,1,1, tzinfo=pytz.utc)
period_end = datetime.now(pytz.utc)
tickers, ticker_dfs = get_ticker_dfs(start=period_start,end=period_end)
tickers = tickers[:20]


random.seed(0)
ma_pairs=[]
for _ in range(10):
    pair1=random.randrange(5,200)
    pair2=random.randrange(5,200)
    if pair1 == pair2: continue
    ma_pairs.append((min(pair1,pair2),max(pair1,pair2)))
sims={}
# for pair in ma_pairs:
#     alpha = Momentum(insts=tickers,dfs=ticker_dfs,start=period_start,end=period_end,pairs=[pair])
#     df = alpha.run_simulation()
#     sims[pair] = df
alpha = Momentum(insts=tickers,dfs=ticker_dfs,start=period_start,end=period_end,pairs=ma_pairs,smooth=True)
df1 = alpha.run_simulation()
alpha = Momentum(insts=tickers,dfs=ticker_dfs,start=period_start,end=period_end,pairs=ma_pairs,smooth=False)
df2 = alpha.run_simulation()


plt.plot(np.log((1 + df1.capital_ret).cumprod()),label="smooth",linewidth=0.6)
plt.plot(np.log((1 + df2.capital_ret).cumprod()),label="rough",linewidth=0.6)
plt.legend()
plt.show()
exit()

sims = load_pickle("df.obj")
for k,v in sims.items():
    if k == "combi": continue
    plt.plot(np.log((1 + v.capital_ret).cumprod()),label=str(k),linewidth=0.6)
plt.plot(np.log((1 + sims["combi"].capital_ret).cumprod()),linewidth=2.5)
plt.legend()
plt.show()