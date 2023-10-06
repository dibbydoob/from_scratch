import lzma
import dill as pickle
import pytz
import numpy as np
import pandas as pd

def load_pickle(path):
    with lzma.open(path,"rb") as fp:
        file = pickle.load(fp)
    return file

def save_pickle(path,obj):
    with lzma.open(path,"wb") as fp:
        pickle.dump(obj,fp)

def get_pnl_stats(date, prev, portfolio_df, insts, idx, dfs):
    day_pnl = 0
    nominal_ret = 0
    for inst in insts:
        units = portfolio_df.loc[idx - 1, "{} units".format(inst)]
        if units != 0:
            delta = dfs[inst].loc[date,"close"] - dfs[inst].loc[prev,"close"]
            inst_pnl = delta * units
            day_pnl += inst_pnl
            nominal_ret += portfolio_df.loc[idx - 1, "{} w".format(inst)] * dfs[inst].loc[date, "ret"]
    capital_ret = nominal_ret * portfolio_df.loc[idx - 1, "leverage"]
    portfolio_df.loc[idx,"capital"] = portfolio_df.loc[idx - 1,"capital"] + day_pnl
    portfolio_df.loc[idx,"day_pnl"] = day_pnl
    portfolio_df.loc[idx,"nominal_ret"] = nominal_ret
    portfolio_df.loc[idx,"capital_ret"] = capital_ret
    return day_pnl, capital_ret


class Momentum():
    
    def __init__(self, insts, dfs, start, end, portfolio_vol=0.20, pairs=[],smooth=True):
        self.insts = insts
        self.dfs = dfs
        self.start = start 
        self.end = end
        self.portfolio_vol=portfolio_vol
        self.pairs = pairs
        self.smooth=smooth

    def init_portfolio_settings(self, trade_range):
        portfolio_df = pd.DataFrame(index=trade_range)\
            .reset_index()\
            .rename(columns={"index":"datetime"})
        portfolio_df.loc[0,"capital"] = 10000
        portfolio_df.loc[0,"capital_ret"] = 0.0
        portfolio_df.loc[0,"nominal_ret"] = 0.0
        return portfolio_df

    def compute_meta_info(self,trade_range):
        for inst in self.insts:
            df=pd.DataFrame(index=trade_range)
            vol = (-1 + self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)).rolling(30).std()
            
            crossover = np.zeros(len(self.dfs[inst]))
            for pair in self.pairs:
                crossover += np.where(self.dfs[inst].close.rolling(pair[0]).mean() > self.dfs[inst].close.rolling(pair[1]).mean(), 1, 0)
            self.dfs[inst]["alpha"] = crossover
            
            if self.smooth:
                global exp_mean 
                exp_mean = 0.0
                def smooth(val):
                    global exp_mean 
                    exp_mean = exp_mean * 0.94 + 0.06 * val
                    return exp_mean
                self.dfs[inst]["alpha"] = self.dfs[inst]["alpha"].apply(smooth)
            
            self.dfs[inst]["vol"] = vol

            self.dfs[inst] = df.join(self.dfs[inst]).fillna(method="ffill").fillna(method="bfill")
            self.dfs[inst]["vol"] = np.where(self.dfs[inst]["vol"] < 0.005, 0.005, self.dfs[inst]["vol"])
            
            self.dfs[inst]["ret"] = -1 + self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).fillna(method="bfill")
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)
            self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)
        return 

    def get_strat_scaler(self, target_vol, ewmas, ewstrats):
        ann_realized_vol = np.sqrt(ewmas[-1] * 253)
        return target_vol / ann_realized_vol * ewstrats[-1]
    
    def run_simulation(self):
        print("running backtest")
        date_range = pd.date_range(start=self.start,end=self.end, freq="D", tz=pytz.utc)
        self.compute_meta_info(trade_range=date_range)
        portfolio_df = self.init_portfolio_settings(trade_range=date_range)
        ewmas, ewstrats = [0.01], [1]
        for i in portfolio_df.index:
            date = portfolio_df.loc[i,"datetime"]

            eligibles = [inst for inst in self.insts if self.dfs[inst].loc[date,"eligible"]]
            non_eligibles = [inst for inst in self.insts if inst not in eligibles]
            strat_scalar = 2
            
            if i != 0:
                date_prev = portfolio_df.loc[i-1, "datetime"]
                strat_scalar = self.get_strat_scaler(
                    target_vol=self.portfolio_vol,
                    ewmas=ewmas,
                    ewstrats=ewstrats
                )
                day_pnl, capital_ret = get_pnl_stats(
                    date=date,
                    prev=date_prev,
                    portfolio_df=portfolio_df,
                    insts=self.insts,
                    idx=i,
                    dfs=self.dfs
                )

                ewmas.append(0.06 * (capital_ret**2) + 0.94 * ewmas[-1] if capital_ret != 0 else ewmas[-1])
                ewstrats.append(0.06 * strat_scalar + 0.94 * ewstrats[-1] if capital_ret != 0 else ewstrats[-1])
            
            alpha_scores = {}
            for inst in eligibles:
                alpha_scores[inst] = self.dfs[inst].loc[date, "alpha"]
            
            for inst in non_eligibles:
                portfolio_df.loc[i, "{} w".format(inst)] = 0
                portfolio_df.loc[i, "{} units".format(inst)] = 0
            
            forecast_chips = np.linalg.norm(list(alpha_scores.values()), ord=1)
            vol_target = strat_scalar * (self.portfolio_vol / np.sqrt(253)) * portfolio_df.at[i,"capital"]
            nominal_tot = 0
            for inst in eligibles:
                forecast = alpha_scores[inst]
                position = (forecast / forecast_chips) \
                    * vol_target \
                    / (self.dfs[inst].loc[date,"vol"] * self.dfs[inst].loc[date,"close"]) \
                    if forecast_chips != 0 else 0
                portfolio_df.loc[i, inst + " units"] = position 
                nominal_tot += abs(position * self.dfs[inst].loc[date,"close"])

            for inst in eligibles:
                units = portfolio_df.loc[i, inst + " units"]
                nominal_inst = units * self.dfs[inst].loc[date,"close"]
                inst_w = nominal_inst / nominal_tot if nominal_tot != 0 else 0
                portfolio_df.loc[i, inst + " w"] = inst_w
            
            portfolio_df.loc[i, "nominal"] = nominal_tot
            portfolio_df.loc[i, "leverage"] = nominal_tot / portfolio_df.loc[i, "capital"]
            if i%100 == 0: print(portfolio_df.loc[i])

        return portfolio_df.set_index('datetime', drop=True)