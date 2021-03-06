import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import talib
import cbpro
from itertools import product


class BBRSI:
    """
    """

    def __init__(self, sandbox):
        if sandbox:
            self.url = 'https://api-public.sandbox.pro.coinbase.com'
        else:
            self.url = 'https://api.pro.coinbase.com'

    def _get_historical_data(self, ticker, time_interval, start_dt, end_dt):
        # initialize coinbase pro
        c = cbpro.PublicClient(api_url=self.url)
        # get historical data
        historical = None
        while historical is None:
            try:
                historical = None
                historical = pd.DataFrame(c.get_product_historic_rates(product_id=ticker, granularity=time_interval,
                                                                       start=start_dt, end=end_dt))
            except Exception as e:
                print(f'Error obtaining historical: {e}, local time: {str(datetime.datetime.now())}')
        return historical

    def get_price(self, ticker, time_interval, start_dt, end_dt):
        # get historical data
        # start_dt and end_dt are None if we want real time data
        if (start_dt is None) & (end_dt is None):
            historical = self._get_historical_data(ticker=ticker, time_interval=time_interval, start_dt=start_dt,
                                                   end_dt=end_dt)
        # if start and end are defined
        else:
            # get expected number of rows (300 row limit per API call)
            n_rows = ((pd.to_datetime(end_dt, utc=True) - pd.to_datetime(start_dt,
                                                                         utc=True)).total_seconds() // time_interval) + 1
            # if expected number of rows is <= 300 then multiple calls in chunk are not necessary
            if n_rows <= 300:
                historical = self._get_historical_data(ticker=ticker, time_interval=time_interval,
                                                       start_dt=start_dt.isoformat(), end_dt=end_dt.isoformat())
            # if expected number of rows is > 300 then multiple calls in chunks are necessary
            else:
                # get date cutoffs for each 300 timestamp chunk
                chunk = 1
                dates = [start_dt]
                if time_interval == 3600:
                    while chunk < np.ceil(n_rows / 300):
                        dates.append(start_dt + datetime.timedelta(hours=chunk * 300))
                        chunk = chunk + 1
                elif time_interval == 86400:
                    while chunk < np.ceil(n_rows / 300):
                        dates.append(start_dt + datetime.timedelta(days=chunk * 300))
                        chunk = chunk + 1
                dates.append(end_dt)
                historical = pd.DataFrame()
                # call and concatenate data chunks
                for i in range(len(dates) - 1):
                    hist = self._get_historical_data(ticker=ticker, time_interval=time_interval,
                                                     start_dt=dates[i].isoformat(), end_dt=dates[i + 1].isoformat())
                    historical = pd.concat([historical, hist], axis=0)
        historical.columns = ["date", "open", "high", "low", "close", "volume"]
        historical['date'] = pd.to_datetime(historical['date'], unit='s')
        historical.sort_values(by='date', ascending=True, inplace=True)
        historical.reset_index(inplace=True, drop=True)
        return historical

    def get_bb(self, close, matype=0, timeperiods=20, bb_lwr_stdv=2, bb_upr_stdv=2):
        # get upper, middle, lower bollinger bands
        upper, middle, lower = talib.BBANDS(close, matype=matype, timeperiod=timeperiods, nbdevup=bb_upr_stdv,
                                            nbdevdn=bb_lwr_stdv)
        out_df = dict(bb_upper=upper, bb_middle=middle, bb_lower=lower)
        out_df = pd.DataFrame(out_df, columns=['bb_upper', 'bb_middle', 'bb_lower'])
        return out_df

    def get_rsi(self, close, timeperiods=14):
        # get rsi values
        rsi_vals = talib.RSI(close, timeperiod=timeperiods)
        out_df = dict(rsi=rsi_vals)
        out_df = pd.DataFrame(out_df, columns=['rsi'])
        return out_df

    def get_strategy_data(self, ticker, interval, start_dt, end_dt, bb_lwr_stdv, bb_upr_stdv):
        df_price = self.get_price(ticker=ticker, time_interval=interval, start_dt=start_dt, end_dt=end_dt)
        df_bb = self.get_bb(close=df_price['close'], matype=0, timeperiods=20, bb_lwr_stdv=bb_lwr_stdv,
                            bb_upr_stdv=bb_upr_stdv)
        df_rsi = self.get_rsi(close=df_price['close'], timeperiods=14)
        df = pd.concat([df_price, df_bb, df_rsi], axis=1)
        return df

    def _get_avg_num_trades(self, backtest_data_cln):
        month_ct = backtest_data_cln[['month', 'pos', 'neg']].groupby('month').sum().reset_index()
        year_ct = backtest_data_cln[['year', 'pos', 'neg']].groupby('year').sum().reset_index()
        avg_num_pos_m = np.mean(month_ct['pos'])
        avg_num_neg_m = np.mean(month_ct['neg'])
        avg_num_pos_y = np.mean(year_ct['pos'])
        avg_num_neg_y = np.mean(year_ct['neg'])
        tot_num_pos = np.sum(backtest_data_cln['pos'])
        tot_num_neg = np.sum(backtest_data_cln['neg'])
        return [tot_num_pos, tot_num_neg, avg_num_pos_m, avg_num_neg_m, avg_num_pos_y, avg_num_neg_y]

    def _get_avg_return_per_trade(self, backtest_data_cln):
        r_per_t = np.mean(backtest_data_cln['return'])
        r_per_pos_t = np.mean(backtest_data_cln[backtest_data_cln['pos'] == 1]['return'])
        r_per_neg_t = np.mean(backtest_data_cln[backtest_data_cln['neg'] == 1]['return'])
        return [r_per_t, r_per_pos_t, r_per_neg_t]

    def _get_avg_return_per_month(self, backtest_data_cln):
        performance = backtest_data_cln.groupby('month').agg(
            {'return': 'mean', 'profit': 'sum', 'balance': 'first', 'date': 'count'}).reset_index()
        performance.columns = ['month', 'average_return_per_trade', 'total_profit',
                               'total_account_balance_begin_of_month', 'number_of_completed_trades']
        performance['months_return'] = (
                performance['total_profit'] / performance['total_account_balance_begin_of_month'])
        r_per_m = np.mean(performance['months_return'])
        return r_per_m

    def _get_avg_return_per_year(self, backtest_data_cln, start_dt, end_dt):
        performance = backtest_data_cln.groupby('year').agg(
            {'return': 'mean', 'profit': 'sum', 'balance': 'first', 'date': 'count'}).reset_index()
        performance.columns = ['year', 'average_return_per_trade', 'total_profit',
                               'total_account_balance_begin_of_year', 'number_of_completed_trades']
        performance['years_return'] = (
                performance['total_profit'] / performance['total_account_balance_begin_of_year'])

        r_per_y = []
        num_trades_y = []
        years = []
        for i in range(start_dt.year, end_dt.year + 1):
            years.append(str(i))
            if pd.Period(str(i)) not in performance['year'].tolist():
                r_per_y.append(0)
                num_trades_y.append(0)
            else:
                r_per_y.append(float(performance[performance['year'] == str(i)]['years_return']))
                num_trades_y.append(float(performance[performance['year'] == str(i)]['number_of_completed_trades']))

        final_dict = {'r_per_y': r_per_y, 'num_trades_y': num_trades_y, 'years': years}
        return final_dict

    def _get_std_return_trade(self, backtest_data_cln):
        r_per_t_std = np.std(backtest_data_cln['return'])
        r_per_neg_t_std = np.std(backtest_data_cln[backtest_data_cln['neg'] == 1]['return'])
        return [r_per_t_std, r_per_neg_t_std]

    def _get_sharp_ratio(self, r_per_t, r_per_t_std):
        if r_per_t_std == 0:
            sharp_r = float('inf')
        else:
            sharp_r = r_per_t / r_per_t_std
        return sharp_r

    def _get_sortino_ratio(self, r_per_t, r_per_neg_t_std):
        if r_per_neg_t_std == 0:
            sort_r = float('inf')
        else:
            sort_r = r_per_t / r_per_neg_t_std
        return sort_r

    def _get_backtest_results(self, backtest_data, ticker, interval, bb_lwr_stdv, b_rsi, bb_upr_stdv, s_rsi, start_dt,
                              end_dt):
        # backtest_data might be empty if no trades were finished (buy and sell)
        if len(backtest_data) == 0:
            parameters = f"interval:{interval}, buy_bb_lwr:{bb_lwr_stdv}, buy_rsi:{b_rsi}, sell_bb_upr:{bb_upr_stdv}, sell_rsi:{s_rsi}"
            years = [str(i) for i in range(start_dt.year, end_dt.year + 1)]
            cols = ['ticker', 'parameters', 'tot_num_pos', 'tot_num_neg', 'avg_num_pos_m', 'avg_num_neg_m',
                    'avg_num_pos_y', 'avg_num_neg_y', 'r_per_t', 'r_per_t_std', 'sharp_r', 'r_per_pos_t',
                    'r_per_neg_t', 'r_per_neg_t_std', 'sort_r', 'r_per_m'] + ['return_' + x for x in years] +\
                   ['num_t_' + x for x in years]
            results = pd.DataFrame(
                [[ticker, parameters] + [0] * (len(cols) - 2)], columns=cols)
        else:
            # create columns
            backtest_data['month'] = backtest_data['date'].dt.to_period('M')
            backtest_data['year'] = backtest_data['date'].dt.to_period('Y')
            backtest_data['pos'] = np.where(backtest_data['return'] > 0, 1, 0)
            backtest_data['neg'] = np.where(backtest_data['return'] < 0, 1, 0)

            # avg number positive/negative trades per month/year
            avg_num_trades_list = self._get_avg_num_trades(backtest_data_cln=backtest_data)

            # avg return per trade
            r_per_t_list = self._get_avg_return_per_trade(backtest_data_cln=backtest_data)

            # standard deviation of return per trade
            return_std_list = self._get_std_return_trade(backtest_data_cln=backtest_data)

            # sharp ratio return per trade
            sharp_r = self._get_sharp_ratio(r_per_t=r_per_t_list[0], r_per_t_std=return_std_list[0])

            # sortino ratio return per trade
            sort_r = self._get_sortino_ratio(r_per_t=r_per_t_list[0], r_per_neg_t_std=return_std_list[1])

            # avg monthly return
            r_per_m = self._get_avg_return_per_month(backtest_data_cln=backtest_data)

            # avg return per year
            year_dict = self._get_avg_return_per_year(backtest_data_cln=backtest_data, start_dt=start_dt,
                                                      end_dt=end_dt)

            # final output
            parameters = f"interval:{interval}, buy_bb_lwr:{bb_lwr_stdv}, buy_rsi:{b_rsi}, sell_bb_upr:{bb_upr_stdv}, sell_rsi:{s_rsi}"
            results = pd.DataFrame(
                [[ticker, parameters, avg_num_trades_list[0], avg_num_trades_list[1], avg_num_trades_list[2],
                  avg_num_trades_list[3], avg_num_trades_list[4], avg_num_trades_list[5], r_per_t_list[0],
                  return_std_list[0], sharp_r, r_per_t_list[1], r_per_t_list[2], return_std_list[1], sort_r,
                  r_per_m] + year_dict['r_per_y'] + year_dict['num_trades_y']],
                columns=['ticker', 'parameters', 'tot_num_pos', 'tot_num_neg', 'avg_num_pos_m', 'avg_num_neg_m',
                         'avg_num_pos_y', 'avg_num_neg_y', 'r_per_t', 'r_per_t_std', 'sharp_r', 'r_per_pos_t',
                         'r_per_neg_t', 'r_per_neg_t_std', 'sort_r', 'r_per_m'] +
                        ['return_' + x for x in year_dict['years']] + ['num_t_' + x for x in year_dict['years']])
        return results

    def backtest(self, tickers, intervals, buy_signal, sell_signal, buy_size, buying_power, start_dt, end_dt, fees):

        # start_dt=datetime.datetime(2020, 1, 1)
        # end_dt=datetime.datetime(2022, 3, 30)
        # buy_size = 100 # amount in dollars to buy
        # buying_power = 1000 # total initial account balance
        # tickers = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD']
        # intervals = [3600, 86400]  # {60, 300, 900, 3600, 21600, 86400} - one minute, five minutes, fifteen minutes,
        # one hour, six hours, and one day, respectively
        # buy_signal = {'bb_lwr': [2, 3, 4], 'rsi': [10, 15, 20, 30, 0]}
        # sell_signal = {'bb_upr': [2, 3, 4, 0], 'rsi': [50, 60, 70, 75, 80, 0]}

        # get every combination of hyper-parameters
        tests = list(product(tickers, intervals, buy_signal['bb_lwr'], buy_signal['rsi'], sell_signal['bb_upr'],
                             sell_signal['rsi']))

        # initialize dataframe
        final_results = pd.DataFrame()

        # run through each test
        for test in tqdm(tests):
            # account balance at the end of the backtest - will be same as initial balance at the start
            balance = buying_power

            # initialize lists to keep record during backtest
            shares, price, timestamp, profits, acc_balance, trade_principle, returns = [], [], [], [], [], [], []

            # get parameters for test
            ticker = test[0]
            interval = test[1]
            bb_lwr_stdv = test[2]
            bb_upr_stdv = test[4]
            # bbrsi = BBRSI(tickers=ticker, interval=interval, start_dt=start_dt, end_dt=end_dt, sandbox=False)

            # get data based on parameters
            status = None
            while status is None:
                try:
                    df_final = self.get_strategy_data(ticker=ticker, interval=interval, start_dt=start_dt,
                                                      end_dt=end_dt, bb_lwr_stdv=bb_lwr_stdv, bb_upr_stdv=bb_upr_stdv)
                    status = df_final.copy()
                except Exception as e:
                    status = None
                    print(f'Error obtaining data: {e}')

            # drop rows with NA (this happens because bands are not computed until the 20th time step)
            df_final = df_final.dropna()
            df_final.reset_index(drop=True, inplace=True)

            # check for buy and sell signals
            for i in range(len(df_final)):
                # buy
                b_rsi = test[3]
                if ((df_final['low'][i] <= df_final['bb_lower'][i]) | (df_final['close'][i] <= df_final['bb_lower'][i]))\
                        & (df_final['rsi'][i] <= b_rsi) & (buying_power >= buy_size):
                    # fee (0.6% to buy)
                    fee = buy_size * fees
                    # number of shares bought
                    num_s_b = (buy_size - fee) / df_final['bb_lower'][i]
                    # buying price
                    price_b = df_final['bb_lower'][i]
                    # reduce buying power
                    buying_power = buying_power - buy_size
                    # keep record
                    shares.append(num_s_b)
                    price.append(price_b)
                    trade_principle.append(buy_size)
                    avg_price = np.sum(np.array(shares) * np.array(price)) / np.sum(shares)
                    # print(f"B: {num_s_b} for {price_b} @ {df_final['date'][i]}")

                # sell
                s_rsi = test[5]
                if ((df_final['high'][i] >= df_final['bb_upper'][i]) | (df_final['close'][i] >= df_final['bb_upper'][i]))\
                        & (df_final['rsi'][i] >= s_rsi) & (len(shares) > 0):
                    # number of shares sold
                    num_s_s = np.sum(shares)
                    # selling price
                    price_s = df_final['bb_upper'][i]
                    # timestamp of sale
                    ts_s = df_final['date'][i]
                    # fee (0.6% to sell)
                    fee = num_s_s * price_s * fees
                    # keep record
                    profit = ((price_s - avg_price) * np.sum(shares)) - fee
                    balance = balance + profit
                    perc_return = (profit / np.sum(trade_principle))
                    # print(f"S: {num_s_s} for {price_s} @ {ts_s}, profit: {profit}, return: {perc_return}, balance: {balance}")
                    profits.append(profit)
                    returns.append(perc_return)
                    timestamp.append(ts_s)
                    acc_balance.append(balance)
                    # reset
                    buying_power = buying_power + (np.sum(trade_principle))
                    shares = []
                    price = []
                    trade_principle = []

            # get results
            backtest_data = pd.DataFrame(
                {'date': timestamp, 'return': returns, 'profit': profits, 'balance': acc_balance})
            #return df_final
            print(backtest_data)
            results = self._get_backtest_results(backtest_data=backtest_data, ticker=ticker, interval=interval,
                                                 bb_lwr_stdv=bb_lwr_stdv, b_rsi=b_rsi, bb_upr_stdv=bb_upr_stdv,
                                                 s_rsi=s_rsi, start_dt=start_dt, end_dt=end_dt)
            final_results = pd.concat([final_results, results], axis=0)

        # compute final results of backtest
        final_results['tot_num_trades'] = final_results['tot_num_pos'] + final_results['tot_num_neg']
        final_results['prop_pos_trades'] = final_results['tot_num_pos'] / final_results['tot_num_trades']
        final_results['prop_neg_trades'] = final_results['tot_num_neg'] / final_results['tot_num_trades']
        final_results['diff_bw_pos_neg_returns'] = final_results['r_per_pos_t'] - abs(final_results['r_per_neg_t'])

        return final_results
