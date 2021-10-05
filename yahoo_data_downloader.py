#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""

import numpy as np
import datetime
import pandas as pd
from pandas_datareader import data as dataread
import pickle
import yfinance as yf
import glob
import yahoo_data_tools
import os


class YahooDataDownloader:

    def __init__(self, dataset):
        self.dataset = dataset
        self.data_path = './' + dataset + '/'
        self.ymd_format_str = '%Y-%m-%d'
        self.start_date = None
        self.end_date = None
        self.new_data = False  # Boolean flag indicating that new data have been downloaded

    # %% DOWNLOADING AND STORING DATA SECTION
    def merge_stock_data(self):
        close_adj_filename = self.data_path + 'quotes_intraday_adj.pkl'
        if os.path.isfile(close_adj_filename):
            close_ds = pd.read_pickle(close_adj_filename)
        else:
            close_ds = {}

        data_file_collection = glob.glob(self.data_path + 'data*.pkl')
        data_file_collection.sort(reverse=True)

        for data_file in data_file_collection:
            close_df = pd.read_pickle(data_file).Close

            # Replace inf with nan
            close_df = close_df.replace([np.inf, -np.inf], np.nan)

            date_str_list = yahoo_data_tools.get_date_list(close_df)

            ts_str_list = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in close_df.index]

            for d in date_str_list:
                matching = [d in s for s in ts_str_list]
                close_df_curr = close_df[matching]

                if d in close_ds:  # data frame for day d is already in dataset
                    # update current data frame only with a new one carrying more data
                    if np.count_nonzero(~np.isnan(close_df_curr)) > np.count_nonzero(~np.isnan(close_ds[d])):
                        close_ds[d] = close_df_curr
                else:
                    close_ds[d] = close_df_curr

        return close_ds

    def merge_adjust_stk_data(self):
        # Merge stock data into a single dataset
        print(' --- Building dataset... ---')
        close_ds = self.merge_stock_data()
        print(' --- done. --- ')

        # Download updated daily closing prices
        date_str_list = list(close_ds.keys())
        date_str_list.sort()
        self.start_date = datetime.datetime.strptime(date_str_list[0], self.ymd_format_str)
        self.end_date = datetime.datetime.strptime(date_str_list[-1], self.ymd_format_str)
        self.end_date = self.end_date + datetime.timedelta(days=1)

        print(' --- Downloading daily closing prices for back-adjusting... ---')
        data_1d = yahoo_data_tools.download_stock_data(self.dataset, self.start_date, self.end_date, '1d')
        print(' --- done. --- ')
        close_1d_df = data_1d.Close

        # Back-adjust prices to account for dividends and splits
        print(' --- Back-adjusting stock prices... ---')
        close_adj_ds = yahoo_data_tools.adjust_price(close_ds, close_1d_df)
        print(' --- done. --- ')

        with open(self.data_path + 'quotes_intraday_adj.pkl', 'wb') as handle:
            pickle.dump(close_adj_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def merge_mktcap_data(self):
        data_file_collection = glob.glob(self.data_path + 'mktcap_*.pkl')
        data_file_collection.sort()

        mktcap_df = pd.DataFrame()

        for data_file in data_file_collection:
            df = pd.read_pickle(data_file)
            date_str = data_file[-14:-10] + '-' + data_file[-9:-7] + '-' + data_file[-6:-4]
            df.name = date_str
            mktcap_df = mktcap_df.append(df)

        df_datetime_idx = pd.to_datetime(mktcap_df.index, format=self.ymd_format_str)
        mktcap_df.set_index(pd.DatetimeIndex(df_datetime_idx), inplace=True)

        return mktcap_df

    def download_latest_data(self):
        today = yahoo_data_tools.ts_time_rounder(datetime.datetime.today())
        date_1m_1 = today - datetime.timedelta(days=7)

        timestamp_string = datetime.date.today().strftime(self.ymd_format_str)

        print(' --- Downloading latest intraday prices... --- ')
        data_1m = yahoo_data_tools.download_stock_data(self.dataset, date_1m_1, today, '1m')

        data = data_1m

        # Downcast dataset to float32 datatype to reduce memory usage on disk
        # data = data.astype('float32')

        with open(self.data_path + f'data_{timestamp_string}.pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Download market capitalization data
        try:
            print(' --- Downloading market capitalization data...')
            close_df = data.Close
            close_df_clean = close_df.dropna(axis=1, how='all')
            mkt_cap = dataread.get_quote_yahoo(close_df_clean.columns)['marketCap']
            print(' --- done. --- ')

            with open(self.data_path + f'mktcap_{timestamp_string}.pkl', 'wb') as handle:
                pickle.dump(mkt_cap, handle, protocol=pickle.HIGHEST_PROTOCOL)

            mktcap_df = self.merge_mktcap_data()
            with open(self.data_path + 'mktcap.pkl', 'wb') as handle:
                pickle.dump(mktcap_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            print('\n --- Warning! Yahoo Finance API not working --- \n')

        # Merge and adjust stock prices
        self.merge_adjust_stk_data()

        # Set indicator of new data to True
        self.new_data = True

    # %% LOADING DATA SECTION
    @staticmethod
    def get_returns_volat(close_ds, verbose=False):
        date_str_list = list(close_ds.keys())
        date_str_list.sort()
        last_price_df = pd.DataFrame()
        real_vol_df = pd.DataFrame()

        for d in date_str_list:
            close_df_curr = close_ds[d]

            # Drop spurious rows filled with (almost all) nans
            close_df_curr.dropna(axis=0, thresh=int(.05 * len(close_df_curr.columns)), inplace=True)

            # Ensure that the index is a datetime index (for time interp. of volatility)
            df_datetime_idx = pd.to_datetime(close_df_curr.index, utc=True)
            close_df_curr.set_index(pd.DatetimeIndex(df_datetime_idx), inplace=True)

            # ----- LAST PRICE -----
            # Last price is the last available sample of the day
            last_pr_curr = close_df_curr.iloc[[-1]].copy()

            # but we allow, for some stocks, to pick an earlier price, if needed
            timestamp = last_pr_curr.index
            for stk in last_pr_curr.columns[last_pr_curr.isna().any()].tolist():  # stk_list:
                # Get last valid data sample
                last_idx = close_df_curr[stk].last_valid_index()
                if last_idx:
                    last_pr_curr.loc[timestamp, stk] = close_df_curr.loc[last_idx, stk]

            last_price_df = last_price_df.append(last_pr_curr)

            # ----- REALIZED VOLATILITY -----
            # If data are available at a fixed time step (i.e., with no missing sampples),
            # use integrated volatility, otherwise use sample standard deviation
            use_high_freq_vol = True
            if use_high_freq_vol:
                real_vol_df_curr = np.log(close_df_curr)
                real_vol_df_curr = real_vol_df_curr.diff()
                real_vol_df_curr = real_vol_df_curr.dropna(how='all')
                real_vol_df_curr = real_vol_df_curr.pow(2)
                n_nans = real_vol_df_curr.isna().sum()
                penalty = 1. / pow(1. - n_nans / real_vol_df_curr.shape[0], 2)
                real_vol_df_curr = real_vol_df_curr.interpolate(method='time')
                real_vol_df_curr = real_vol_df_curr.sum()
                real_vol_df_curr = np.sqrt(real_vol_df_curr) * penalty
            else:
                real_vol_df_curr = close_df_curr.pct_change(fill_method=None)
                n_nans = real_vol_df_curr.isna().sum()
                penalty = 1. / pow(1. - n_nans / real_vol_df_curr.shape[0], 2)
                real_vol_df_curr = real_vol_df_curr.std() * penalty

            real_vol_df_curr = real_vol_df_curr.replace(0, np.nan)

            # Convert series to dataframe row
            real_vol_df_curr = real_vol_df_curr.to_frame().transpose()

            # Change datetime index to include only the day
            real_vol_df_curr.index = last_pr_curr.index

            if real_vol_df_curr.shape[1] > 0:  # ensure that row is not empty
                real_vol_df = real_vol_df.append(real_vol_df_curr)

            if verbose:
                print(d)

        last_price_df = last_price_df.dropna(axis=1, how='all')
        real_vol_df = real_vol_df[last_price_df.columns]

        # Check for errors in the position of decimal point of last prices
        last_price_df = yahoo_data_tools.correct_data_anomalies(last_price_df)

        # ----- RETURNS -----
        ret_df = last_price_df.pct_change(fill_method=None)
        ret_df = ret_df.dropna(axis=0, how='all')

        last_price_df = last_price_df.iloc[1:]  # Delete first row of last price
        real_vol_df = real_vol_df.iloc[1:]  # and realized volatility

        return last_price_df, ret_df, real_vol_df

    def get_stock_data(self):
        # Load price dataset and compute daily returns and volatility
        close_adj_filename = self.data_path + 'quotes_intraday_adj.pkl'
        close_adj_ds = pd.read_pickle(close_adj_filename)

        timestamp_string = datetime.date.today().strftime(self.ymd_format_str)
        filename = self.data_path + 'stk_data_' + timestamp_string + '.pkl'

        if os.path.isfile(filename) and not self.new_data:
            # Just load data from pickle file
            [last_price_df, ret_df, real_vol_df, mktcap_df] = pd.read_pickle(filename)
        else:
            # Discard days with little or no valid data
            for d in list(close_adj_ds):
                n_valid = np.count_nonzero(~np.isnan(close_adj_ds[d]))
                n_total = np.prod(close_adj_ds[d].shape)
                if n_valid / n_total < 0.4:
                    del close_adj_ds[d]

            last_price_df, ret_df, real_vol_df = self.get_returns_volat(close_adj_ds)

            # Get market capitalization data
            mktcap_filename = self.data_path + 'mktcap.pkl'
            mktcap_df_pkl = pd.read_pickle(mktcap_filename)

            # Reindex and interpolate data frame of mkt. cap. data over entire price time range
            mktcap_df = yahoo_data_tools.reindex_by_date(mktcap_df_pkl, ret_df.index)

            with open(filename, 'wb') as handle:
                pickle.dump([last_price_df, ret_df, real_vol_df, mktcap_df],
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Delete old pickle files (if present)
        old_files = glob.glob(self.data_path + 'stk_data_*.pkl')
        for f in old_files:
            if f != filename:
                os.remove(f)

        self.new_data = False

        if self.start_date is None:
            self.start_date = last_price_df.index[0]
        if self.end_date is None:
            self.end_date = last_price_df.index[-1] + datetime.timedelta(days=1)

        return last_price_df, ret_df, real_vol_df, mktcap_df, close_adj_ds

    def get_mkt_data(self, datetimeindex):

        timestamp_string = datetime.date.today().strftime(self.ymd_format_str)
        filename = self.data_path + 'mkt_data_' + timestamp_string + '.pkl'

        if os.path.isfile(filename):  # Just load data from pickle file
            [mkt_ret_df, mkt_idx_df] = pd.read_pickle(filename)
        else:
            # Set start and end date for download
            if datetimeindex is None:
                start_date = self.start_date
                end_date = self.end_date
            else:
                start_date = datetimeindex[0] - datetime.timedelta(days=3)
                end_date = datetimeindex[-1] + datetime.timedelta(days=1)

            # Set asset symbol for download
            if self.dataset == 'STOXXE600':
                symbol = '^STOXX'
            elif self.dataset == 'SP500':
                symbol = '^SP500TR'
            else:
                return None

            # Download index data
            data = yf.download(symbol, start=start_date, end=end_date, interval='1d',
                               auto_adjust=False, actions=False, threads=False)

            # Index value and returns
            mkt_idx_df_tmp = data['Close']
            mkt_ret_df_tmp = mkt_idx_df_tmp.pct_change(fill_method=None)

            # Reindex the two series
            mkt_idx_df = yahoo_data_tools.reindex_by_date(mkt_idx_df_tmp, datetimeindex)
            mkt_ret_df = yahoo_data_tools.reindex_by_date(mkt_ret_df_tmp, datetimeindex)

            # Replace missing values with zeros for returns and last values for index
            mkt_ret_df = mkt_ret_df.fillna(value=0)
            mkt_idx_df = mkt_idx_df.fillna(method='ffill')

            with open(filename, 'wb') as handle:
                pickle.dump([mkt_ret_df, mkt_idx_df], handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Delete old pickle files
        old_files = glob.glob(self.data_path + 'mkt_data_*.pkl')
        for f in old_files:
            if f != filename:
                os.remove(f)

        return mkt_ret_df, mkt_idx_df
