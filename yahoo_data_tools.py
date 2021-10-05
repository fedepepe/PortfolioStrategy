#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:51:00 2021

@author: federico
"""

import numpy as np
import pandas as pd
import symbols_string
import yfinance as yf


def get_date_list(close_df):
    date_str_list = close_df.index
    date_str_list = [d.strftime('%Y-%m-%d') for d in date_str_list]
    date_str_list = list(set(date_str_list))
    date_str_list.sort()
    return date_str_list

def ts_time_rounder(ts):
    # Rounds time of datetime to midnight
    return ts.replace(second=0, microsecond=0, minute=0, hour=0)

def download_stock_data(dataset, date_start, date_end, intrvl_str):
    if dataset == 'STOXXE600':
        # Get string with tickers stored in a file
        symbols = symbols_string.get_symbols_string_STOXXE600()
        # import eikon as tr
        # tr.set_app_id('blah')
        # df, e = tr.get_data(['.STOXX'], 
        #                     ['TR.IndexConstituentRIC','TR.IndexConstituentName','TR.IndexConstituentWeightPercent'], 
        #                     {'SDate':'20160601'})
    elif dataset == 'SP500':
        # Get string with tickers stored in a file
        # symbols = symbols_string.get_symbols_string_SP500()

        # or, altenatively, get updated tickers from Wikipedia
        sp500_wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp500_constituents = pd.read_html(sp500_wiki_url, header=0)[0]
        symbols = sp500_constituents.Symbol.tolist()
    else:
        return None

    data = yf.download(symbols, start=date_start, end=date_end, interval=intrvl_str,
                       threads=True, auto_adjust=True, actions=True)
    data = data.dropna(how='all')
    return data
    
def reindex_by_date(df_old, idx_new):
    if isinstance(df_old, pd.Series):
        df_new = pd.Series(index=idx_new)
        for ts in idx_new:
            matching = [y and m and d for y, m, d in zip(df_old.index.year == ts.year,
                                                         df_old.index.month == ts.month,
                                                         df_old.index.day == ts.day)]
            if any(matching):
                df_new[ts] = df_old[matching].values
        
    elif isinstance(df_old, pd.DataFrame):
        df_new = pd.DataFrame(index=idx_new, columns=df_old.columns)
        for ts in idx_new:
            matching = [y and m and d for y, m, d in zip(df_old.index.year == ts.year,
                                                         df_old.index.month == ts.month,
                                                         df_old.index.day == ts.day)]
            if any(matching):
                df_new.loc[ts, :] = df_old.loc[matching, :].values
        
        df_new[df_new.columns] = df_new[df_new.columns].apply(pd.to_numeric)
        df_new = df_new.interpolate(method='time')
        df_new = df_new.fillna(method='bfill')
    
    return df_new

def adjust_price(close_ds, close_1d_df):
    date_str_list = list(close_ds.keys())
    date_str_list.sort()

    for d in date_str_list:
        df_curr = close_ds[d]
        tckr_list = df_curr.columns
        for tckr in tckr_list:
            try:
                close_1d_curr = close_1d_df.loc[d, tckr]
            except:
                continue

            # Check if we have an adjusted price
            check_if_isnan = np.isnan(close_1d_curr)

            if not check_if_isnan:
                # If yes, compute the scale factor and scale
                price_ser_curr = df_curr[tckr]

                last_idx = price_ser_curr.last_valid_index()
                if last_idx is None:
                    continue
                last_valid_price = price_ser_curr.loc[last_idx]

                if last_valid_price.size > 1:  # Bad data
                    continue

                adj_factor = close_1d_curr / last_valid_price
                price_ser_adj = adj_factor * price_ser_curr

                df_curr[tckr] = price_ser_adj

        close_ds[d] = df_curr
        
    return close_ds

    
def correct_data_anomalies(df):
    # Check for errors in position of decimal point of last prices
    col_list = df.columns
    for col in col_list:
        change_df = df[col].pct_change(fill_method=None)
        pos_jumps = (change_df > 8)
        neg_jumps = (change_df < -0.8)
        large_changes_idx = change_df[pos_jumps | neg_jumps].index
                    
        if len(large_changes_idx) > 0:
            for _ in np.arange(2):            
                # Get price level as rolling median over last 3 months
                data_level = df[col].rolling(60, min_periods=1).median()
                for idx in large_changes_idx:
                    # Detect abnormally high or low prices
                    enorm_hi_price = (df.loc[idx, col] > 8 * data_level[idx])
                    enorm_lo_price = (df.loc[idx, col] < 1/8 * data_level[idx])
                    if (enorm_hi_price or enorm_lo_price):
                        price_ratio = df.loc[idx, col]/data_level[idx]
                        try:
                            corr_factor = pow(10, -round(np.log10(abs(price_ratio))))
                        except:
                            breakpoint()
                        df.loc[idx, col] = corr_factor * df.loc[idx, col]
                
                change_df = df[col].pct_change(fill_method=None)
                pos_jumps = (change_df > 8)
                neg_jumps = (change_df < -0.8)
                large_changes_idx = change_df[pos_jumps | neg_jumps].index
                
    return df