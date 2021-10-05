#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""


def main():
    import numpy as np
    from yahoo_data_downloader import YahooDataDownloader
    from portfolio_sim import PortfolioSimulator
    import portfolio_plot as pf_plot
    import time

    # %% Load data

    # Load dataset of closing prices
    dataset = 'SP500'  # SP500 or STOXXE600

    yahoo_data_toolbox = YahooDataDownloader(dataset)

    # Download latest stock prices
    yahoo_data_toolbox.download_latest_data()

    last_price_df, return_df, real_vol_df, mktcap_df, close_adj_ds = yahoo_data_toolbox.get_stock_data()

    mkt_ret_df, mkt_idx_df = yahoo_data_toolbox.get_mkt_data(return_df.index)

    # %% Set parameters and initialize
    endow = 1e6  # Initial amount of money to be invested
    n_stk = 8  # Number of stocks to hold in the portfolio
    n_obs_ar = np.arange(5, 16, 1)  # Number of past observations to use as training data
    n_reb_ar = np.arange(5, 16, 1)  # Rate of portfolio rebalancing (in trading days)

    algo = 'sev+'  # Algorithm to use for stock selection

    wght_mtd = ['equal', 'metric', 'mktcap', 'riskpar', 'lotp', 'tp']  # Weighting method for stock alloc.
    # wght_mtd = ['mktcap']
    trsctn_fee_fix = 2  # Fixed transaction fees
    trsctn_fee_prop = 3e-4  # Proportional transaction fees

    results_dir, results_tag = './' + dataset + '/results/', 'sw_' + f'{n_stk}'

    portfolio_sim = PortfolioSimulator(n_stk=n_stk, n_obs=n_obs_ar[0], n_reb=n_reb_ar[0],
                                       algo=algo, wght_mtd=wght_mtd, last_price_df=last_price_df,
                                       return_df=return_df, volat_df=real_vol_df, mktcap_df=mktcap_df,
                                       bema_ret_df=0 * mkt_ret_df, endow=endow, idx_start=max(n_obs_ar), lag=1,
                                       trsctn_fee_fix=trsctn_fee_fix, trsctn_fee_prop=trsctn_fee_prop,
                                       risk_avers_factor=np.inf,
                                       multi_proc=True, cv_opt_bw=False,
                                       results_dir=results_dir, results_tag=results_tag,
                                       save_stk_hist=False)

    # %% Simulate
    start = time.time()

    # Main loop
    n_sim = len(n_obs_ar) * len(n_reb_ar)
    n_done = 0
    for n_obs in n_obs_ar:
        for n_reb in n_reb_ar:
            portfolio_sim.n_obs = n_obs
            portfolio_sim.n_reb = n_reb

            print(f'\n --- Running sim. {n_done + 1} of {n_sim} ---')

            portfolio_sim.backtest()

            end_curr = time.time()
            elapsed = end_curr - start  # Total time elapsed

            # Analyze portfolio performance
            portfolio_sim.analyze(mkt_ret_df)

            # Print results to text file
            portfolio_sim.print_results()

            n_done += 1
            elapsed_mean = elapsed / n_done

            print('\n --- Total time elapsed: {0} (average sim. time: {1}) --- \n'.format(
                time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed)),
                time.strftime('%Mm %Ss', time.gmtime(elapsed_mean))))

    # %% Plot
    for wm in wght_mtd:
        results_df = portfolio_sim.load_results(wght_mtd=wm, filename=None)
        pf_plot.plot_heatmap(results_df, 'alpha', descr=algo + ', ' + wm)


if __name__ == '__main__':
    main()
