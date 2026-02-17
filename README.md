# Weekly Covered Call Backtester

Simulates Friday-expiry weekly ATM/OTM covered calls (Monday open → Friday close) vs buy-and-hold with cash-collected premiums, assignment & rebuy logic, and P&L comparison charts.

Live: https://weekly-covered-call-backtester.streamlit.app

![Main Input Controls](screenshots/main-inputs.png)  
*Backtest parameters & estimated call premiums (Black-Scholes or Binomial model using user-assumed IV)*

![Stock Price History](screenshots/stock-price-history.png)  
*Underlying stock daily close prices over the selected backtest period*

![Dashboard Summary](screenshots/dashboard-summary.png)  
*Core metrics: total premium collected, net liquidation value, max capital required, strategy vs buy & hold summary*

![Weekly Details Table](screenshots/weekly-table.png)  
*Per-week breakdown: strikes, premiums, assignment status, weekly P&L, running capital, yield, NAV*

![PnL Comparison Chart](screenshots/pnl-comparison.png)  
*Cumulative profit & loss over time: strategy vs buy-and-hold (absolute $ and %)*

![NAV Comparison Chart](screenshots/nav-comparison.png)  
*Net asset value progression: strategy vs buy-and-hold across weeks*
