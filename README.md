# Weekly Covered Call Backtester

Simulates weekly Monday → Friday-expiry at-the-money or slightly out-of-the-money covered call strategies vs. buy-and-hold, estimated historical option prices using Binomial / Black-Scholes, assignment & rebuy logic, P&L comparison charts.

https://weekly-covered-call-backtester.streamlit.app

![Main Input Controls](screenshots/main-inputs.png)  
*Backtest parameters & estimated call premiums (Black-Scholes or Binomial model using user-assumed IV)*

![Stock Price History](screenshots/stock-price-history.png)  
*Underlying stock price history over selected backtest period*

![Dashboard Summary](screenshots/dashboard-summary.png)  
*Core metrics: total premium collected, net liquidation value, max capital required, strategy vs. buy & hold summary*

![Weekly Details Table](screenshots/weekly-table.png)  
*Per-week breakdown: strikes, premiums, assignment status, weekly P&L, running capital, yield on cost, NAV*

![PnL Comparison Chart](screenshots/pnl-comparison.png)  
*Cumulative profit & loss over time: strategy vs. buy-and-hold (absolute $ and %)*

![NAV Comparison Chart](screenshots/nav-comparison.png)  
*Net asset value progression: strategy vs. buy-and-hold across weeks*
