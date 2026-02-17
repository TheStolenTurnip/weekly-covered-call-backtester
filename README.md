# Weekly Covered Call Backtester

MIT License

Simulated weekly covered calls (ATM/OTM) vs. buy-and-hold on selected stocks. Premiums are treated as cash collected (not reinvested). Compares strategy P&L, assignment handling, yield on cost, and more — with live charts (delayed data).

## Live App

Run the backtester directly in your browser — no installation required:

https://weekly-covered-call-backtester.streamlit.app

## Features

- Weekly ATM or OTM covered call simulation
- Assignment logic and rebuy handling
- Cumulative premiums, position value, running capital
- P&L comparison vs. buy-and-hold (absolute $ and %)
- Interactive charts: strategy vs. B&H over time
- Customizable stock, time period, strike increments, etc.

Built with Streamlit + Python. Data sourced via public APIs (delayed).
