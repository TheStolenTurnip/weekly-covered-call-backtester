import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st
import numpy as np
from scipy.stats import norm
import math
import yfinance as yf
from collections import Counter
import time
from zoneinfo import ZoneInfo


def black_scholes_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def binomial_american_call(S, K, T, r, sigma, n=100):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0)
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    stock = np.array([S * (u ** i) * (d ** (n - i)) for i in range(n + 1)])
    option = np.maximum(stock - K, 0)

    for step in range(n - 1, -1, -1):
        for i in range(step + 1):
            option[i] = discount * (p * option[i + 1] + (1 - p) * option[i])
            stock_i = S * (u ** i) * (d ** (step - i))
            option[i] = max(option[i], stock_i - K)
    return option[0]


st.set_page_config(page_title="Weekly ATM / Slightly OTM Covered Calls Backtester", layout="wide")


# ── Clock: Local time + NY time + difference (fixed calculation) ──────────
col_title, col_clock = st.columns([7, 1.5])  # Wider title to avoid cut-off

with col_title:
    st.title("Weekly ATM (Or Slightly OTM) Covered Calls Backtester")
    st.caption("Simulated weekly covered calls vs buy & hold — premiums are cash (not reinvested)")

with col_clock:
    st.components.v1.html(
        """
        <div id="clock" style="text-align: right; font-family: monospace; color: white; padding: 8px 12px; background: rgba(30,30,30,0.7); border-radius: 8px; border: 1px solid #444; max-width: 220px; margin-left: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.5); margin-top: 0;">
            <div style="font-size: 1.0em; color: #ddd; margin-bottom: 2px;">Your Local Time</div>
            <div id="localTime" style="font-size: 1.8em; font-weight: bold; letter-spacing: 1px;"></div>
            
            <div style="font-size: 1.0em; color: #ddd; margin: 8px 0 2px 0;">NY Eastern</div>
            <div id="nyTime" style="font-size: 1.5em; font-weight: bold; letter-spacing: 1px;"></div>
            
            <div id="diff" style="font-size: 0.9em; margin-top: 6px; font-weight: 500;"></div>
        </div>

        <script>
            function updateClock() {
                const now = new Date();

                // Local time
                const local = now.toLocaleTimeString('en-US', {
                    hour12: false,
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                });
                document.getElementById('localTime').innerText = local;

                // NY time
                const ny = now.toLocaleTimeString('en-US', {
                    timeZone: 'America/New_York',
                    hour12: false,
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                });
                document.getElementById('nyTime').innerText = ny + ' ET';

                // Accurate difference using Intl
                const formatter = new Intl.DateTimeFormat('en-US', {
                    timeZone: 'America/New_York',
                    timeZoneName: 'shortOffset'
                });
                const nyParts = formatter.formatToParts(now);
                const nyOffsetStr = nyParts.find(p => p.type === 'timeZoneName').value; // e.g. "GMT-5" or "GMT-4"
                const nyOffset = parseInt(nyOffsetStr.replace('GMT', '')) || 0;

                const localOffset = -now.getTimezoneOffset() / 60; // positive for east of UTC
                const diff = Math.round(localOffset - nyOffset);

                let diffText = '';
                if (diff > 0) {
                    diffText = `Your are ${diff} hours ahead of NY`;
                } else if (diff < 0) {
                    diffText = `Your are ${diff} hours behind NY`;
                } else {
                    diffText = 'Same as NY time';
                }
                document.getElementById('diff').innerText = diffText;
            }

            updateClock();
            setInterval(updateClock, 1000);
        </script>
        """,
        height=150  # Reduced height to fit better
    )

# ── Inputs ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.text_input("Ticker", value="BMNR").upper().strip()

real_increment = 0.50
detected_message = ""
if symbol:
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if expirations:
            chain = ticker.option_chain(expirations[0])
            calls = chain.calls
            strikes = sorted(calls['strike'].unique())
            if len(strikes) >= 3:
                diffs = [strikes[i+1] - strikes[i] for i in range(len(strikes)-1)]
                most_common = Counter(diffs).most_common(1)[0][0]
                if most_common > 0:
                    real_increment = most_common
                    detected_message = f"Detected strike increment from current chain: **${real_increment:.2f}** (nearest expiration)"
    except Exception:
        pass
if detected_message:
    st.info(detected_message)

strike_increment_options = [0.25, 0.50, 1.00, 2.50, 5.00, 7.50, 10.00, 12.50]
strike_increment = st.selectbox(
    "Strike price increment",
    options=strike_increment_options,
    index=strike_increment_options.index(real_increment) if real_increment in strike_increment_options else 1,
    format_func=lambda x: f"${x:.2f}",
    help="Strike price increments detected from Yahoo Finance may differ from the actual options chain. Always check the full strike display offered by your broker."
)

with col2:
    iv_percent = st.number_input("Assumed IV (%)", 10.0, 300.0, value=75.0, step=5.0,
                                 help="Implied volatility used to estimate call premiums (check your options chain)") / 100.0
with col3:
    num_shares = st.number_input("Shares (1 lot)", min_value=100, value=100, step=100,
                                help="Number of shares held / contracts written")
option_style = st.radio(
    "Option Pricing Model",
    options=["European (Black-Scholes)", "American (Binomial Tree)"],
    index=1,
)

col_a, col_b = st.columns(2)
with col_a:
    use_date_range = st.checkbox("Use custom date range", value=False)
with col_b:
    reopen_if_assigned = st.checkbox("Re-open after assignment", value=True,
                                     help="Buy back the same number of shares next Monday (open) if assigned")

# ── Date inputs (with keys) ───────────────────────────────────────────────
entry_date = st.date_input("Entry date (start)", value=datetime(2025, 7, 14),
                           disabled=not use_date_range,
                           help="Start of backtest period")
exit_date = st.date_input("Exit date (end)", value=datetime(2026, 2, 13),
                          disabled=not use_date_range,
                          help="End of backtest period")

if use_date_range and entry_date >= exit_date:
    st.error("Entry date must be before exit date.")
    st.stop()


# ── Fetch stock data ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600 * 24 * 7)
def fetch_stock_data(symbol):
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="max", interval="1d")
            if df.empty:
                return None
            df = df.reset_index()
            df = df.rename(columns={'Date': 'date', 'Open': 'open', 'Close': 'close'})
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            return df[['date', 'open', 'close']]
        except Exception as e:
            if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                time.sleep(10 * (attempt + 1))
            else:
                st.error(f"Error fetching data: {e}")
                return None
    st.error("Failed after 3 retries due to rate limiting.")
    return None


df = None
if symbol:
    df = fetch_stock_data(symbol)
    if df is not None and not df.empty:
        full_df = df.copy()

        st.subheader(f"{symbol} Daily Close Price (Full History)")

        price_fig = px.line(full_df, x='date', y='close',
                            title=f"{symbol} Daily Close — Full History")

        price_fig.update_layout(hovermode="x unified", dragmode='pan')

        if use_date_range:
            start_dt = pd.to_datetime(entry_date)
            end_dt = pd.to_datetime(exit_date)
            margin = timedelta(days=30)

            view_start = max(full_df['date'].min(), start_dt - margin)
            view_end = min(full_df['date'].max(), end_dt + margin)

            price_fig.update_xaxes(range=[view_start, view_end])

            price_fig.add_vrect(
                x0=start_dt, x1=end_dt,
                fillcolor="rgba(173, 216, 230, 0.2)",
                line_width=0,
                layer="below"
            )
            price_fig.add_vline(x=start_dt, line_dash="dash", line_color="blue")
            price_fig.add_vline(x=end_dt, line_dash="dash", line_color="blue")

        st.plotly_chart(price_fig, use_container_width=True)


# ── Backtest (your original code) ─────────────────────────────────────────
if st.button("Run Weekly Backtest", type="primary"):
    if df is None or df.empty:
        st.error("No data available.")
        st.stop()

    backtest_df = df.copy()

    if use_date_range:
        start_dt = pd.to_datetime(entry_date)
        end_dt = pd.to_datetime(exit_date)

        backtest_df = backtest_df[
            (backtest_df['date'] >= start_dt) &
            (backtest_df['date'] <= end_dt)
        ]

        if backtest_df.empty:
            st.error("No data in selected date range.")
            st.stop()

    initial_price = backtest_df['close'].iloc[0]
    initial_capital = initial_price * num_shares
    current_capital = initial_capital
    running_max_capital = initial_capital

    bh_start_date = backtest_df['date'].min().strftime('%Y-%m-%d')
    bh_end_date = backtest_df['date'].max().strftime('%Y-%m-%d')

    backtest_df.set_index('date', inplace=True)
    df_weekly = backtest_df.resample('W-FRI').agg({'open': 'first', 'close': 'last'}).dropna().reset_index()

    strategy_results = []
    holding_stock = True
    total_premium = 0.0
    cumulative_premium = 0.0
    remaining_shares = num_shares
    cost_basis_per_share = initial_price

    for i in range(len(df_weekly)):
        row = df_weekly.iloc[i]
        monday_open = row['open']
        friday_close = row['close']
        date = row['date']

        if not holding_stock:
            strategy_results.append({
                'week': date.strftime('%Y-%m-%d'),
                'monday_open': None,
                'friday_close': friday_close,
                'strike': None,
                'premium_collected': 0.0,
                'cumulative_premium': cumulative_premium,
                'assignment_gain': 0.0,
                'assigned': False,
                'reentry_price': None,
                'cost_basis_per_share': None,
                'missed_upside': 0.0,
                'working_capital': 0.0,
                'yield_on_cost': 0.0,
                'running_max_capital': running_max_capital,
                'net_liq_value': 0.0,
            })
            continue

        this_week_working_capital = remaining_shares * cost_basis_per_share

        strike = math.ceil(monday_open / strike_increment) * strike_increment
        T = 7 / 365.0
        r = 0.05

        if option_style == "European (Black-Scholes)":
            premium = black_scholes_call(monday_open, strike, T, r, iv_percent)
        else:
            premium = binomial_american_call(monday_open, strike, T, r, iv_percent, n=100)

        premium_collected = premium * 100 * (num_shares // 100)
        total_premium += premium_collected
        cumulative_premium += premium_collected

        assigned = friday_close >= strike
        assignment_gain = 0.0
        missed_upside = 0.0
        reentry_price = None

        if assigned:
            assignment_gain = (strike - monday_open) * num_shares
            missed_upside = max(0, (friday_close - strike) * num_shares)

            current_capital = 0
            remaining_shares = 0
            cost_basis_per_share = None

            if reopen_if_assigned and i < len(df_weekly) - 1:
                next_monday_open = df_weekly.iloc[i + 1]['open']
                reentry_price = next_monday_open
                remaining_shares = num_shares
                cost_basis_per_share = next_monday_open
                current_capital = next_monday_open * num_shares
                running_max_capital = max(running_max_capital, current_capital)
            else:
                holding_stock = False

        running_max_capital = max(running_max_capital, current_capital)

        current_position_value = remaining_shares * friday_close
        net_liq_value = cumulative_premium + current_position_value + assignment_gain

        strategy_results.append({
            'week': date.strftime('%Y-%m-%d'),
            'monday_open': monday_open,
            'friday_close': friday_close,
            'strike': strike,
            'premium_collected': premium_collected,
            'cumulative_premium': cumulative_premium,
            'assignment_gain': assignment_gain,
            'assigned': assigned,
            'reentry_price': reentry_price,
            'cost_basis_per_share': cost_basis_per_share,
            'missed_upside': missed_upside,
            'working_capital': this_week_working_capital,
            'yield_on_cost': (premium_collected / this_week_working_capital * 100) if this_week_working_capital > 0 else 0.0,
            'running_max_capital': running_max_capital,
            'net_liq_value': net_liq_value,
        })

    df_strategy = pd.DataFrame(strategy_results)

    for col in ['assignment_gain', 'reentry_price', 'cost_basis_per_share',
                'missed_upside', 'working_capital', 'yield_on_cost', 'net_liq_value']:
        df_strategy[col] = df_strategy[col].fillna(0.0)

    final_remaining_shares = remaining_shares
    final_cost_basis = cost_basis_per_share if final_remaining_shares > 0 else None

    # ── Buy & Hold Summary ────────────────────────────────────────────────────
    bh_start_price = initial_price
    bh_end_price = backtest_df['close'].iloc[-1]
    bh_pnl_dollar = (bh_end_price - bh_start_price) * num_shares
    bh_pnl_pct = ((bh_end_price / bh_start_price) - 1) * 100 if bh_start_price > 0 else 0
    bh_net_liq = bh_end_price * num_shares

    st.subheader("Buy & Hold Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Start Date", bh_start_date,
                  help="Start of the backtest period")
        st.metric("End Date", bh_end_date,
                  help="End of the backtest period")
    with c2:
        st.metric("Start Price", f"${bh_start_price:,.2f}",
                  help="Stock price at start")
        st.metric("End Price", f"${bh_end_price:,.2f}",
                  help="Stock price at end")
    with c3:
        st.metric("Net Liq Value", f"${bh_net_liq:,.2f}",
                  help="Final value of position (shares × end price)")
        st.metric("P&L ($)", f"${bh_pnl_dollar:,.2f}",
                  delta=f"{bh_pnl_pct:+.2f}%")

    # ── Strategy Summary ──────────────────────────────────────────────────────
    final_position_value = final_remaining_shares * bh_end_price if final_remaining_shares > 0 else 0.0
    shares_pnl = final_position_value - (final_cost_basis * final_remaining_shares if final_cost_basis else 0)
    total_pnl_incl_premium = shares_pnl + total_premium
    total_pnl_pct_vs_max = (total_pnl_incl_premium / running_max_capital) * 100 if running_max_capital > 0 else 0
    strategy_net_liq = total_premium + final_position_value

    st.subheader("Strategy Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Max Capital Required", f"${running_max_capital:,.2f}",
                  help="Highest amount of capital actually tied up in shares at any point")
        st.metric("Total Premium Collected", f"${total_premium:,.2f}",
                  help="Sum of all estimated call premiums received as cash")
    with c2:
        st.metric("Current Position",
                  f"{final_remaining_shares} @ ${final_cost_basis:,.2f}" if final_cost_basis else "0",
                  help="Shares still held and their average cost basis (0 if assigned)")
        st.metric("Position Value", f"${final_position_value:,.2f}",
                  help="Value of remaining shares at the end price (0 if assigned)")
    with c3:
        st.metric("Net Liq Value", f"${strategy_net_liq:,.2f}",
                  help="Total account value at end: premiums + value of remaining shares")
        st.metric(
            "Strategy P&L ($)",
            f"${total_pnl_incl_premium:,.2f}",
            delta=f"{total_pnl_pct_vs_max:+.2f}%",
            help="Net gain including premiums + assignment proceeds + share P&L (return vs max capital required)"
        )

        st.caption(
            "Note: % Return uses max capital ever required as denominator (conservative return on peak commitment). Weekly yield uses each week's working capital."
        )

    # ── Strategy Weekly Details ───────────────────────────────────────────────
    display_cols = [
        'week', 'monday_open', 'friday_close', 'strike',
        'premium_collected', 'cumulative_premium', 'assignment_gain', 'assigned',
        'reentry_price', 'cost_basis_per_share', 'missed_upside',
        'working_capital', 'yield_on_cost', 'net_liq_value'
    ]
    df_display = df_strategy[display_cols]

    # Add this line ↓
    df_display.index = range(1, len(df_display) + 1)   # ← Makes left column start at 1

    st.subheader("Strategy Weekly Details")
    st.dataframe(
        df_display.style.format({
            'premium_collected': '${:,.2f}',
            'cumulative_premium': '${:,.2f}',
            'assignment_gain': '${:,.2f}',
            'monday_open': '${:.2f}',
            'friday_close': '${:.2f}',
            'strike': '${:.2f}',
            'reentry_price': '${:.2f}',
            'cost_basis_per_share': '${:.2f}',
            'missed_upside': '${:,.2f}',
            'working_capital': '${:,.2f}',
            'yield_on_cost': '{:.2f}%',
            'net_liq_value': '${:,.2f}',
        }),
        column_config={
            'week': st.column_config.Column("Week", help="Week ending date (Friday)"),
            'monday_open': st.column_config.Column("Monday Open", help="Stock open price used for strike"),
            'friday_close': st.column_config.Column("Friday Close", help="Stock price at expiration"),
            'strike': st.column_config.Column("Strike", help="Call strike sold (rounded to selected increment)"),
            'premium_collected': st.column_config.Column("Premium", help="Estimated cash from selling the call"),
            'cumulative_premium': st.column_config.Column("Cum. Premium", help="Running total of premiums"),
            'assignment_gain': st.column_config.Column(
                "Assignment Gain",
                help="Net profit from assignment this week (strike - open price) × shares — $0 if not assigned"
            ),
            'assigned': st.column_config.Column("Assigned", help="Call was ITM or ATM and shares called away"),
            'reentry_price': st.column_config.Column("Re-entry Price", help="Price paid to buy back next week (if reopened)"),
            'cost_basis_per_share': st.column_config.Column("Cost Basis/Share", help="Average cost per share of current position"),
            'missed_upside': st.column_config.Column("Missed Upside ($)", help="Extra gain given up on assigned weeks: (close - strike) × shares"),
            'working_capital': st.column_config.Column(
                "Working Capital",
                help="Cost of shares held during that week (purchase cost × shares)"
            ),
            'yield_on_cost': st.column_config.Column(
                "Yield on Cost (%)",
                help="Weekly premium ÷ working capital this week × 100"
            ),
            'net_liq_value': st.column_config.Column(
                "Net Liq Value",
                help="Total account value at end of week: shares value + cumulative premiums + assignment gain that week"
            ),
        },
        use_container_width=True,
    )

    # ── PnL Comparison Chart ──────────────────────────────────────────────────
    st.subheader("PnL Comparison Over Time")

    chart_df = df_strategy[['week', 'running_max_capital', 'cumulative_premium',
                            'cost_basis_per_share', 'friday_close']].copy()

    chart_df['remaining_shares'] = final_remaining_shares
    chart_df['current_position_value'] = chart_df['remaining_shares'] * chart_df['friday_close']
    chart_df['current_invested'] = chart_df['remaining_shares'] * chart_df['cost_basis_per_share']
    chart_df['Strategy PnL ($)'] = chart_df['cumulative_premium'] + chart_df['current_position_value'] - chart_df['current_invested']
    chart_df['Strategy PnL (%)'] = (chart_df['Strategy PnL ($)'] / chart_df['running_max_capital']) * 100

    chart_df['Buy & Hold PnL ($)'] = (chart_df['friday_close'] - initial_price) * num_shares
    chart_df['Buy & Hold PnL (%)'] = ((chart_df['friday_close'] / initial_price) - 1) * 100

    chart_long_dollar = chart_df.melt(id_vars=['week'],
                                      value_vars=['Buy & Hold PnL ($)', 'Strategy PnL ($)'],
                                      var_name='Metric', value_name='Value')

    chart_long_pct = chart_df.melt(id_vars=['week'],
                                   value_vars=['Buy & Hold PnL (%)', 'Strategy PnL (%)'],
                                   var_name='Metric', value_name='Value')

    fig_pnl = px.line(chart_long_dollar, x='week', y='Value', color='Metric',
                      title="PnL Comparison Over Time",
                      labels={'week': 'Week Ending'},
                      markers=True)

    for trace in fig_pnl.data:
        if trace.name in ['Buy & Hold PnL ($)', 'Strategy PnL ($)']:
            trace.update(hovertemplate='%{x|%Y-%m-%d}<br>%{fullData.name}: $%{y:,.2f}<extra></extra>')

    fig_pnl.add_scatter(x=chart_long_pct[chart_long_pct['Metric'] == 'Buy & Hold PnL (%)']['week'],
                        y=chart_long_pct[chart_long_pct['Metric'] == 'Buy & Hold PnL (%)']['Value'],
                        mode='lines+markers',
                        name='Buy & Hold PnL (%)',
                        line_dash='dot',
                        yaxis='y2')

    fig_pnl.add_scatter(x=chart_long_pct[chart_long_pct['Metric'] == 'Strategy PnL (%)']['week'],
                        y=chart_long_pct[chart_long_pct['Metric'] == 'Strategy PnL (%)']['Value'],
                        mode='lines+markers',
                        name='Strategy PnL (%)',
                        line_dash='dot',
                        yaxis='y2')

    fig_pnl.update_layout(
        xaxis_title="Week Ending",
        yaxis_title="PnL ($)",
        yaxis2=dict(title="PnL (%)", overlaying='y', side='right'),
        legend_title="Metric",
        hovermode="x unified",
        dragmode='pan',
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        )
    )

    st.plotly_chart(fig_pnl, use_container_width=True,
                    config={'scrollZoom': False, 'displayModeBar': True,
                            'modeBarButtonsToAdd': ['pan2d'],
                            'modeBarButtonsToRemove': ['zoom2d', 'lasso2d', 'select2d'],
                            'displaylogo': False})

    # ── Value Comparison Chart ────────────────────────────────────────────────
    st.subheader("Value Comparison Over Time")

    chart_df['Strategy NAV'] = chart_df['cumulative_premium'] + chart_df['current_position_value']
    chart_df['Buy & Hold NAV'] = chart_df['friday_close'] * num_shares

    chart_long_value = chart_df.melt(id_vars=['week'],
                                     value_vars=['Buy & Hold NAV', 'Strategy NAV'],
                                     var_name='Metric', value_name='Value')

    fig_value = px.line(chart_long_value, x='week', y='Value', color='Metric',
                        title="Value Comparison Over Time",
                        labels={'week': 'Week Ending'},
                        markers=True)

    for trace in fig_value.data:
        if trace.name in ['Buy & Hold NAV', 'Strategy NAV']:
            trace.update(hovertemplate='%{x|%Y-%m-%d}<br>%{fullData.name}: $%{y:,.2f}<extra></extra>')

    fig_value.update_layout(
        xaxis_title="Week Ending",
        yaxis_title="Net Liquidation Value ($)",
        legend_title="Metric",
        hovermode="x unified",
        dragmode='pan',
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        )
    )

    st.plotly_chart(fig_value, use_container_width=True,
                    config={'scrollZoom': False, 'displayModeBar': True,
                            'modeBarButtonsToAdd': ['pan2d'],
                            'modeBarButtonsToRemove': ['zoom2d', 'lasso2d', 'select2d'],
                            'displaylogo': False})

    st.info("""
    - **Buy & Hold NAV** is shares × current price
    - **Strategy NAV** is cumulative premiums + shares value
    - Click + drag pans the chart; scroll wheel or box select to zoom
    - Hover shows exact $ values (rounded to 2 decimals)
    """)

    st.markdown("---")

    st.markdown("""
    ### Important Disclaimer
    - **This is NOT financial advice.** The results shown are **simulated estimates** based on the Black-Scholes model and user-provided implied volatility assumptions.
    - Actual historical options premiums, bid/ask spreads, transaction costs, dividends, borrow fees, assignment timing, taxes, and slippage are **not** included.
    - Past performance (even simulated) does **not** predict future results. Options trading involves substantial risk of loss and is not suitable for all investors.
    - Always do your own research and consult a qualified financial advisor before making any investment decisions.
    """)

    st.markdown("""
    ### How Weekly Call Premiums Are Calculated
    - Premiums are **estimated** using:
      - **European**: Black-Scholes model (no early exercise)
      - **American**: Binomial tree (Cox-Ross-Rubinstein) with 100 steps (allows early exercise check)
    - Inputs used:
      - Spot price = Monday's open price
      - Strike = rounded to selected increment above Monday open
      - Time to expiration = 7 days (exactly 7/365 years)
      - Risk-free rate = fixed 5% (0.05)
      - Implied volatility = **user-entered value** — **not** real historical IV
      - No dividends are modeled
    - Real weekly options prices would differ due to: actual market IV, skew, supply/demand, bid-ask spreads, and intraday movements.
    - This is a **simplified educational/illustrative tool**, not a precise replication of tradable options.
    """)

    st.caption("Built with assistance from Grok by xAI")