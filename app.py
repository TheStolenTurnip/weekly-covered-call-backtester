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

def black_scholes_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def binomial_american_call(S, K, T, r, sigma, n=50):
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

# ── Clock with Market Countdown ──────────────────────────────────────────
col_title, col_clock = st.columns([7, 1.5])

with col_title:
    st.title("Weekly ATM / Slightly OTM Covered Calls Backtester")
    st.caption("Simulated weekly covered calls vs buy & hold — premiums are cash (not reinvested)")

with col_clock:
    st.components.v1.html(
        """
        <div id="clock" style="text-align: right; font-family: monospace; color: white; padding: 10px 14px; background: rgba(30,30,30,0.8); border-radius: 10px; border: 1px solid #555; max-width: 300px; margin-left: auto; box-shadow: 0 3px 10px rgba(0,0,0,0.6);">
            <div style="font-size: 1.0em; color: #ccc;">Your Local Time</div>
            <div id="localTime" style="font-size: 1.9em; font-weight: bold;"></div>
            
            <div style="font-size: 1.0em; color: #ccc; margin-top: 8px;">US Eastern / Market</div>
            <div id="nyTime" style="font-size: 1.6em; font-weight: bold;"></div>
            
            <div id="marketStatus" style="font-size: 1.15em; margin-top: 8px; font-weight: 600;"></div>
            <div id="countdown" style="font-size: 1.05em; color: #0f0;"></div>
        </div>

        <script>
            // Hardcoded NYSE full market closures 2025–2028 (from official NYSE announcements)
            const marketHolidays = new Set([
                // 2025
                "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
                "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
                // 2026 (current year in your test)
                "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
                "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
                // 2027
                "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
                "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
                // 2028 (partial, for safety)
                "2028-01-01", "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29"
            ]);

            function isHoliday(date) {
                const yyyy = date.getFullYear();
                const mm = String(date.getMonth() + 1).padStart(2, '0');
                const dd = String(date.getDate()).padStart(2, '0');
                const key = `${yyyy}-${mm}-${dd}`;
                return marketHolidays.has(key);
            }

            function updateClock() {
                const now = new Date();
                const et = new Date(now.toLocaleString('en-US', {timeZone: 'America/New_York'}));

                document.getElementById('localTime').innerText = now.toLocaleTimeString('en-US', {hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'});

                document.getElementById('nyTime').innerText = et.toLocaleTimeString('en-US', {hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'}) + ' ET';

                const day = et.getDay();
                const hour = et.getHours();
                const min = et.getMinutes();
                const isWeekday = day >= 1 && day <= 5;

                let status = "Market Closed";
                let countdownText = "";
                let color = "#ccc";

                if (isHoliday(et)) {
                    status = "Market Holiday";
                    color = "#ff9800";  // orange
                } else if (isWeekday) {
                    const todayOpen = new Date(et);
                    todayOpen.setHours(9, 30, 0, 0);
                    const todayClose = new Date(et);
                    todayClose.setHours(16, 0, 0, 0);

                    if (et >= todayOpen && et < todayClose) {
                        status = "Market Open";
                        color = "#0f0";
                        const timeToClose = todayClose - et;
                        const h = Math.floor(timeToClose / 3600000);
                        const m = Math.floor((timeToClose % 3600000) / 60000);
                        countdownText = `Closes in ${h}h ${m}m`;
                    } else if (et < todayOpen) {
                        const timeToOpen = todayOpen - et;
                        const h = Math.floor(timeToOpen / 3600000);
                        const m = Math.floor((timeToOpen % 3600000) / 60000);
                        countdownText = `Opens in ${h}h ${m}m`;
                    } else {
                        // After close → next open (skip weekends/holidays)
                        let next = new Date(et);
                        next.setDate(next.getDate() + 1);
                        while (next.getDay() === 0 || next.getDay() === 6 || isHoliday(next)) {
                            next.setDate(next.getDate() + 1);
                        }
                        next.setHours(9, 30, 0, 0);
                        const timeToOpen = next - et;
                        const totalH = Math.floor(timeToOpen / 3600000);
                        const d = Math.floor(totalH / 24);
                        const h = totalH % 24;
                        const m = Math.floor((timeToOpen % 3600000) / 60000);
                        countdownText = `Opens in ${d > 0 ? d + 'd ' : ''}${h}h ${m}m`;
                    }
                } else {
                    // Weekend → next Monday (skip if holiday)
                    let next = new Date(et);
                    const daysAhead = (8 - day) % 7 || 7;
                    next.setDate(next.getDate() + daysAhead);
                    while (isHoliday(next)) {
                        next.setDate(next.getDate() + 1);
                    }
                    next.setHours(9, 30, 0, 0);
                    const timeToOpen = next - et;
                    const totalH = Math.floor(timeToOpen / 3600000);
                    const d = Math.floor(totalH / 24);
                    const h = totalH % 24;
                    const m = Math.floor((timeToOpen % 3600000) / 60000);
                    countdownText = `Opens in ${d > 0 ? d + 'd ' : ''}${h}h ${m}m`;
                }

                document.getElementById('marketStatus').innerText = status;
                document.getElementById('marketStatus').style.color = color;
                document.getElementById('countdown').innerText = countdownText;
            }

            updateClock();
            setInterval(updateClock, 1000);
        </script>
        """,
        height=180
    )

# ── Inputs ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.text_input("Ticker", value="SOFI").upper().strip()

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
    "Strike increment",
    options=strike_increment_options,
    index=strike_increment_options.index(real_increment) if real_increment in strike_increment_options else 1,
    format_func=lambda x: f"${x:.2f}",
)

with col2:
    iv_percent = st.number_input("Assumed IV (%)", 10.0, 300.0, value=55.0, step=5.0,
                                 help="Implied volatility used to estimate call premiums (check your options chain)") / 100.0
with col3:
    num_shares = st.number_input("Shares (by lot)", min_value=100, value=100, step=100,
                                help="Number of shares held / contracts written")

    
option_style = st.radio("Pricing Model", ["European (Black-Scholes)", "American (Binomial)"], index=1,
                        help="European: Black-Scholes (no early exercise)\nAmerican: Binomial tree (allows early exercise check)")

col_a, col_b = st.columns(2)
with col_a:
    use_date_range = st.checkbox("Custom date range", value=False)
with col_b:
    reopen_if_assigned = st.checkbox("Re-open after assignment", value=True,
                                     help="Buy back the same number of shares next Monday (open) if assigned")

entry_date = st.date_input("Entry date (start)", value=datetime(2025, 7, 14),
                           disabled=not use_date_range,
                           help="Start of backtest period")
exit_date = st.date_input("Exit date (end)", value=datetime(2026, 2, 13),
                          disabled=not use_date_range,
                          help="End of backtest period")

if use_date_range and entry_date >= exit_date:
    st.error("Start date must be before end date.")
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
full_df = None
if symbol:
    df = fetch_stock_data(symbol)
    if df is not None and not df.empty:
        full_df = df.copy()

# ── Price History Chart (shows immediately when ticker entered) ───────
if use_date_range and entry_date and exit_date:
    start_str = entry_date.strftime("%Y-%m-%d")
    end_str   = exit_date.strftime("%Y-%m-%d")
    subheader_title = f"{symbol} Daily Close Price (Selected Range)"
    fig_title       = f"{symbol} Daily Close — {start_str} to {end_str}"
else:
    subheader_title = f"{symbol} Daily Close Price (Full History)"
    fig_title       = f"{symbol} Daily Close — Full History"

st.subheader(subheader_title)

price_fig = px.line(full_df, x='date', y='close',
                    title=fig_title)

price_fig.update_layout(
    hovermode="x unified",
    dragmode='pan'
)

price_fig.update_traces(
    hovertemplate='%{x|%Y-%m-%d}<br>Close: $%{y:,.2f}<extra></extra>'
)

# ── Only add range highlight & zoom if custom range is selected ──────────
if use_date_range and entry_date and exit_date:
    start_dt = pd.to_datetime(entry_date)
    end_dt   = pd.to_datetime(exit_date)
    margin   = timedelta(days=30)

    view_start = max(full_df['date'].min(), start_dt - margin)
    view_end   = min(full_df['date'].max(), end_dt + margin)

    price_fig.update_xaxes(range=[view_start, view_end])

    price_fig.add_vrect(
        x0=start_dt,
        x1=end_dt,
        fillcolor="rgba(173, 216, 230, 0.2)",
        line_width=0,
        layer="below"
    )

st.plotly_chart(price_fig, use_container_width=True, key="price_history_chart")

# ── Backtest ──────────────────────────────────────────────────────────────
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

    initial_price = backtest_df['open'].iloc[0]
    initial_capital = initial_price * num_shares
    current_capital = initial_capital
    running_max_capital = initial_capital

    bh_start_date = backtest_df['date'].min().strftime('%Y-%m-%d')
    bh_end_date = backtest_df['date'].max().strftime('%Y-%m-%d')

    backtest_df.set_index('date', inplace=True)
    df_weekly = backtest_df.resample('W-FRI').agg({'open': 'first', 'close': 'last'}).dropna().reset_index()

    strategy_results = []
    holding_stock = True
    cash = 0.0
    cum_external_injected = 0.0
    cum_pnl = 0.0
    cum_assignment_gain = 0.0
    cum_premium_on_lot = 0.0
    total_premium = 0.0
    cumulative_premium = 0.0
    cumulative_realized = 0.0
    remaining_shares = num_shares
    current_cost_basis = initial_price
    is_new_lot = False

    for i in range(len(df_weekly)):
        row = df_weekly.iloc[i]
        monday_open = row['open']
        friday_close = row['close']
        date = row['date']

        reentry_price = None
        addl_cash_for_reentry = 0.0
        strike = None
        premium_collected = 0.0
        assignment_gain = 0.0
        assignment_proceeds = 0.0
        assigned = False
        this_week_working_capital = 0.0
        yield_on_cost = 0.0
        adj_cost_basis = 'N/A'
        weekly_pnl = 0.0
        assignment_strike = 'N/A'
        missed_upside = 0.0

        if not holding_stock:
            if reopen_if_assigned:
                reentry_price = monday_open
                reentry_cost = monday_open * num_shares
                cash_used = min(cash, reentry_cost)
                cash -= cash_used
                addl_cash_for_reentry = reentry_cost - cash_used
                cum_external_injected += addl_cash_for_reentry
                remaining_shares = num_shares
                current_cost_basis = monday_open
                cum_premium_on_lot = 0.0
                this_week_working_capital = reentry_cost
                current_capital = reentry_cost
                running_max_capital = max(running_max_capital, current_capital)
                holding_stock = True
                is_new_lot = True

        monday_open_display = monday_open if holding_stock else 'N/A'

        if holding_stock:
            this_week_working_capital = remaining_shares * current_cost_basis
            strike = math.ceil(monday_open / strike_increment) * strike_increment
            T = 7 / 365.0
            r = 0.05

            if option_style == "European (Black-Scholes)":
                premium = black_scholes_call(monday_open, strike, T, r, iv_percent)
            else:
                premium = binomial_american_call(monday_open, strike, T, r, iv_percent, n=100)

            premium_collected = premium * 100 * (remaining_shares // 100)
            total_premium += premium_collected
            cumulative_premium += premium_collected
            cum_premium_on_lot += premium_collected
            cash += premium_collected
            weekly_pnl += premium_collected
            cum_pnl += premium_collected

            adj_cost_value = current_cost_basis - (cum_premium_on_lot / remaining_shares) if remaining_shares > 0 else 0.0
            adj_cost_basis = f"{adj_cost_value:.2f}*" if is_new_lot else f"{adj_cost_value:.2f}"

            assigned = friday_close > strike
            if assigned:
                assignment_strike = strike
                assignment_proceeds = strike * remaining_shares
                cash += assignment_proceeds
                assignment_gain = (strike - current_cost_basis) * remaining_shares
                cumulative_realized += assignment_gain
                cum_assignment_gain += assignment_gain
                weekly_pnl += assignment_gain
                cum_pnl += assignment_gain
                missed_upside = max(0, (friday_close - strike) * remaining_shares)
                remaining_shares = 0
                current_cost_basis = None
                cum_premium_on_lot = 0.0
                current_capital = 0
                holding_stock = False

            yield_on_cost = (premium_collected / this_week_working_capital * 100) if this_week_working_capital > 0 else 0.0

        is_new_lot = False

        current_position_value = remaining_shares * friday_close if holding_stock else 0.0
        net_liq_value = cash + current_position_value

        strategy_results.append({
            'week': date.strftime('%Y-%m-%d'),
            'monday_open': monday_open_display,
            'friday_close': friday_close,
            'strike': strike,
            'premium_collected': premium_collected,
            'cumulative_premium': cumulative_premium,
            'assignment_gain': assignment_gain,
            'cum_assignment_gain': cum_assignment_gain,
            'assigned': assigned,
            'reentry_price': reentry_price,
            'addl_cash_for_reentry': addl_cash_for_reentry,
            'cost_basis_per_share': current_cost_basis,
            'adj_cost_basis': adj_cost_basis,
            'yield_on_cost': yield_on_cost,
            'weekly_pnl': weekly_pnl,
            'cum_pnl': cum_pnl,
            'net_liq_value': net_liq_value,
            'remaining_shares_at_end': remaining_shares,
            'running_max_capital': running_max_capital,
            'cum_external_injected': cum_external_injected,
            'assignment_strike': assignment_strike,
            'missed_upside': missed_upside,
            'working_capital': this_week_working_capital,
        })

    df_strategy = pd.DataFrame(strategy_results)

    if df_strategy.empty:
        st.error("No weeks in backtest range.")
        st.stop()

    # ── Visual fixes for table ───────────────────────────────────────────────
    # Carry forward cost basis for visual clarity
    df_strategy['cost_basis_per_share'] = df_strategy['cost_basis_per_share'].ffill()

    # Force week 1 to show Monday open as cost basis (visual only)
    if not df_strategy.empty:
        df_strategy.loc[0, 'cost_basis_per_share'] = df_strategy.loc[0, 'monday_open']

    # Fillna numeric columns (skip Cost & Adj Cost)
    for col in ['premium_collected', 'cumulative_premium', 'assignment_gain', 'cum_assignment_gain',
                'weekly_pnl', 'cum_pnl', 'missed_upside', 'working_capital', 'yield_on_cost', 'net_liq_value']:
        df_strategy[col] = df_strategy[col].fillna(0.0)

    final_remaining_shares = remaining_shares
    final_cost_basis = current_cost_basis

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
        st.metric("Net Liquidation Value", f"${bh_net_liq:,.2f}",
                  help="Final value of position (shares × end price)")
        st.metric("P&L ($)", f"${bh_pnl_dollar:,.2f}",
                  delta=f"{bh_pnl_pct:+.2f}%")

    # ── Strategy Summary ──────────────────────────────────────────────────────
    final_position_value = final_remaining_shares * bh_end_price
    total_pnl = cash + final_position_value - (initial_capital + cum_external_injected)
    total_pnl_pct_vs_max = (total_pnl / running_max_capital) * 100 if running_max_capital > 0 else 0.0
    strategy_net_liq = cash + final_position_value

    st.subheader("Strategy Summary")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Initial Capital",
            f"${initial_capital:,.2f}",
            help="Starting capital invested: number of shares × starting stock price"
        )
        st.metric(
            "Max Capital Required",
            f"${running_max_capital:,.2f}",
            help="Highest amount of capital tied up in shares at any point during the backtest (peak commitment)"
        )
        st.metric(
            "Total Premium Collected",
            f"${total_premium:,.2f}",
            help="Sum of all call option premiums received as cash over the entire backtest period"
        )

    with c2:
        st.metric(
            "Additional Capital Injected (Cumulative)",
            f"${cum_external_injected:,.2f}",
            help="Additional capital that had to be added to repurchase shares after assignments (when re-opening is enabled)"
        )
        st.metric(
            "Current Position",
            f"{final_remaining_shares} @ ${final_cost_basis:,.2f}" if final_cost_basis else "0",
            help="Number of shares still held at the end and their average cost basis per share (0 if fully assigned by end date and not re-entered)"
        )
        st.metric(
            "Position Value",
            f"${final_position_value:,.2f}",
            help="Current market value of any remaining shares at the final closing price (excludes assignment proceeds gain/loss)"
        )

    with c3:
        st.metric(
            "Net Liquidiation Value",
            f"${strategy_net_liq:,.2f}",
            help="Final value of shares (including premiums and assignment proceeds gain/loss)"
        )
        st.metric(
            "Strategy P&L ($)",
            f"${total_pnl:,.2f}",
            delta=f"{total_pnl_pct_vs_max:+.2f}%",
            help="Net profit or loss: includes all premiums collected + final position value - total capital invested (initial + injected). % is return on max capital required (conservative peak-commitment basis)"
        )

        st.caption(
        "Note: % Return uses max capital ever required as denominator (conservative return on peak commitment). "
        "Weekly yield uses each week's working capital."
        )

    # ── Strategy Weekly Details ───────────────────────────────────────────────
    st.subheader("Strategy Weekly Details")

    # 1. Internal columns → Nice display names (for headers)
    display_names = {
        'week':                'Week',
        'monday_open':         'Mon Open',
        'friday_close':        'Fri Close',
        'strike':              'Strike',
        'premium_collected':   'Prem',
        'assignment_gain':     'Asgmt Proc',
        'weekly_pnl':          'Weekly P&L',
        'assigned':            'Assigned',
        'cumulative_premium':  'Σ Prem',
        'cum_assignment_gain': 'Σ Asgmt Proc',
        'cum_pnl':             'Σ Weekly P&L',
        'reentry_price':       'Rebuy',
        'cost_basis_per_share':'Cost',
        'adj_cost_basis':      'Adj Cost',
        'missed_upside':       'Missed $',
        'working_capital':     'Running Cap',
        'yield_on_cost':       'Y%',
        'net_liq_value':       'NAV'
    }

    # 2. Order of columns to show (same as your original core_cols)
    display_columns = [
        'week', 'monday_open', 'friday_close', 'strike',
        'premium_collected', 'assignment_gain', 'weekly_pnl', 'assigned',
        'cumulative_premium', 'cum_assignment_gain', 'cum_pnl',
        'reentry_price', 'cost_basis_per_share', 'adj_cost_basis',
        'missed_upside', 'working_capital', 'yield_on_cost', 'net_liq_value'
    ]

    # 3. Create display DataFrame with renamed columns
    table_df = df_strategy[display_columns].copy().rename(columns=display_names)

    # Reset index to start from 1 (your original style)
    table_df.index = range(1, len(table_df) + 1)

    # Fillna for numeric display columns (your original list)
    numeric_short_cols = ['Prem', 'Asgmt Proc', 'Weekly P&L', 'Σ Prem', 
                        'Σ Asgmt Proc', 'Σ Weekly P&L', 'Missed $', 
                        'Running Cap', 'NAV']
    for col in numeric_short_cols:
        if col in table_df.columns:
            table_df[col] = table_df[col].fillna(0.0)

    # 4. Your original column_config — unchanged! (all help texts & widths preserved)
    #    We just need to make sure the keys match the *new display names*
    column_config = {
        'Week': st.column_config.Column("Week", help="Week ending date (Friday)", width=80),
        'Mon Open': st.column_config.Column("Mon Open", help="Stock open price used for strike", width=65),
        'Fri Close': st.column_config.Column("Fri Close", help="Stock price at expiration", width=65),
        'Strike': st.column_config.Column("Strike", help="Call strike sold (rounded to selected increment)", width=55),
        'Prem': st.column_config.Column("Prem", help="Estimated cash from selling the call (dollar per lot = 100 shares)", width=55),
        'Asgmt Proc': st.column_config.Column(
            "Asgmt Proc",
            help="Net profit from assignment this week (strike - open price) × shares — $0 if not assigned\n\n "
            "Can be negative if strike below adj cost basis",
            width=75
        ),
        'Weekly P&L': st.column_config.Column(
            "Weekly P&L",
            help="Net weekly P&L: premium + assignment gain/loss",
            width=80
        ),
        'Assigned': st.column_config.Column("Assigned", help="Call was ITM or ATM and shares called away", width=65),
        'Σ Prem': st.column_config.Column("Σ Prem", help="Running total of premiums", width=80),
        'Σ Asgmt Proc': st.column_config.Column(
            "Σ Asgmt Proc",
            help="Running total of assignment gains/losses over all weeks",
            width=85
        ),
        'Σ Weekly P&L': st.column_config.Column(
            "Σ Weekly P&L",
            help="Running total P&L over all weeks (sum of weekly P&L)",
            width=85
        ),
        'Rebuy': st.column_config.Column("Rebuy", help="Price paid to buy back this week (if reopened after prior assignment)", width=55),
        'Cost': st.column_config.Column("Cost", help="Average cost per share of current position", width=60),
        'Adj Cost': st.column_config.Column(
            "Adj Cost",
            help="Effective cost basis (raw cost - premiums on this lot / shares); * = new lot after re-buy",
            width=60
        ),
        'Missed $': st.column_config.Column(
            "Missed $",
            help="Extra gain given up on assigned weeks: (close - strike) × shares",
            width=60
        ),
        'Running Cap': st.column_config.Column(
            "Running Cap",
            help="Cost of shares held during that week (purchase cost × shares)",
            width=85
        ),
        'Y%': st.column_config.Column(
            "Y%",
            help="Weekly premium ÷ working capital this week × 100",
            width=55
        ),
        'NAV': st.column_config.Column(
            "NAV",
            help="Total account value at end of week: cash (premiums + proceeds) + position value (or Net Liquidation Value)",
            width=100
        ),
    }

    # 5. Formatting (same logic as before, just using display names)
    st.dataframe(
        table_df.style.format({
            'Mon Open':    '${:,.2f}',
            'Fri Close':   '${:,.2f}',
            'Strike':      '${:,.2f}',
            'Prem':        '${:,.2f}',
            'Asgmt Proc':  '${:,.2f}',
            'Weekly P&L':  '${:,.2f}',
            'Σ Prem':      '${:,.2f}',
            'Σ Asgmt Proc':'${:,.2f}',
            'Σ Weekly P&L':'${:,.2f}',
            'Rebuy':       '${:,.2f}',
            'Cost':        '${:,.2f}',
            'Adj Cost': lambda x: (
                f"{float(x.rstrip('*')):,.2f}" + 
                ('*' if isinstance(x, str) and x.endswith('*') else '')
            ) if pd.notna(x) else "",
            'Y%':          '{:,.2f}%',
            'Missed $':    '${:,.2f}',
            'Running Cap': '${:,.2f}',
            'NAV':         '${:,.2f}',
        }),
        column_config=column_config,
        hide_index=False,
        use_container_width=True,
    )

    # ── PnL Comparison Over Time ──────────────────────────────────────────────
    st.subheader("PnL Comparison Over Time")

    chart_df = df_strategy[['week', 'running_max_capital', 'cumulative_premium',
                            'cost_basis_per_share', 'friday_close', 'remaining_shares_at_end', 
                            'cum_external_injected', 'net_liq_value', 'cum_pnl']].copy()

    chart_df['remaining_shares'] = chart_df['remaining_shares_at_end']
    chart_df['current_position_value'] = chart_df['remaining_shares'] * chart_df['friday_close']
    chart_df['Strategy PnL ($)']   = chart_df['net_liq_value'] - (initial_capital + chart_df['cum_external_injected'])
    chart_df['Strategy PnL (%)']   = (chart_df['Strategy PnL ($)'] / chart_df['running_max_capital']) * 100

    chart_df['Buy & Hold PnL ($)'] = (chart_df['friday_close'] - initial_price) * num_shares
    chart_df['Buy & Hold PnL (%)'] = ((chart_df['friday_close'] / initial_price) - 1) * 100

    # ── Dollar lines first (solid) ─────────────────────────────────────────────
    chart_long_dollar = chart_df.melt(
        id_vars=['week'],
        value_vars=['Strategy PnL ($)', 'Buy & Hold PnL ($)'],
        var_name='Metric', value_name='Value'
    )

    # ── Percent lines second (dashed) ──────────────────────────────────────────
    chart_long_pct = chart_df.melt(
        id_vars=['week'],
        value_vars=['Strategy PnL (%)', 'Buy & Hold PnL (%)'],
        var_name='Metric', value_name='Value'
    )

    # Create figure with ordered traces
    fig_pnl = px.line(
        chart_long_dollar,
        x='week',
        y='Value',
        color='Metric',
        title="PnL Comparison Over Time",
        labels={'week': 'Week Ending'},
        markers=True,
        color_discrete_map={
            'Strategy PnL ($)':   '#6baed6',   # light blue
            'Buy & Hold PnL ($)': '#08519c'    # dark blue
        }
    )

    # Add percent lines **after** dollar lines → they appear lower in legend
    fig_pnl.add_scatter(
        x=chart_long_pct[chart_long_pct['Metric'] == 'Strategy PnL (%)']['week'],
        y=chart_long_pct[chart_long_pct['Metric'] == 'Strategy PnL (%)']['Value'],
        mode='lines+markers',
        name='Strategy PnL (%)',
        line=dict(dash='dot', color='#fdd49e'),          # beige/light tan dotted
        yaxis='y2',
        hovertemplate='%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2f}%<extra></extra>'
    )

    fig_pnl.add_scatter(
        x=chart_long_pct[chart_long_pct['Metric'] == 'Buy & Hold PnL (%)']['week'],
        y=chart_long_pct[chart_long_pct['Metric'] == 'Buy & Hold PnL (%)']['Value'],
        mode='lines+markers',
        name='Buy & Hold PnL (%)',
        line=dict(dash='dot', color='#d62728'),          # red dotted
        yaxis='y2',
        hovertemplate='%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2f}%<extra></extra>'
    )

    # ── Layout & hover ─────────────────────────────────────────────────────────
    fig_pnl.update_traces(
        hovertemplate='%{x|%Y-%m-%d}<br>%{fullData.name}: $%{y:,.2f}<extra></extra>',
        selector=dict(name__in=['Strategy PnL ($)', 'Buy & Hold PnL ($)'])
    )

    fig_pnl.update_layout(
        xaxis_title="Week Ending",
        yaxis_title="PnL ($)",
        yaxis2=dict(
            title="PnL (%)",
            overlaying='y',
            side='right'
        ),
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

    st.plotly_chart(
        fig_pnl,
        use_container_width=True,
        config={
            'scrollZoom': False,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['pan2d'],
            'modeBarButtonsToRemove': ['zoom2d', 'lasso2d', 'select2d'],
            'displaylogo': False
        },
        key="pnl_comparison_chart"
    )

    # ── Value Comparison Over Time ────────────────────────────────────────────
    st.subheader("Value Comparison Over Time")

    chart_df['Strategy NAV'] = chart_df['net_liq_value']
    chart_df['Buy & Hold NAV'] = chart_df['friday_close'] * num_shares

    chart_long_value = chart_df.melt(id_vars=['week'],
                                     value_vars=['Strategy NAV', 'Buy & Hold NAV'],
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
                            'displaylogo': False},
                    key="value_comparison_chart")

    st.info("""
    - **Buy & Hold NAV** is shares × current price
    - **Strategy NAV** is cash (premiums + assignment proceeds) + position value
    - **Note**: Strategy NAV may be inflated if "Re-open after assignment" checkbox is checked (additional capital is injected to fulfill trades rather than Buy & Hold)
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
      - **American**: Binomial tree (Cox-Ross-Rubinstein) with 50 steps (allows early exercise check)
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