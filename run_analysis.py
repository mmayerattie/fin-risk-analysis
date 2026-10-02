"""Run all analysis cells and print full outputs with figures saved to figures/."""
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')   # headless — no display needed

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew, kurtosis as excess_kurt
from arch import arch_model
import cvxpy as cp
import os, time

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#F8F9FA',
    'axes.grid': True, 'grid.alpha': 0.35,
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold'
})

TICKERS   = ['ABNB', 'CCL', 'TSLA', 'KER.PA', 'PUM.DE']
NAMES     = {'ABNB': 'Airbnb', 'CCL': 'Carnival', 'TSLA': 'Tesla',
             'KER.PA': 'Kering', 'PUM.DE': 'Puma'}
COLORS    = {'ABNB': '#E63946', 'CCL': '#2196F3', 'TSLA': '#FF6F00',
             'KER.PA': '#9C27B0', 'PUM.DE': '#4CAF50'}
START, END   = '2023-01-01', '2025-12-31'
RF_MONTHLY   = 0.004
HOLDING      = 21
SAVE_DPI     = 150
os.makedirs('figures', exist_ok=True)

# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────────
def _download_one(ticker, start, end, retries=4, delay=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=False, progress=False)
            if len(df) > 10:
                px = df['Close']
                if isinstance(px, pd.DataFrame):
                    px = px.iloc[:, 0]
                px.name = ticker
                return px
        except Exception as e:
            print(f'  {ticker}: error "{e}", retrying…')
        time.sleep(delay)
    raise RuntimeError(f'Could not download {ticker}')

print('=' * 65)
print('DOWNLOADING PRICE DATA')
print('=' * 65)
px_dict = {}
for t in TICKERS + ['^GSPC']:
    print(f'  {t}…', end=' ', flush=True)
    px_dict[t] = _download_one(t, START, END)
    print(f'✓  {len(px_dict[t])} days')

prices  = pd.DataFrame({t: px_dict[t] for t in TICKERS})
mkt_px  = px_dict['^GSPC']
log_ret = np.log(prices / prices.shift(1))
mkt_ret = np.log(mkt_px / mkt_px.shift(1))
all_data = pd.concat([log_ret, mkt_ret.rename('^GSPC')], axis=1).dropna()
log_ret  = all_data[TICKERS]
mkt_ret  = all_data['^GSPC']
print(f'\nPeriod : {log_ret.index[0].date()} → {log_ret.index[-1].date()}')
print(f'Days   : {len(log_ret)} common trading-day observations')

# ── EDA: NORMALIZED PRICES ────────────────────────────────────────────────────
norm_px = prices.divide(prices.bfill().iloc[0]) * 100
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
for i, t in enumerate(TICKERS):
    ax = axes[i]
    ax.plot(norm_px.index, norm_px[t], color=COLORS[t], linewidth=1.5)
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_title(NAMES[t])
    ax.tick_params(axis='x', rotation=40, labelsize=8)
    if i == 0:
        ax.set_ylabel('Index (Jan 2023 = 100)')
fig.suptitle('Normalized Price Series — Jan 2023 to Dec 2025', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('figures/01_normalized_prices.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.close()

# Print final normalized values (end-of-period returns)
print('\n' + '=' * 65)
print('EDA — CUMULATIVE RETURNS (Jan 2023 = 100)')
print('=' * 65)
last_vals = norm_px.ffill().iloc[-1]
for t in TICKERS:
    chg = last_vals[t] - 100
    print(f'  {NAMES[t]:12s}: {last_vals[t]:6.1f}  ({chg:+.1f}%)')

# ── EDA: LOG-RETURNS ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
for i, t in enumerate(TICKERS):
    ax = axes[i]
    ax.plot(log_ret.index, log_ret[t] * 100, color=COLORS[t], linewidth=0.75, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel(f'{NAMES[t]}\n(%)', fontsize=9)
axes[0].set_title('Daily Log-Returns (%) — All Tickers, 2023–2025')
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig('figures/02_log_returns.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.close()

# ── EDA: SUMMARY STATISTICS ───────────────────────────────────────────────────
print('\n' + '=' * 65)
print('EDA — SUMMARY STATISTICS OF DAILY LOG-RETURNS')
print('=' * 65)
stats_rows = []
for t in TICKERS:
    r = log_ret[t].dropna().values
    stats_rows.append({
        'Name'          : NAMES[t],
        'Mean (daily%)' : round(r.mean() * 100, 4),
        'Std (daily%)'  : round(r.std()  * 100, 4),
        'Ann. Vol (%)'  : round(r.std()  * np.sqrt(252) * 100, 2),
        'Skewness'      : round(skew(r), 3),
        'Ex. Kurtosis'  : round(excess_kurt(r), 3),
        'Min (%)'       : round(r.min() * 100, 3),
        'Max (%)'       : round(r.max() * 100, 3),
    })
stats_df = pd.DataFrame(stats_rows).set_index('Name')
print(stats_df.to_string())
stats_df.to_csv('figures/summary_statistics.csv')

# ── EDA: ROLLING 30-DAY VOL ───────────────────────────────────────────────────
rolling_vol = log_ret.rolling(30).std() * np.sqrt(252) * 100
fig, ax = plt.subplots(figsize=(14, 5))
for t in TICKERS:
    ax.plot(rolling_vol.index, rolling_vol[t], label=NAMES[t], color=COLORS[t], linewidth=1.6)
ax.set_title('30-Day Rolling Annualised Volatility (%) — 2023–2025')
ax.set_ylabel('Annualised Volatility (%)')
ax.legend(ncol=5, loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('figures/03_rolling_volatility.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.close()

# ── DISTRIBUTION: NORMAL vs STUDENT-t ────────────────────────────────────────
print('\n' + '=' * 65)
print('DISTRIBUTION FITTING — NORMAL vs STUDENT-t (KS TESTS)')
print('=' * 65)
dist_results = {}
for t in TICKERS:
    x = log_ret[t].dropna().values
    mu_n, sig_n       = stats.norm.fit(x)
    df_t, loc_t, sc_t = stats.t.fit(x)
    ks_n = stats.kstest(x, 'norm', args=(mu_n, sig_n))
    ks_t = stats.kstest(x, 't',    args=(df_t, loc_t, sc_t))
    dist_results[t] = {'df': df_t, 'loc': loc_t, 'scale': sc_t,
                       'ks_norm_p': ks_n.pvalue, 'ks_t_p': ks_t.pvalue}
    norm_verdict = 'REJECT' if ks_n.pvalue < 0.05 else 'pass'
    t_verdict    = 'REJECT' if ks_t.pvalue < 0.05 else 'pass'
    print(f'  {NAMES[t]:12s} | t-df = {df_t:5.2f} | '
          f'KS-Normal p = {ks_n.pvalue:.4f} ({norm_verdict}) | '
          f'KS-t p = {ks_t.pvalue:.4f} ({t_verdict})')

fig, ax = plt.subplots(figsize=(8, 4))
dfs = [dist_results[t]['df'] for t in TICKERS]
bars = ax.bar([NAMES[t] for t in TICKERS], dfs,
              color=[COLORS[t] for t in TICKERS], edgecolor='black', linewidth=0.5)
ax.axhline(5, color='gray', linestyle='--', linewidth=1, label='df=5 (moderate tails)')
for b, v in zip(bars, dfs):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
            f'{v:.1f}', ha='center', va='bottom', fontsize=9)
ax.set_title('Student-t Degrees of Freedom (lower = heavier tails)')
ax.set_ylabel('Estimated df')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('figures/04_t_degrees_of_freedom.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.close()

# ── GARCH MODEL SELECTION ─────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('GARCH MODEL SELECTION — GARCH(1,1)-t vs GJR-GARCH(1,1,1)-t')
print('=' * 65)
garch_res = {}
for t in TICKERS:
    ret_pct = (100 * log_ret[t]).dropna()
    fg = arch_model(ret_pct, vol='GARCH', p=1, q=1,
                    mean='Constant', dist='t').fit(disp='off')
    fj = arch_model(ret_pct, vol='GARCH', p=1, o=1, q=1,
                    mean='Constant', dist='t').fit(disp='off')
    best      = fg if fg.aic <= fj.aic else fj
    best_name = 'GARCH(1,1)-t' if fg.aic <= fj.aic else 'GJR-GARCH-t'
    p = best.params
    alpha   = p.get('alpha[1]', 0)
    beta    = p.get('beta[1]',  0)
    gamma   = p.get('gamma[1]', 0)
    persist = alpha + beta + 0.5 * gamma
    garch_res[t] = {'fit': best, 'model': best_name,
                    'aic_g': fg.aic, 'aic_gjr': fj.aic,
                    'persist': persist, 'params': p,
                    'alpha': alpha, 'beta': beta, 'gamma': gamma}
    print(f'  {NAMES[t]:12s} | Winner: {best_name:15s} | '
          f'AIC(GARCH)={fg.aic:.1f}  AIC(GJR)={fj.aic:.1f} | '
          f'α={alpha:.4f}  β={beta:.4f}  γ={gamma:.4f}  Persist={persist:.4f}')

# ── CONDITIONAL VOLATILITY PLOT ───────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
for i, t in enumerate(TICKERS):
    ax = axes[i]
    ret_pct = (100 * log_ret[t]).dropna()
    cv = garch_res[t]['fit'].conditional_volatility
    ax.plot(ret_pct.index, ret_pct.values,
            color='gray', linewidth=0.65, alpha=0.55, label='Return (%)')
    ax.fill_between(ret_pct.index, -2*cv, 2*cv,
                    color=COLORS[t], alpha=0.28, label=f'±2σ ({garch_res[t]["model"]})')
    ax.set_ylabel(NAMES[t], fontsize=9)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
axes[0].set_title('Log-Returns with GARCH Conditional Volatility Band (±2σ), 2023–2025')
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig('figures/05_conditional_volatility.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.close()

# Print avg conditional vol per ticker
print()
for t in TICKERS:
    cv = garch_res[t]['fit'].conditional_volatility
    print(f'  {NAMES[t]:12s} | Avg cond. vol = {cv.mean():.3f}%/day  '
          f'| Peak = {cv.max():.3f}%/day  | '
          f'Ann. avg = {cv.mean()*np.sqrt(252):.1f}%')

# ── VaR / ES FUNCTIONS ────────────────────────────────────────────────────────
def compute_var(losses, levels):
    x = pd.Series(losses).dropna().astype(float)
    df_t, loc_t, sc_t = stats.t.fit(x)
    mu, sig = x.mean(), x.std(ddof=1)
    rows = []
    for a in levels:
        rows.append({
            'Level'     : a,
            'Historical': np.percentile(x, 100*a),
            'Normal'    : mu + sig * norm.ppf(a),
            'T-dist'    : stats.t.ppf(a, df_t, loc=loc_t, scale=sc_t)
        })
    return pd.DataFrame(rows).set_index('Level')

def compute_es(losses, levels):
    x = pd.Series(losses).dropna().astype(float)
    df_t, loc_t, sc_t = stats.t.fit(x)
    mu, sig = x.mean(), x.std(ddof=1)
    rows = []
    for a in levels:
        var_h = np.percentile(x, 100*a)
        es_h  = x[x >= var_h].mean()
        z     = norm.ppf(a)
        es_n  = mu + sig * norm.pdf(z) / (1 - a)
        q     = stats.t.ppf(a, df_t)
        es_t  = (loc_t + sc_t * (df_t + q**2) / ((df_t - 1) * (1 - a))
                 * stats.t.pdf(q, df_t)) if df_t > 1 else np.nan
        rows.append({'Level': a, 'Historical': es_h, 'Normal': es_n, 'T-dist': es_t})
    return pd.DataFrame(rows).set_index('Level')

LEVELS = [0.95, 0.99]
var_all, es_all = {}, {}
for t in TICKERS:
    loss = -(100 * log_ret[t]).dropna()
    var_all[t] = compute_var(loss, LEVELS)
    es_all[t]  = compute_es(loss, LEVELS)

print('\n' + '=' * 65)
print('VaR (daily loss %) — COMPARISON ACROSS ALL METHODS')
print('=' * 65)
for a in LEVELS:
    print(f'\n  VaR at {a:.0%}:')
    print(f'  {"Name":12s} {"Historical":>12} {"Normal":>10} {"T-dist":>10}')
    for t in TICKERS:
        print(f'  {NAMES[t]:12s} '
              f'{var_all[t].loc[a,"Historical"]:>12.3f} '
              f'{var_all[t].loc[a,"Normal"]:>10.3f} '
              f'{var_all[t].loc[a,"T-dist"]:>10.3f}')

print('\n' + '=' * 65)
print('ES (daily loss %) — COMPARISON ACROSS ALL METHODS')
print('=' * 65)
for a in LEVELS:
    print(f'\n  ES at {a:.0%}:')
    print(f'  {"Name":12s} {"Historical":>12} {"Normal":>10} {"T-dist":>10}')
    for t in TICKERS:
        print(f'  {NAMES[t]:12s} '
              f'{es_all[t].loc[a,"Historical"]:>12.3f} '
              f'{es_all[t].loc[a,"Normal"]:>10.3f} '
              f'{es_all[t].loc[a,"T-dist"]:>10.3f}')

# VaR/ES bar chart
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
labels = [NAMES[t] for t in TICKERS]
colors = [COLORS[t] for t in TICKERS]
var99  = [var_all[t].loc[0.99, 'T-dist'] for t in TICKERS]
es99   = [es_all[t].loc[0.99,  'T-dist'] for t in TICKERS]
for ax, vals, title, ylabel in [
        (axes[0], var99, 'VaR 99% — T-distribution (daily loss %)', 'Loss (%)'),
        (axes[1], es99,  'ES 99% — T-distribution (daily loss %)',  'Loss (%)')]:
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig('figures/06_var_es_comparison.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.close()

# ── DYNAMIC VaR — TSLA ────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('DYNAMIC VaR — TSLA (AR(1)-GARCH(1,1)-t)')
print('=' * 65)
t_dyn = 'TSLA'
ret_pct_dyn = (100 * log_ret[t_dyn]).dropna()
mod_dyn = arch_model(ret_pct_dyn, vol='Garch', p=1, q=1,
                     mean='AR', lags=1, dist='t')
res_dyn = mod_dyn.fit(disp='off')
fcst    = res_dyn.forecast(start=0)
mu_r    = fcst.mean.iloc[:, -1]
sig_r   = np.sqrt(fcst.variance.iloc[:, -1])
nu      = res_dyn.params['nu']
lam     = sig_r * np.sqrt((nu - 2) / nu)
idx     = mu_r.index
q95     = stats.t.ppf(0.05, df=nu, loc=mu_r.values, scale=lam.values)
q99     = stats.t.ppf(0.01, df=nu, loc=mu_r.values, scale=lam.values)
var95_dyn = pd.Series(-q95, index=idx)
var99_dyn = pd.Series(-q99, index=idx)
loss_dyn  = (-ret_pct_dyn).reindex(idx)

print(f'  Static VaR 99% (T-dist) : {var_all["TSLA"].loc[0.99, "T-dist"]:.3f}%')
print(f'  Dynamic VaR 99% mean    : {var99_dyn.mean():.3f}%')
print(f'  Dynamic VaR 99% max     : {var99_dyn.max():.3f}%  (peak risk day)')
print(f'  Dynamic VaR 99% min     : {var99_dyn.min():.3f}%  (calmest day)')
breaches_99 = int((loss_dyn > var99_dyn).sum())
n_days      = len(loss_dyn)
print(f'  In-sample breaches 99%  : {breaches_99} / {n_days} days  '
      f'({breaches_99/n_days*100:.2f}%, expected 1.00%)')

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(loss_dyn.index, loss_dyn, color='gray', alpha=0.5, linewidth=0.8, label='Loss (%)')
ax.plot(var99_dyn.index, var99_dyn, color='#B71C1C', linestyle='--',
        linewidth=1.5, label='Dynamic VaR 99%')
ax.plot(var95_dyn.index, var95_dyn, color='#E57373', linestyle='--',
        linewidth=1.2, label='Dynamic VaR 95%')
ax.axhline(0, color='black', linewidth=0.4)
ax.set_title(f'Dynamic VaR — Tesla (AR(1)-GARCH(1,1)-t, 2023–2025)')
ax.set_xlabel('Date'); ax.set_ylabel('Daily Loss (%)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('figures/07_dynamic_var_tsla.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.close()

# ── PORTFOLIO: CORRELATION ────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('PORTFOLIO — CORRELATION MATRIX')
print('=' * 65)
corr = log_ret.corr()
corr_named = corr.rename(index=NAMES, columns=NAMES)
print(corr_named.round(3).to_string())
# Highlight extremes
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
vals = corr.values[mask]
pairs = [(TICKERS[i], TICKERS[j])
         for i in range(len(TICKERS)) for j in range(i+1, len(TICKERS))]
max_idx = np.argmax(vals)
min_idx = np.argmin(vals)
print(f'\n  Highest correlation: {NAMES[pairs[max_idx][0]]} — {NAMES[pairs[max_idx][1]]}: {vals[max_idx]:.3f}')
print(f'  Lowest  correlation: {NAMES[pairs[min_idx][0]]} — {NAMES[pairs[min_idx][1]]}: {vals[min_idx]:.3f}')
print(f'  Average off-diagonal: {vals.mean():.3f}')

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_named, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, annot_kws={'size': 11},
            cbar_kws={'shrink': 0.8})
ax.set_title('Return Correlation Matrix — 2023–2025')
plt.tight_layout()
plt.savefig('figures/09_correlation_heatmap.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.close()

# ── PORTFOLIO: EFFICIENT FRONTIER ────────────────────────────────────────────
def compute_frontier(rf, mean_ret, cov_mat, n_pts=80):
    targets = np.linspace(mean_ret.min(), mean_ret.max(), n_pts)
    risks, rets, weights, sharpes = [], [], [], []
    mv_r, mv_ret, mv_w = np.inf, None, None
    Sigma     = cov_mat.values.astype(float)
    Sigma_reg = Sigma + 1e-8 * np.eye(len(Sigma))
    mu        = mean_ret.values.astype(float)
    n         = len(mu)
    for tgt in targets:
        w    = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma_reg)),
                          [cp.sum(w) == 1, w >= 0, mu @ w == tgt])
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-5)
        if prob.status not in ('optimal', 'optimal_inaccurate') or w.value is None:
            continue
        wv = np.clip(w.value, 0, None); wv /= wv.sum()
        pstd = float(np.sqrt(max(float(wv @ Sigma @ wv), 0)))
        pret = float(mu @ wv)
        risks.append(pstd); rets.append(pret); weights.append(wv)
        sharpes.append((pret - rf) / pstd if pstd > 0 else -np.inf)
        if pstd < mv_r:
            mv_r, mv_ret, mv_w = pstd, pret, wv
    idx_tang = int(np.argmax(sharpes))
    return {'risks': np.array(risks), 'returns': np.array(rets),
            'portfolios': weights, 'sharpes': np.array(sharpes),
            'min_var': (mv_r, mv_ret, mv_w),
            'tangency': (risks[idx_tang], rets[idx_tang],
                         weights[idx_tang], sharpes[idx_tang])}

mean_21d = log_ret.mean() * HOLDING
cov_21d  = log_ret.cov()  * HOLDING
std_21d  = log_ret.std()  * np.sqrt(HOLDING)

print('\n' + '=' * 65)
print('PORTFOLIO OPTIMISATION (21-day horizon, long-only)')
print('=' * 65)
print('Running efficient frontier…')
frontier = compute_frontier(RF_MONTHLY, mean_21d, cov_21d)
mv_r, mv_ret, mv_w         = frontier['min_var']
tg_r, tg_ret, tg_w, tg_sr = frontier['tangency']

print('\n  MINIMUM VARIANCE PORTFOLIO')
for t, w in zip(TICKERS, mv_w):
    print(f'    {NAMES[t]:12s}: {w*100:5.1f}%')
print(f'    Expected Return (21d): {mv_ret*100:.3f}%')
print(f'    Std Deviation  (21d): {mv_r*100:.3f}%')
print(f'    Ann. Vol equiv      : {mv_r/np.sqrt(HOLDING)*np.sqrt(252)*100:.1f}%')

print(f'\n  TANGENCY PORTFOLIO  (Sharpe = {tg_sr:.3f})')
for t, w in zip(TICKERS, tg_w):
    print(f'    {NAMES[t]:12s}: {w*100:5.1f}%')
print(f'    Expected Return (21d): {tg_ret*100:.3f}%')
print(f'    Std Deviation  (21d): {tg_r*100:.3f}%')
print(f'    Ann. Vol equiv      : {tg_r/np.sqrt(HOLDING)*np.sqrt(252)*100:.1f}%')

print('\n  INDIVIDUAL ASSET RISK (21-day std dev)')
for t in TICKERS:
    print(f'    {NAMES[t]:12s}: {std_21d[t]*100:.3f}%  '
          f'(Ann. {log_ret[t].std()*np.sqrt(252)*100:.1f}%)')

min_ind_risk = std_21d.min() * 100
print(f'\n  Risk reduction (MVP vs least risky stock): '
      f'{min_ind_risk:.3f}% → {mv_r*100:.3f}%  '
      f'({(1 - mv_r*100/min_ind_risk)*100:.1f}% reduction)')

# Frontier plot
risks_f = frontier['risks']
rets_f  = frontier['returns']
mv_idx  = int(np.argmin(risks_f))
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
ax.plot(risks_f[:mv_idx+1], rets_f[:mv_idx+1],
        color='gold', linewidth=2, linestyle='--', label='Inefficient')
ax.plot(risks_f[mv_idx:], rets_f[mv_idx:],
        color='royalblue', linewidth=2.5, label='Efficient frontier')
cal_x = np.linspace(0, tg_r * 1.5, 200)
cal_y = RF_MONTHLY + (tg_ret - RF_MONTHLY) / tg_r * cal_x
ax.plot(cal_x, cal_y, 'k--', linewidth=1.5, label='CAL')
for t in TICKERS:
    ax.scatter(std_21d[t], mean_21d[t], color=COLORS[t], s=90, zorder=5,
               edgecolors='black', linewidth=0.5)
    ax.annotate(NAMES[t], (std_21d[t], mean_21d[t]),
                textcoords='offset points', xytext=(5, 2), fontsize=8)
ax.scatter(mv_r, mv_ret, color='red',  s=160, marker='D', zorder=6,
           edgecolors='black', linewidth=0.5, label='Min Variance')
ax.scatter(tg_r, tg_ret, color='gold', s=220, marker='*', zorder=6,
           edgecolors='black', linewidth=0.5, label='Tangency')
ax.scatter(0, RF_MONTHLY, color='black', s=90, marker='x', zorder=6,
           label=f'Risk-free')
ax.set_xlabel('Risk — Std Dev (21-day)')
ax.set_ylabel('Expected Return (21-day)')
ax.set_title('Efficient Frontier & Capital Allocation Line\n(5-asset long-only, 21-day)')
ax.legend(fontsize=8, loc='upper left')

ax2 = axes[1]
x_pos = np.array([0, 1])
width = 0.14
for i, t in enumerate(TICKERS):
    wts = [mv_w[i]*100, tg_w[i]*100]
    ax2.bar(x_pos + i*width - 2*width, wts, width,
            label=NAMES[t], color=COLORS[t], edgecolor='black', linewidth=0.4)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Min Variance\nPortfolio', 'Tangency\nPortfolio'])
ax2.set_ylabel('Weight (%)')
ax2.set_title('Portfolio Weights: MVP vs Tangency')
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig('figures/10_efficient_frontier.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.close()

# Risk comparison bar
ind_risks  = [std_21d[t]*100 for t in TICKERS]
port_risks = [mv_r*100, tg_r*100]
labels_all = [NAMES[t] for t in TICKERS] + ['Min Variance\nPortfolio', 'Tangency\nPortfolio']
values_all = ind_risks + port_risks
colors_all = [COLORS[t] for t in TICKERS] + ['#C62828', '#FFA000']
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(labels_all, values_all, color=colors_all, edgecolor='black', linewidth=0.5)
for b, v in zip(bars, values_all):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03,
            f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Standard Deviation (21-day, %)')
ax.set_title('Diversification: Individual Asset Risk vs Portfolio Risk')
ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig('figures/11_risk_comparison.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.close()

# ── MASTER COMPARISON TABLE ───────────────────────────────────────────────────
print('\n' + '=' * 65)
print('MASTER COMPARISON TABLE')
print('=' * 65)
rows = []
for t in TICKERS:
    p = garch_res[t]['params']
    a = p.get('alpha[1]', 0); b = p.get('beta[1]', 0); g = p.get('gamma[1]', 0)
    rows.append({
        'Ticker'           : t,
        'Name'             : NAMES[t],
        'Best Model'       : garch_res[t]['model'],
        'Ann. Vol (%)'     : round(log_ret[t].std() * np.sqrt(252) * 100, 2),
        'Persistence'      : round(a + b + 0.5*g, 4),
        'VaR 99% T (%)'    : round(var_all[t].loc[0.99, 'T-dist'], 3),
        'ES 99% T (%)'     : round(es_all[t].loc[0.99,  'T-dist'], 3),
        't-df'             : round(dist_results[t]['df'], 2),
    })
master = pd.DataFrame(rows).set_index('Ticker')
pd.set_option('display.max_columns', None); pd.set_option('display.width', 200)
print(master.to_string())
master.to_csv('figures/master_comparison.csv')

print('\n  RANKING BY VaR 99%:')
ranked = master.sort_values('VaR 99% T (%)', ascending=False)
for rank, (idx, row) in enumerate(ranked.iterrows(), 1):
    print(f'  #{rank}  {row["Name"]:12s}  VaR={row["VaR 99% T (%)"]:.3f}%  '
          f'ES={row["ES 99% T (%)"]:.3f}%  Vol={row["Ann. Vol (%)"]:.1f}%  '
          f'Persist={row["Persistence"]:.4f}  df={row["t-df"]:.1f}')

# ── Q1 2026 OUT-OF-SAMPLE ─────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('Q1 2026 OUT-OF-SAMPLE EVALUATION')
print('=' * 65)
Q1_START, Q1_END = '2026-01-01', '2026-03-31'
px_2026 = {}
for t in TICKERS:
    print(f'  {t}…', end=' ', flush=True)
    try:
        px_2026[t] = _download_one(t, Q1_START, Q1_END)
        print(f'✓  {len(px_2026[t])} days')
    except Exception as e:
        print(f'✗  {e}')

prices_2026  = pd.DataFrame(px_2026).dropna(how='all')
log_ret_2026 = np.log(prices_2026 / prices_2026.shift(1)).dropna()
if len(log_ret_2026):
    print(f'Period: {log_ret_2026.index[0].date()} → {log_ret_2026.index[-1].date()}  '
          f'({len(log_ret_2026)} trading days)')

    print(f'\n  {"Name":12s} {"VaR99%(T)":>10} {"Q1days":>7} '
          f'{"Breaches":>9} {"Breach%":>8} {"Expected":>9} {"Status":>7}')
    print('  ' + '-' * 62)
    for t in TICKERS:
        if t not in log_ret_2026.columns:
            continue
        var99   = var_all[t].loc[0.99, 'T-dist']
        losses  = -(100 * log_ret_2026[t]).dropna()
        n_days  = len(losses)
        n_b     = int((losses > var99).sum())
        bpct    = n_b / n_days * 100 if n_days else 0
        exp     = n_days * 0.01
        flag    = '⚠ OVER' if n_b > exp * 2 else '✓ OK'
        print(f'  {NAMES[t]:12s} {var99:>10.3f}% {n_days:>7} '
              f'{n_b:>9} {bpct:>7.1f}% {exp:>8.1f}   {flag}')

    print('\n  VOLATILITY: 2023-2025 vs Q1 2026')
    for t in TICKERS:
        if t not in log_ret_2026.columns:
            continue
        hv = log_ret[t].std() * np.sqrt(252) * 100
        qv = log_ret_2026[t].std() * np.sqrt(252) * 100
        chg = qv - hv
        direction = '↑ HIGHER' if chg > 3 else ('↓ LOWER' if chg < -3 else '≈ similar')
        print(f'  {NAMES[t]:12s}  2023-25={hv:.1f}%  Q1-2026={qv:.1f}%  '
              f'Δ={chg:+.1f}pp  {direction}')

    # Extended price + vol comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    all_px = pd.concat([prices, prices_2026]).sort_index()
    all_px = all_px[~all_px.index.duplicated()]
    norm_all = all_px.divide(all_px.bfill().iloc[0]) * 100
    for t in TICKERS:
        if t not in norm_all.columns:
            continue
        mh = norm_all.index <= '2025-12-31'
        m2 = norm_all.index >= '2026-01-01'
        ax.plot(norm_all.index[mh], norm_all[t][mh],
                color=COLORS[t], linewidth=1.4, label=NAMES[t])
        ax.plot(norm_all.index[m2], norm_all[t][m2],
                color=COLORS[t], linewidth=2.2, linestyle='--')
    ax.axvline(pd.Timestamp('2026-01-01'), color='black',
               linestyle=':', linewidth=1.5, label='Q1 2026 start')
    ax.set_title('Prices: 2023–2025 (solid) vs Q1 2026 (dashed)')
    ax.set_ylabel('Index (Jan 2023 = 100)')
    ax.legend(fontsize=8, ncol=2)

    ax2 = axes[1]
    hist_vols = [log_ret[t].std()*np.sqrt(252)*100 for t in TICKERS]
    q1_vols   = [log_ret_2026[t].std()*np.sqrt(252)*100
                 if t in log_ret_2026.columns else 0 for t in TICKERS]
    x = np.arange(len(TICKERS)); w = 0.35
    ax2.bar(x - w/2, hist_vols, w, label='2023–2025',
            color=[COLORS[t] for t in TICKERS], alpha=0.6, edgecolor='black', linewidth=0.5)
    ax2.bar(x + w/2, q1_vols,   w, label='Q1 2026',
            color=[COLORS[t] for t in TICKERS], alpha=1.0, edgecolor='black',
            linewidth=0.5, hatch='//')
    ax2.set_xticks(x)
    ax2.set_xticklabels([NAMES[t] for t in TICKERS], rotation=15)
    ax2.set_ylabel('Annualised Volatility (%)')
    ax2.set_title('Volatility: Historical vs Q1 2026')
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/12_q1_2026_evaluation.png', dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()

print('\n' + '=' * 65)
print('ALL FIGURES SAVED TO figures/')
print('=' * 65)
figs = sorted(os.listdir('figures'))
for f in figs:
    if f.endswith('.png') or f.endswith('.csv'):
        print(f'  {f}')
