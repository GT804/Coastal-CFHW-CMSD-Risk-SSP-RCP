import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from scipy.special import logit, expit
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Volumes/T7/***/data.xls'
OUTPUT_PATH = '/Volumes/T7/***/result.xlsx'

xl = pd.ExcelFile(DATA_PATH)

hist_raw = xl.parse('贝叶斯结构时间序列历史数据')
hist_raw.columns = ['province', 'year', 'CMSD', 'pop', 'gdp', 'isa', 'isa_growth', 'cmsd_growth', 'note']
hist_df = hist_raw[hist_raw['province'].notna() & hist_raw['year'].notna()].copy()
hist_df = hist_df[['province', 'year', 'CMSD', 'pop', 'gdp', 'isa']].dropna()
hist_df['year'] = hist_df['year'].astype(int)

fut_raw = xl.parse('BSTS未来情景数据')
fut_raw.columns = [
    'province', 'year', 'CMSD',
    'ssp126_pop', 'ssp126_gdp', 'ssp126_isa',
    'ssp245_pop', 'ssp245_gdp', 'ssp245_isa',
    'ssp585_pop', 'ssp585_gdp', 'ssp585_isa',
    'isa_growth_126', 'isa_growth_245', 'isa_growth_585', 'note'
]
fut_df = fut_raw[fut_raw['province'].notna() & fut_raw['year'].notna()].copy()
fut_df = fut_df[['province', 'year',
                  'ssp126_pop', 'ssp126_gdp', 'ssp126_isa',
                  'ssp245_pop', 'ssp245_gdp', 'ssp245_isa',
                  'ssp585_pop', 'ssp585_gdp', 'ssp585_isa']].dropna(subset=['year'])
fut_df['year'] = fut_df['year'].astype(int)

provinces = hist_df['province'].unique()
print(f"沿海省份: {list(provinces)}")

def standardize(series, mu=None, sd=None):
    if mu is None:
        mu = series.mean()
    if sd is None:
        sd = series.std()
    return (series - mu) / (sd + 1e-10), mu, sd

def fit_bsts_province(prov_hist):
    cmsd = prov_hist['CMSD'].values.astype(float)
    cmsd_clipped = np.clip(cmsd, 1e-6, 1 - 1e-6)
    y = logit(cmsd_clipped)

    pop_s, pop_mu, pop_sd = standardize(prov_hist['pop'])
    gdp_s, gdp_mu, gdp_sd = standardize(prov_hist['gdp'])
    isa_s, isa_mu, isa_sd = standardize(prov_hist['isa'])

    pop = pop_s.values.astype(float)
    gdp = gdp_s.values.astype(float)
    isa = isa_s.values.astype(float)
    T = len(y)

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=y.mean(), sigma=1.0)
        b_pop = pm.Normal('b_pop', mu=0, sigma=0.5)
        b_gdp = pm.Normal('b_gdp', mu=0, sigma=0.5)
        b_isa = pm.Normal('b_isa', mu=0, sigma=0.5)

        rho = pm.Beta('rho', alpha=2, beta=2)
        sigma_ar = pm.HalfNormal('sigma_ar', sigma=0.3)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.3)

        trend = alpha + b_pop * pop + b_gdp * gdp + b_isa * isa

        ar_innov = pm.Normal('ar_innov', mu=0, sigma=sigma_ar, shape=T)
        ar_comp = pm.Deterministic('ar_comp', pm.math.cumsum(ar_innov))

        mu = trend + ar_comp

        obs = pm.Normal('obs', mu=mu, sigma=sigma_obs, observed=y)

        trace = pm.sample(
            draws=800, tune=1000, chains=2,
            progressbar=False, random_seed=42,
            target_accept=0.95,
            return_inferencedata=True,
            nuts_sampler_kwargs={'max_treedepth': 12}
        )

    stats = {
        'pop_mu': pop_mu, 'pop_sd': pop_sd,
        'gdp_mu': gdp_mu, 'gdp_sd': gdp_sd,
        'isa_mu': isa_mu, 'isa_sd': isa_sd,
    }

    post = trace.posterior
    params = {
        'alpha_samples': post['alpha'].values.flatten(),
        'b_pop_samples': post['b_pop'].values.flatten(),
        'b_gdp_samples': post['b_gdp'].values.flatten(),
        'b_isa_samples': post['b_isa'].values.flatten(),
        'rho_samples': post['rho'].values.flatten(),
        'sigma_ar_samples': post['sigma_ar'].values.flatten(),
        'sigma_obs_samples': post['sigma_obs'].values.flatten(),
        'last_ar_samples': post['ar_comp'].values[:, :, -1].flatten(),
        'stats': stats,
    }
    return params

def predict_future(params, fut_pop_raw, fut_gdp_raw, fut_isa_raw, n_steps):
    s = params['stats']
    fut_pop = (fut_pop_raw - s['pop_mu']) / (s['pop_sd'] + 1e-10)
    fut_gdp = (fut_gdp_raw - s['gdp_mu']) / (s['gdp_sd'] + 1e-10)
    fut_isa = (fut_isa_raw - s['isa_mu']) / (s['isa_sd'] + 1e-10)

    n_samples = len(params['alpha_samples'])
    predictions = np.zeros((n_samples, n_steps))

    for i in range(n_samples):
        alpha = params['alpha_samples'][i]
        b_pop = params['b_pop_samples'][i]
        b_gdp = params['b_gdp_samples'][i]
        b_isa = params['b_isa_samples'][i]
        sigma_ar = params['sigma_ar_samples'][i]
        ar = params['last_ar_samples'][i]

        for t in range(n_steps):
            ar = ar + np.random.normal(0, sigma_ar)
            trend = alpha + b_pop * fut_pop[t] + b_gdp * fut_gdp[t] + b_isa * fut_isa[t]
            predictions[i, t] = trend + ar

    pred_mean = predictions.mean(axis=0)
    pred_cmsd = expit(pred_mean)
    pred_cmsd = np.clip(pred_cmsd, 0.0, 1.0)
    return pred_cmsd

all_results = []

for prov in provinces:
    print(f"\n处理省份: {prov}")
    prov_hist = hist_df[hist_df['province'] == prov].sort_values('year').copy()
    prov_fut = fut_df[fut_df['province'] == prov].sort_values('year').copy()

    if len(prov_hist) < 5:
        print(f"  {prov} 历史数据不足，跳过")
        continue

    print(f"  拟合BSTS模型 (n={len(prov_hist)})...")
    params = fit_bsts_province(prov_hist)

    for ssp in ['ssp126', 'ssp245', 'ssp585']:
        pop_col = f'{ssp}_pop'
        gdp_col = f'{ssp}_gdp'
        isa_col = f'{ssp}_isa'

        prov_fut_ssp = prov_fut[['year', pop_col, gdp_col, isa_col]].dropna().copy()
        if len(prov_fut_ssp) == 0:
            continue

        fut_pop = prov_fut_ssp[pop_col].values.astype(float)
        fut_gdp = prov_fut_ssp[gdp_col].values.astype(float)
        fut_isa = prov_fut_ssp[isa_col].values.astype(float)
        fut_years = prov_fut_ssp['year'].values
        n_steps = len(fut_years)

        pred_cmsd = predict_future(params, fut_pop, fut_gdp, fut_isa, n_steps)

        for i, yr in enumerate(fut_years):
            all_results.append({
                'province': prov,
                'year': yr,
                'ssp': ssp.upper(),
                'CMSD': round(float(pred_cmsd[i]), 6)
            })

results_df = pd.DataFrame(all_results)
pivot_df = results_df.pivot_table(
    index=['province', 'year'],
    columns='ssp',
    values='CMSD'
).reset_index()

pivot_df.columns.name = None
pivot_df = pivot_df.rename(columns={
    'province': 'province',
    'year': 'year',
})

for col in ['SSP126', 'SSP245', 'SSP585']:
    if col in pivot_df.columns:
        pivot_df[col] = pivot_df[col].clip(0.0, 1.0)

pivot_df = pivot_df.sort_values(['province', 'year']).reset_index(drop=True)

pivot_df.to_excel(OUTPUT_PATH, index=False)
print(f"\n结果已保存至: {OUTPUT_PATH}")
print(f"结果预览:\n{pivot_df.head(20)}")
print(f"\n总行数: {len(pivot_df)}, 省份数: {pivot_df['province'].nunique()}")
