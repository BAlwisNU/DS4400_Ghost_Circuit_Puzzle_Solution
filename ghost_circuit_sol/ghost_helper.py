""" helpers for Ghost on the Circuit: forward stepwise regression, its
evaluation, a benchmark to compare it against, and the notebook's figures """
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy import stats

from ghost_circuit import GhostCircuit, OFFICIAL


def _residualize(x, base, y=None):
    """ removes from x (and y) the part explained by [1, base] """
    q, _ = np.linalg.qr(np.column_stack([np.ones(len(x)), base]))
    out = x - q @ (q.T @ x)
    return out if y is None else (out, y - q @ (q.T @ y))


def stepwise(x, y, alpha=.05, max_dev=30):
    """ names devices on the circuit by forward stepwise regression

    model: y = baseline + sum of watts_i * x_i + noise.  starting from no
    devices, repeatedly add the device which most reduces the squared error,
    among those whose fitted watts would be positive.  stop once the best
    addition isn't significant: its F-test p-value must beat alpha divided by
    the number of devices still in the running (a Bonferroni correction, as
    we're picking the best of many candidates each step).

    each step scores every candidate at once: with v = candidate and r =
    meter, both with the chosen devices' part removed, adding the candidate
    cuts squared error by (v . r)^2 / (v . v) and its fitted watts have the
    sign of v . r.  this gives the same answer as refitting one regression
    per candidate, about 10x faster.

    safeguards (neither ever triggered on the test circuits): refit and drop
    any chosen device whose watts are no longer positive, repeating until
    none are; stop after max_dev devices.

    Returns:
        device_idx (np.array): columns of x named as on the circuit
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n_hour, n_dev = x.shape
    chosen = []
    while len(chosen) < min(max_dev, n_hour - 3):
        v, r = _residualize(x, x[:, chosen], y)
        vv, vr = (v * v).sum(axis=0), v.T @ r
        gain = np.full(n_dev, -np.inf)
        ok = (vr > 0) & (vv > 1e-9)
        gain[ok] = vr[ok] ** 2 / vv[ok]
        gain[chosen] = -np.inf
        best = int(np.argmax(gain))
        if not np.isfinite(gain[best]):
            break
        rss, dof = r @ r, n_hour - len(chosen) - 2
        f = gain[best] / ((rss - gain[best]) / dof)
        if stats.f.sf(f, 1, dof) > alpha / (n_dev - len(chosen)):
            break
        chosen.append(best)
    while chosen:
        a = np.column_stack([np.ones(n_hour), x[:, chosen]])
        watts = np.linalg.lstsq(a, y, rcond=None)[0][1:]
        if (watts > 0).all():
            break
        chosen = [i for i, w in zip(chosen, watts) if w > 0]
    return np.array(sorted(chosen), dtype=int)


def correlated(x, y):
    """ starter baseline: every device whose on / off pattern correlates
    with the meter (r > .2); a device never on gets r = nan, never named """
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.array([np.corrcoef(x[:, i], y)[0, 1] for i in range(x.shape[1])])
    return np.where(r > .2)[0]


N_OFF = OFFICIAL['n_dev'] - OFFICIAL['n_on']  # off-circuit devices per circuit


def circuits(n_circuit=1000, seed=5, **setup):
    """ n_circuit fresh circuits.  seed 5 was never looked at while
    developing (seeds 1, 3, 7 were), so it's the final test set """
    setup = {**OFFICIAL, **setup}
    return [GhostCircuit.sample(**setup, seed=seed * 1_000_003 + idx)
            for idx in range(n_circuit)]


def validate(estimate, n_circuit=1000, seed=5, **setup):
    """ runs estimate on fresh circuits.  hidden truth is used only to grade

    Returns:
        result (dict):
            tpr, fpr (float): fraction of on- / off-circuit devices named
            fpr_lo, fpr_hi (float): 95% range for fpr (exact Poisson, as
                wrong names are rare)
            n_wrong (int): off-circuit devices named, over all circuits
            energy (float): mean share, per circuit, of the energy the
                on-circuit devices used (watts x hours on) that comes from
                devices named
            watts (np.array): watts of every on-circuit device
            found (np.array): bool, was that device named
    """
    n_wrong = n_off = 0
    watts, found, energy = [], [], []
    for c in circuits(n_circuit, seed, **setup):
        guess = set(np.asarray(estimate(c.x, c.y)).tolist())
        truth = c._on_circuit.tolist()
        watts += list(c._watts[truth])
        found += [i in guess for i in truth]
        used = c._watts[truth] * c.x[:, truth].sum(axis=0)
        energy.append(used[found[-len(truth):]].sum() / used.sum())
        n_wrong += len(guess - set(truth))
        n_off += c.n_dev - len(truth)
    lo = stats.chi2.ppf(.025, 2 * n_wrong) / 2 if n_wrong else 0
    hi = stats.chi2.ppf(.975, 2 * n_wrong + 2) / 2
    found = np.array(found)
    return dict(tpr=found.mean(), fpr=n_wrong / n_off, fpr_lo=lo / n_off,
                fpr_hi=hi / n_off, n_wrong=n_wrong, energy=np.mean(energy),
                watts=np.array(watts), found=found)


def tradeoff(estimate, alphas, n_circuit=1000, seed=5):
    """ (fpr, tpr) of estimate(x, y, alpha=a) for each a in alphas """
    curve = []
    for a in alphas:
        r = validate(lambda x, y: estimate(x, y, alpha=a), n_circuit, seed)
        curve.append([r['fpr'], r['tpr']])
    return np.array(curve)


def _t_stat(x, y, base, cand):
    """ t statistic of each candidate column added to the model [1, base] """
    v, r = _residualize(x[:, cand].astype(float), x[:, base], y)
    vv, vr = (v * v).sum(axis=0), v.T @ r
    t = np.full(len(cand), -np.inf)  # a device never on can't be detected
    ok = vv > 1e-9
    dof = len(y) - len(base) - 2
    res_var = ((r @ r) - vr[ok] ** 2 / vv[ok]) / dof
    t[ok] = vr[ok] / vv[ok] * np.sqrt(vv[ok] / res_var)
    return t


def oracle_curve(fprs, n_circuit=1000, seed=5):
    """ benchmark: TPR at each FPR in fprs for a test that is TOLD the other
    on-circuit devices.  each on-circuit device is scored given the other 7
    true devices; each off-circuit device is scored given all 8.  no real
    method knows this, so it's a practical ceiling (not a strict bound) """
    t_on, t_off = [], []
    for c in circuits(n_circuit, seed):
        truth = c._on_circuit.tolist()
        off = [i for i in range(c.n_dev) if i not in truth]
        t_off += list(_t_stat(c.x, c.y, truth, off))
        t_on += [_t_stat(c.x, c.y, [j for j in truth if j != i], [i])[0]
                 for i in truth]
    t_on, t_off = np.array(t_on), np.array(t_off)
    return np.array([(t_on > np.quantile(t_off, 1 - f)).mean() for f in fprs])


def more_hours(estimate, hours=(72, 144, 288), n_circuit=200):
    """ TPR of estimate with more hours of logs (other settings official) """
    return {h: validate(estimate, n_circuit, seed=6, n_hour=h)['tpr']
            for h in hours}


def plot_found_by_watts(res, noise, n_bin=8):
    """ fraction of on-circuit devices found vs device watts (log bins) """
    edges = np.geomspace(20, 1500, n_bin + 1)
    bin_idx = np.clip(np.digitize(res['watts'], edges) - 1, 0, n_bin - 1)
    rate = [100 * res['found'][bin_idx == b].mean() for b in range(n_bin)]
    plt.figure()
    plt.semilogx(np.sqrt(edges[:-1] * edges[1:]), rate, marker='o')
    plt.axvline(noise, color='gray', linestyle='--',
                label=f'meter noise ({noise} W standard deviation)')
    ax = plt.gca()
    ax.set_xticks([20, 50, 100, 200, 500, 1500])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set(xlabel='device power (watts)', ylabel='% of on-circuit devices found',
           ylim=(0, 105))
    ax.legend(loc='lower right')


def plot_tradeoff(curve, bench, starter, alphas, n_off=N_OFF):
    """ % found vs wrong names per circuit: stepwise, benchmark, starter """
    plt.figure()
    per = curve[:, 0] * n_off
    plt.semilogx(per, 100 * curve[:, 1], marker='o',
                 label='stepwise (each dot: one strictness setting)')
    plt.semilogx(per, 100 * bench, marker='s', linestyle='--', color='gray',
                 label='benchmark told the other 7 devices')
    for a, xx, yy in zip(alphas, per, 100 * curve[:, 1]):
        if a == .05:
            plt.annotate('submitted (alpha = 0.05)', (xx, yy), xytext=(8, -14),
                         textcoords='offset points')
    plt.scatter(starter['fpr'] * n_off, 100 * starter['tpr'], color='C3',
                label='starter: correlation > 0.2')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set(xlabel='devices wrongly named per circuit (log scale)',
           ylabel='% of on-circuit devices found')
    ax.legend(loc='center right')
