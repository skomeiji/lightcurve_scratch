import os
import time

import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle


def structure_func(time_arr, flux, mag, nbins=100, make_plot=True):
    mask = np.isfinite(time_arr) & np.isfinite(flux) & np.isfinite(mag) & (flux > 0)

    time_arr = np.asarray(time_arr[mask], dtype=float)
    flux = np.asarray(flux[mask], dtype=float)
    mag = np.asarray(mag[mask], dtype=float)

    order = np.argsort(time_arr)
    time_arr = time_arr[order]
    mag = mag[order]

    dt_min = np.nanmedian(np.diff(time_arr))
    dt_max = time_arr.max() - time_arr.min()

    bin_edges = np.logspace(np.log10(dt_min), np.log10(dt_max), nbins + 1)
    tau = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    sf_sum = np.zeros(nbins)
    n_pairs = np.zeros(nbins, dtype=int)

    N = len(time_arr)

    for i in range(N - 1):
        dt = time_arr[i + 1:] - time_arr[i]
        dm2 = (mag[i + 1:] - mag[i]) ** 2

        inds = np.digitize(dt, bin_edges) - 1
        good = (inds >= 0) & (inds < nbins)

        np.add.at(sf_sum, inds[good], dm2[good])
        np.add.at(n_pairs, inds[good], 1)

    sf = np.full(nbins, np.nan)
    valid = n_pairs > 0
    sf[valid] = sf_sum[valid] / n_pairs[valid]

    good = (n_pairs >= 3) & np.isfinite(sf)

    x = tau[good]
    y = sf[good]

    if len(x) < 5:
        raise ValueError("Not enough populated SF bins to fit structure function.")

    def sf_model(tau_val, C0, C1, t0):
        return C0 + C1 * (1 - np.exp(-tau_val / t0))

    p0 = [
        np.nanmin(y),
        np.nanmax(y) - np.nanmin(y),
        1.0,
    ]

    bounds = (
        [1e-8, 1e-8, dt_min],
        [np.inf, np.inf, dt_max],
    )

    popt, pcov = curve_fit(
        sf_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )

    C0, C1, t0 = popt

    print("C0 =", C0)
    print("C1 =", C1)
    print("Characteristic timescale t0 =", t0, "days")

    if make_plot:
        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 500)
        y_fit = sf_model(x_fit, C0, C1, t0)

        plt.figure(figsize=(7, 5))
        plt.loglog(x, y, "ko-", label="Structure function")
        plt.loglog(x_fit, y_fit, "r-", label="Mas et al. fit")
        plt.axvline(t0, color="blue", linestyle="--", label=fr"$t_0={t0:.2f}$ d")
        plt.xlabel("Time lag τ [days]")
        plt.ylabel(r"SF($\tau$) [mag$^2$]")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return tau, sf, n_pairs, t0, popt


def qm_plots(
    time_arr,
    flux,
    ts_days,
    source_id,
    author,
    sector,
    source_list="",
    target_name="",
):
    time_arr = np.asarray(time_arr, dtype=float)
    flux = np.ma.filled(flux, np.nan)
    flux = np.asarray(flux, dtype=float)

    mask = np.isfinite(time_arr) & np.isfinite(flux) & (flux > 0)

    time_arr = time_arr[mask]
    flux = flux[mask]

    mag = -2.5 * np.log10(flux)

    mean_mag = np.nanmean(mag)
    std_mag = np.nanstd(mag)

    p10 = np.nanpercentile(mag, 10)
    p90 = np.nanpercentile(mag, 90)
    decile_mask = (mag <= p10) | (mag >= p90)

    fig_dir = f"figures/{source_list}"
    os.makedirs(fig_dir, exist_ok=True)

    safe_target = str(target_name).replace(" ", "_").replace("/", "_")
    safe_author = str(author).replace(" ", "_").replace("/", "_")

    # ---------- M-style time-domain plot ----------
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    ax.scatter(time_arr, mag, s=8, c="0.15", alpha=0.55, linewidths=0)
    ax.scatter(
        time_arr[decile_mask],
        mag[decile_mask],
        s=20,
        marker="D",
        alpha=0.9,
        label="Top/bottom deciles",
    )

    ax.axhline(mean_mag, color="0.4", ls=":", lw=1.4, label="Mean")
    ax.axhline(p10, color="tab:blue", ls="--", lw=1.3)
    ax.axhline(p90, color="tab:blue", ls="--", lw=1.3)

    ax.invert_yaxis()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Relative magnitude")
    ax.set_title(source_id)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)

    fig.savefig(
        f"{fig_dir}/{safe_target}_{safe_author}_{sector}_m_plot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # ---------- Q-style phase-folded plot ----------
    phase = ((time_arr - time_arr.min()) / ts_days) % 1
    order = np.argsort(phase)

    phase_sorted = phase[order]
    mag_sorted = mag[order]

    N = len(mag_sorted)
    window = int(0.25 * N)

    if window % 2 == 0:
        window += 1

    window = max(window, 5)

    if window >= N:
        window = N - 1 if N % 2 == 0 else N

    if window < 5:
        raise ValueError("Not enough points for Savitzky-Golay smoothing.")

    smooth_mag = savgol_filter(mag_sorted, window_length=window, polyorder=2)

    phase_ext = np.concatenate([
        phase_sorted - 1,
        phase_sorted,
        phase_sorted + 1,
    ])

    mag_ext = np.concatenate([
        mag_sorted,
        mag_sorted,
        mag_sorted,
    ])

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    ax.scatter(phase_ext, mag_ext, s=8, c="0.65", alpha=0.25, linewidths=0)
    ax.scatter(
        phase_sorted,
        mag_sorted,
        s=10,
        c="0.05",
        alpha=0.85,
        linewidths=0,
        label="Folded data",
    )

    ax.plot(
        phase_sorted,
        smooth_mag,
        color="tab:blue",
        lw=2.5,
        label="Smoothed waveform",
    )

    ax.axvline(0, color="0.55", ls=":", lw=2)
    ax.axvline(1, color="0.55", ls=":", lw=2)

    ax.axhline(mean_mag, color="0.4", ls=":", lw=1.4)
    ax.axhline(mean_mag + 3 * std_mag, color="0.15", ls="--", lw=1.2)
    ax.axhline(mean_mag - 3 * std_mag, color="0.15", ls="--", lw=1.2)

    ax.set_xlim(-0.5, 1.5)
    ax.invert_yaxis()
    ax.set_xlabel("Phase")
    ax.set_ylabel("Relative magnitude")
    ax.set_title(f"{source_id} | timescale = {ts_days:.3f} d")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)

    fig.savefig(
        f"{fig_dir}/{safe_target}_{safe_author}_{sector}_phase_folded.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return phase_sorted, mag_sorted, smooth_mag


def get_lightcurve(tic, author, sector):
    search = lk.search_lightcurve(f"TIC {tic}")

    if len(search) == 0:
        raise ValueError(f"No light curves found for TIC {tic}")

    mission_wanted = f"TESS Sector {sector:02d}"
    author_wanted = str(author).strip()

    mask = (
        (search.table["author"] == author_wanted)
        & (search.table["mission"] == mission_wanted)
    )

    matches = search[mask]

    if len(matches) == 0:
        print("\nAvailable products:")
        print(search)
        raise ValueError(
            f"No matching light curve product found for "
            f"author={author_wanted}, sector={sector}"
        )

    return matches[0].download()


INPUT_CSV = "StarCatalog.csv"
OUTPUT_CSV = "StarCatalog_metrics.csv"

df = pd.read_csv(INPUT_CSV)

for col in ["Q", "M", "Best_Period", "Error"]:
    if col not in df.columns:
        if col in ["Q", "M", "Best_Period"]:
            df[col] = np.nan
        else:
            df[col] = ""


for idx, row in df.iterrows():

    if pd.isna(row["TIC"]) or pd.isna(row["Sector"]) or pd.isna(row["Author"]):
        print(f"Skipping row {idx}: missing TIC, Author, or Sector", flush=True)
        continue

    row_start = time.time()

    try:
        tic = int(row["TIC"])
        sector = int(row["Sector"])
        author = str(row["Author"]).strip()

        target_name = (
            str(row["Target_Name"]).strip()
            if "Target_Name" in row and not pd.isna(row["Target_Name"])
            else f"TIC {tic}"
        )

        source_list = (
            str(row["Source_List"]).strip()
            if "Source_List" in row and not pd.isna(row["Source_List"])
            else "unknown"
        )

        print(f"\n========== Row {idx}: TIC {tic} ==========", flush=True)
        print(f"Author: {author}, Sector: {sector}", flush=True)

        print("Searching/downloading with Lightkurve...", flush=True)
        lc = get_lightcurve(tic, author, sector)

        print("Cleaning light curve...", flush=True)
        lc = lc.remove_nans().remove_outliers()

        time_arr = np.asarray(lc.time.value, dtype=float)

        flux = np.ma.filled(lc.flux.value, np.nan)
        flux = np.asarray(flux, dtype=float)

        flux_err = np.ma.filled(lc.flux_err.value, np.nan)
        flux_err = np.asarray(flux_err, dtype=float)

        clean = (
            np.isfinite(time_arr)
            & np.isfinite(flux)
            & np.isfinite(flux_err)
            & (flux > 0)
            & (flux_err > 0)
        )

        time_arr = time_arr[clean]
        flux = flux[clean]
        flux_err = flux_err[clean]

        mag = -2.5 * np.log10(flux)

        print(f"Cleaned light curve: {len(flux)} points", flush=True)

        if len(flux) < 10:
            raise ValueError("Too few valid points after cleaning.")

        # Lomb-Scargle periodogram
        y_norm = flux / np.nanmedian(flux)

        ls = LombScargle(time_arr, y_norm)

        pg = lk.LightCurve(
            time=time_arr,
            flux=y_norm,
        ).to_periodogram(method="lombscargle")

        fap_1_percent_power = ls.false_alarm_level(
            0.01,
            minimum_frequency=pg.frequency.min().value,
            maximum_frequency=pg.frequency.max().value,
            method="baluev",
        )

        max_power = pg.power.max().value

        if max_power < fap_1_percent_power:
            print("No significant GLS peak; using structure function.", flush=True)
            tau, sf, n_pairs, t0, popt = structure_func(time_arr, flux, mag)
            char_time = t0
        else:
            print("Significant GLS peak; using periodogram period.", flush=True)
            char_time = pg.period_at_max_power

        char_time_days = char_time.value if hasattr(char_time, "value") else float(char_time)

        print("Characteristic timescale =", char_time_days, "days", flush=True)

        # Fold and smooth
        phase = ((time_arr - time_arr.min()) / char_time_days) % 1
        order = np.argsort(phase)

        phase_sorted = phase[order]
        mag_sorted = mag[order]

        N = len(mag_sorted)
        window = int(0.25 * N)

        if window % 2 == 0:
            window += 1

        window = max(window, 5)

        if window >= N:
            window = N - 1 if N % 2 == 0 else N

        if window < 5:
            raise ValueError("Not enough points for Savitzky-Golay smoothing.")

        smooth_mag = savgol_filter(mag_sorted, window_length=window, polyorder=2)

        residual_mag = mag_sorted - smooth_mag

        # Q index
        rms_raw = np.nanstd(mag)
        rms_resid = np.nanstd(residual_mag)

        sigma_mag = (2.5 / np.log(10)) * flux_err / flux
        sigma2_mean = np.nanmean(sigma_mag**2)

        denominator = rms_raw**2 - sigma2_mean

        if denominator <= 0:
            Q = np.nan
        else:
            Q = (rms_resid**2 - sigma2_mean) / denominator

        print(f"Q = {Q:.3f}", flush=True)

        # M index in magnitude space
        print("Computing M...", flush=True)

        p10 = np.nanpercentile(mag, 10)
        p90 = np.nanpercentile(mag, 90)

        low_mean = np.nanmean(mag[mag <= p10])
        high_mean = np.nanmean(mag[mag >= p90])

        M = (0.5 * (low_mean + high_mean) - np.nanmedian(mag)) / np.nanstd(mag)

        print(f"M = {M:.3f}", flush=True)

        df.loc[idx, "Q"] = Q
        df.loc[idx, "M"] = M
        df.loc[idx, "Best_Period"] = char_time_days
        df.loc[idx, "Error"] = ""

        df.to_csv(OUTPUT_CSV, index=False)

        print(
            f"Finished TIC {tic}: Q={Q:.3f}, M={M:.3f}, "
            f"timescale={char_time_days:.4f} d",
            flush=True,
        )

        qm_plots(
            time_arr,
            flux,
            char_time_days,
            source_id=f"TIC {tic}",
            source_list=source_list,
            target_name=target_name,
            author=author,
            sector=sector,
        )

    except Exception as e:
        print(f"TIC {tic if 'tic' in locals() else idx}: {e}", flush=True)
        df.loc[idx, "Error"] = str(e)
        df.to_csv(OUTPUT_CSV, index=False)
        continue


df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved final output to {OUTPUT_CSV}", flush=True)