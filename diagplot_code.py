from astroquery.vizier import Vizier
import numpy as np
import lightkurve_utils as utils
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import gaussian_filter1d
from astroquery.gaia import Gaia
import csv
from pathlib import Path

# HR Diagram data from Gaia DR2
filename = "gaia-hrd-dr2-200pc.csv"

query = """
SELECT TOP 10000000
    bp_rp,
    phot_g_mean_mag + 5 * LOG10(parallax) - 10 AS mg
FROM gaiadr2.gaia_source
WHERE parallax_over_error > 10
  AND visibility_periods_used > 8
  AND phot_g_mean_flux_over_error > 50
  AND phot_bp_mean_flux_over_error > 20
  AND phot_rp_mean_flux_over_error > 20
  AND phot_bp_rp_excess_factor <
      1.3 + 0.06 * POWER(phot_bp_mean_mag - phot_rp_mean_mag, 2)
  AND phot_bp_rp_excess_factor >
      1.0 + 0.015 * POWER(phot_bp_mean_mag - phot_rp_mean_mag, 2)
  AND astrometric_chi2_al / (astrometric_n_good_obs_al - 5) <
      1.44 * GREATEST(1, EXP(-0.4 * (phot_g_mean_mag - 19.5)))
  AND 1000 / parallax <= 200
"""

try:
    gaiarec = np.genfromtxt(
        filename,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    bp_rp = gaiarec["bp_rp"]
    mg = gaiarec["mg"]

except OSError:
    job = Gaia.launch_job_async(
        query,
        dump_to_file=True,
        output_file=filename,
        output_format="csv",
    )
    r = job.get_results()
    bp_rp = np.array(r["bp_rp"])
    mg = np.array(r["mg"])


start_line = 682

with open("StarCatalog.csv", "r", newline="") as f:
    reader = csv.reader(f)

    for _ in range(start_line):
        next(reader, None)

    rows = list(reader)

for i, row in enumerate(rows):
    next_row = rows[i + 1] if i + 1 < len(rows) else None

    print(row)

    # Convert numeric CSV entries from strings to floats
    star_bp_rp = float(row[7])
    plx = float(row[5])
    gmag = float(row[6])

    distance_pc = 1000.0 / plx
    star_mg = gmag + 5 - 5 * np.log10(distance_pc)

    # Check whether the next row is the same TIC/star
    if (
            next_row is not None
            and row[1] == next_row[1]        # same TIC
            and row[2] == next_row[2]        # same author
            and int(next_row[3]) == int(row[3]) + 1  # consecutive sectors
        ):
            sectors = [row[3], next_row[3]]
    else:
            sectors = [row[3]]

    fig = plt.figure(layout="constrained", figsize=(12, 14))
    axd = fig.subplot_mosaic(
        """
        FG
        AA
        BC
        DE
        """
    )

    # Information box
    axd["F"].axis("off")

    info = (
        f"Name: {row[0]}\n"
        f"Type: {row[8]}\n"
        f"Spectral Type: {row[9]}\n"
        f"T_eff: {row[4]}\n"
        f"TIC: {row[1]}\n"
        f"Author: {row[2]}\n"
        f"Sector: {row[3]}\n"
    )

    axd["F"].text(
        0.05,
        0.95,
        info,
        transform=axd["F"].transAxes,
        va="top",
        fontsize=20,
    )

    # HR Diagram
    h = axd["G"].hist2d(
        bp_rp,
        mg,
        bins=300,
        cmin=10,
        norm=colors.PowerNorm(0.5),
        cmap="gray",
    )

    axd["G"].invert_yaxis()
    axd["G"].set_xlabel(r"$G_{BP} - G_{RP}$")
    axd["G"].set_ylabel(r"$M_G$")

    axd["G"].scatter(
        star_bp_rp,
        star_mg,
        marker="*",
        s=400,
        color="red",
        edgecolors="black",
        linewidths=1.5,
        zorder=100,
        label=row[0],
    )

    cbar = fig.colorbar(h[3], ax=axd["G"])
    cbar.set_label("Stellar density")

    # Get lightcurve and periodogram for first sector
    lc, pg = utils.get_lightcurve_periodogram(
        f"TIC {row[1]}",
        author=row[2],
        sector=sectors[0],
    )

    t = lc.time.value
    f_flux = lc.flux.value

    mask = np.isfinite(t) & np.isfinite(f_flux)
    t = t[mask]
    f_flux = f_flux[mask]

    idx = np.argsort(t)
    t = t[idx]
    f_flux = f_flux[idx]

    f_smooth = gaussian_filter1d(f_flux, sigma=50)

    # Raw lightcurve
    lc.plot(ax=axd["A"])
    axd["A"].set_xlabel("Time")
    axd["A"].set_ylabel("Flux")
    axd["A"].set_title("Lightcurve")

    axd["A"].plot(
        t,
        f_smooth,
        linewidth=2,
        color="red",
        label="Large-scale trend",
    )

    axd["A"].legend()

    # Flattened lightcurve
    time = lc.time.value
    flat_lc = lc.flatten(window_length=51)
    flux = flat_lc.flux.value

    finite_flux = flux[np.isfinite(flux)]
    ymin = finite_flux.min()
    ymax = finite_flux.max()
    padding = 0.05 * (ymax - ymin)

    flat_lc.plot(ax=axd["B"])
    axd["B"].set_xlabel("Time")
    axd["B"].set_ylabel("Flux")
    axd["B"].set_title("Flattened Lightcurve")
    axd["B"].set_xlim(time.min(), time.min() + 1.5)
    axd["B"].set_ylim(ymin - padding, ymax + padding)

    legend = axd["B"].get_legend()
    if legend is not None:
        legend.remove()

    # Low-frequency periodogram
    axd["C"].plot(pg.frequency.value, pg.power.value)
    axd["C"].set_xlabel(f"Frequency [{pg.frequency.unit}]")
    axd["C"].set_ylabel("Power")

    freq = pg.frequency.value
    power = pg.power.value

    low_mask = (
        np.isfinite(freq)
        & np.isfinite(power)
        & (freq >= 0)
        & (freq <= 5)
    )

    low_power = power[low_mask]

    low_ymin = low_power.min()
    low_ymax = low_power.max()
    low_pad = 0.05 * (low_ymax - low_ymin)

    axd["C"].set_xlim(0, 5)
    axd["C"].set_ylim(0, low_ymax + low_pad)
    axd["C"].set_title("Low-Frequency Periodogram")

    # High-frequency periodogram
    freq = np.asarray(pg.frequency.value, dtype=float)
    power = np.asarray(pg.power.value, dtype=float)

    high_max = min(100, np.nanmax(freq))

    high_mask = (
        np.isfinite(freq)
        & np.isfinite(power)
        & (freq >= 5)
        & (freq <= high_max)
    )

    high_power = power[high_mask]

    axd["E"].plot(freq, power)
    axd["E"].set_xlabel(f"Frequency [{pg.frequency.unit}]")
    axd["E"].set_ylabel("Power")
    high_max = min(100, freq.max())
    high_mask = (
        np.isfinite(freq)
        & np.isfinite(power)
        & (freq >= 5)
        & (freq <= high_max)
    )

    high_power = power[high_mask]

    high_ymin = high_power.min()
    high_ymax = high_power.max()
    high_pad = 0.05 * (high_ymax - high_ymin)

    axd["E"].set_xlim(5, high_max)
    axd["E"].set_ylim(high_ymin - high_pad, high_ymax + high_pad)
    axd["E"].set_title("High-Frequency Periodogram")

    # Low-pass trends for all relevant sectors
    sector_colors = {}

    if len(sectors) >= 1:
        sector_colors[sectors[0]] = "blue"
    if len(sectors) >= 2:
        sector_colors[sectors[1]] = "red"

    for sector in sectors:
        lc_sector, pg_sector = utils.get_lightcurve_periodogram(
            f"TIC {row[1]}",
            author=row[2],
            sector=sector,
        )

        lc_sector = lc_sector.remove_nans().normalize()

        flat_lc_sector, trend_lc = lc_sector.flatten(
            window_length=201,
            return_trend=True,
        )

        axd["D"].plot(
            trend_lc.time.value,
            trend_lc.flux.value,
            linewidth=2,
            color=sector_colors.get(sector, None),
            label=f"Sector {sector}",
        )

    axd["D"].set_xlabel("Time [BTJD]")
    axd["D"].set_ylabel("Normalized Flux")
    axd["D"].set_title("Low-pass Trends")
    axd["D"].legend()

    outdir = Path(f"figures/Cody/{row[0]}")
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / f"{row[0]}_sec{row[3]}_{row[2]}_summary.png"

    print(f"About to save: {outpath.resolve()}")

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {outpath.resolve()}")
    plt.close(fig)