import lightkurve as lk
import matplotlib.pyplot as plt


def get_lightcurve_periodogram(
    target,
    *,
    author=None,
    sector=None,
    quarter=None
):
    search_result = lk.search_lightcurve(
        target,
        author=author,
        sector=sector,
        quarter=quarter
    )

    print(search_result)

    if len(search_result) == 0:
        raise ValueError("No light curves found.")

    lc = search_result[0].download(quality_bitmask='hardest')
    lc = lc.remove_nans().normalize()

    # No minimum_period, maximum_period, minimum_frequency, or maximum_frequency
    pg = lc.to_periodogram(normalization="amplitude")

    return lc, pg


def plot_lightcurve_and_frequency_periodogram(lc, pg, target_name="Target"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    lc.plot(ax=axes[0])
    axes[0].set_title(f"{target_name} Light Curve")

    axes[1].plot(pg.frequency.value, pg.power.value)
    axes[1].set_xlabel(f"Frequency [{pg.frequency.unit}]")
    axes[1].set_ylabel("Power")
    axes[1].set_title(f"{target_name} Periodogram")

    plt.tight_layout()
    return fig, axes


def save_lightcurve_and_frequency_periodogram(
    lc,
    pg,
    filename,
    target_name="Target"
):
    fig, axes = plot_lightcurve_and_frequency_periodogram(
        lc,
        pg,
        target_name=target_name
    )

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    return fig, axes