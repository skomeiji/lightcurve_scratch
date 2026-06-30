import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from tqdm import tqdm
import time
import numpy as np
import os

INPUT_CSV = "taurus_DECO.csv"
OUTPUT_CSV = "taurus_DECO_enriched.csv"

df = pd.read_csv(INPUT_CSV)

name_col = df.columns[0]
ra_col = "RA"
dec_col = "Dec"

Simbad.reset_votable_fields()
Simbad.add_votable_fields("ids", "sp", "otype")

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"


def clean_name(x):
    x = str(x).strip()

    if x.startswith("2MASS"):
        return x

    # Only add 2MASS prefix for actual 2MASS-style names
    if x.startswith("J") and any(char.isdigit() for char in x):
        return "2MASS " + x

    # For names like Sz111, MYLup, HTLup, etc.
    return x


def safe_value(x):
    if x is None:
        return None
    if np.ma.is_masked(x):
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def get_col(row, possible_names):
    for name in possible_names:
        if name in row.colnames:
            return safe_value(row[name])
    return None


def query_simbad(raw_name, ra_deg=None, dec_deg=None, radius_arcsec=5):
    obj_name = clean_name(raw_name)

    out = {
        "simbad_query_name": obj_name,
        "simbad_name": None,
        "tic_id_simbad": None,
        "startype": None,
        "simbad_otype": None,
        "simbad_error": None,
        "simbad_sep_arcsec": None,
    }

    try:
        # First try by name
        result = Simbad.query_object(obj_name)

        # If name query fails, try by RA/Dec
        if (result is None or len(result) == 0) and ra_deg is not None and dec_deg is not None:
            coord = SkyCoord(ra_deg, dec_deg, unit="deg", frame="icrs")
            result = Simbad.query_region(
                coord,
                radius=radius_arcsec * u.arcsec
            )

        if result is None or len(result) == 0:
            return out

        row = result[0]

        main_id = get_col(row, ["MAIN_ID", "main_id"])
        sp_type = get_col(row, ["SP_TYPE", "sp_type", "SP", "sp"])
        otype = get_col(row, ["OTYPE", "otype"])
        ids = get_col(row, ["IDS", "ids"])

        out["simbad_name"] = str(main_id) if main_id is not None else None
        out["startype"] = str(sp_type) if sp_type is not None else None
        out["simbad_otype"] = str(otype) if otype is not None else None

        if ids is not None:
            for ident in str(ids).split("|"):
                ident = ident.strip()
                if ident.startswith("TIC"):
                    out["tic_id_simbad"] = ident
                    break

    except Exception as e:
        out["simbad_error"] = str(e)

    return out


def query_tic_from_mast(ra_deg, dec_deg, radius_arcsec=5):
    try:
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")

        tic = Catalogs.query_region(
            coord,
            radius=radius_arcsec * u.arcsec,
            catalog="TIC"
        )

        if len(tic) == 0:
            return None

        tic.sort("dstArcSec")
        return int(tic[0]["ID"])

    except Exception:
        return None


def query_tic_teff(tic_id):
    if tic_id is None:
        return None

    try:
        if pd.isna(tic_id):
            return None
    except Exception:
        pass

    tic_str = str(tic_id).replace("TIC", "").strip()

    if tic_str == "":
        return None

    try:
        catalog_data = Catalogs.query_object(
            f"TIC {tic_str}",
            radius=0.001,
            catalog="TIC"
        )

        if len(catalog_data) == 0:
            return None

        if "Teff" not in catalog_data.colnames:
            return None

        teff = catalog_data[0]["Teff"]
        teff = safe_value(teff)

        if teff is None:
            return None

        return float(teff)

    except Exception:
        return None


def query_gaia_dr3(ra_deg, dec_deg, radius_arcsec=5):
    out = {
        "gaia_dr3_source_id": None,
        "plx": None,
        "gmag": None,
        "bp_rp": None,
        "gaia_sep_arcsec": None,
        "gaia_error": None,
    }

    try:
        query = f"""
        SELECT TOP 1
            source_id,
            parallax,
            phot_g_mean_mag,
            bp_rp,
            DISTANCE(
                POINT('ICRS', ra, dec),
                POINT('ICRS', {ra_deg}, {dec_deg})
            ) * 3600.0 AS sep_arcsec
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_arcsec / 3600.0})
        )
        ORDER BY sep_arcsec ASC
        """

        job = Gaia.launch_job_async(query)
        result = job.get_results()

        if len(result) == 0:
            return out

        row = result[0]

        out["gaia_dr3_source_id"] = str(row["source_id"])
        out["plx"] = safe_value(row["parallax"])
        out["gmag"] = safe_value(row["phot_g_mean_mag"])
        out["bp_rp"] = safe_value(row["bp_rp"])
        out["gaia_sep_arcsec"] = safe_value(row["sep_arcsec"])

    except Exception as e:
        out["gaia_error"] = str(e)

    return out


rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    raw_name = row[name_col]
    ra = float(row[ra_col])
    dec = float(row[dec_col])

    simbad_info = query_simbad(raw_name, ra, dec)
    gaia_info = query_gaia_dr3(ra, dec)

    tic_id_mast = query_tic_from_mast(ra, dec)

    tic_id = simbad_info["tic_id_simbad"] or tic_id_mast
    teff = query_tic_teff(tic_id)

    rows.append({
        **row.to_dict(),
        **simbad_info,
        "tic_id_mast": tic_id_mast,
        "tic_id_used_for_teff": tic_id,
        "teff": teff,
        **gaia_info,
    })

    time.sleep(0.2)


out = pd.DataFrame(rows)

print(out.columns.tolist())
print(out[[
    name_col,
    "simbad_query_name",
    "simbad_name",
    "tic_id_simbad",
    "tic_id_mast",
    "teff",
    "plx",
    "gmag",
    "bp_rp"
]].head())

out.to_csv(OUTPUT_CSV, index=False)

print(f"Wrote {OUTPUT_CSV}")
print("Absolute output path:", os.path.abspath(OUTPUT_CSV))