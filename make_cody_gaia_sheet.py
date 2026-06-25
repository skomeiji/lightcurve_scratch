import csv
import time
from astroquery.mast import Observations

INPUT_FILE = "cody_gaia_upload.csv"
OUTPUT_FILE = "cody_with_best_tess_sectors.csv"

# fallback priority
PIPELINES = [
    ["SPOC"],
    ["TESS-SPOC", "TESS_SPOC"],
    ["QLP"],
]


def clean(x):
    if x is None:
        return ""
    return str(x).strip()


def query_tess_observations(row):
    tic = clean(row["TIC"])

    # Query by TIC first
    try:
        obs = Observations.query_criteria(
            objectname=f"TIC {tic}",
            obs_collection="TESS",
            dataproduct_type="timeseries",
            radius="0.001 deg",
        )
        if len(obs) > 0:
            return obs
    except Exception:
        pass

    # Fallback: query by RA/Dec
    return Observations.query_criteria(
        coordinates=f"{row['ra']} {row['dec']}",
        obs_collection="TESS",
        dataproduct_type="timeseries",
        radius="0.001 deg",
    )


def get_sectors_by_pipeline(row):
    obs = query_tess_observations(row)

    sectors_by_pipeline = {}

    for group in PIPELINES:
        label = group[0]
        sectors = set()

        for ob in obs:
            provenance = clean(ob.get("provenance_name", "")).upper()
            project = clean(ob.get("project", "")).upper()
            seq = ob.get("sequence_number", None)

            if seq is None or clean(seq) == "":
                continue

            for name in group:
                name_upper = name.upper()
                if name_upper in provenance or name_upper in project:
                    sectors.add(int(seq))

        sectors_by_pipeline[label] = sorted(sectors)

    return sectors_by_pipeline


def choose_best_pipeline(sectors_by_pipeline):
    for group in PIPELINES:
        label = group[0]
        sectors = sectors_by_pipeline.get(label, [])
        if sectors:
            return label, sectors

    return "", []


with open(INPUT_FILE, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = list(reader.fieldnames)

for col in [
    "Sector",
    "Sector Source",
    "SPOC Sectors",
    "TESS-SPOC Sectors",
    "QLP Sectors",
]:
    if col not in fieldnames:
        fieldnames.append(col)

for i, row in enumerate(rows):
    print(f"{i+1}/{len(rows)} TIC {row['TIC']} {row['Target']}")

    try:
        sectors_by_pipeline = get_sectors_by_pipeline(row)
        source, sectors = choose_best_pipeline(sectors_by_pipeline)

        row["Sector"] = ",".join(map(str, sectors))
        row["Sector Source"] = source

        row["SPOC Sectors"] = ",".join(map(str, sectors_by_pipeline.get("SPOC", [])))
        row["TESS-SPOC Sectors"] = ",".join(map(str, sectors_by_pipeline.get("TESS-SPOC", [])))
        row["QLP Sectors"] = ",".join(map(str, sectors_by_pipeline.get("QLP", [])))

        print(f"  Using {source or 'none'}: {row['Sector'] or 'none'}")

    except Exception as e:
        print(f"  failed: {e}")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    time.sleep(0.2)

print(f"Saved {OUTPUT_FILE}")