"""Ancestry composition inference using gnomAD v4 population allele frequencies.

Approach:
  1. For each ancestry-informative marker (AIM), query gnomAD v4's GraphQL API
     in two steps: first resolve the rsID to a variantId, then fetch per-population
     allele counts. AF is computed as AC / AN.
  2. Cache the result locally so all subsequent runs are fully offline.
  3. Use Hardy-Weinberg log-likelihood across all markers to score how well the
     user's genotypes match each gnomAD population, then normalise via softmax
     and aggregate to superpopulations for display.
"""

import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Ancestry-informative markers ───────────────────────────────────────────
# Selected for high Fst across continental groups and presence on major
# genotyping arrays (23andMe v3/v4/v5, Illumina OmniExpress).
ANCESTRY_AIMS = [
    # Very high Fst — strongest continental discriminators
    "rs1426654",   # SLC24A5  skin pigmentation        AFR↑  EUR↓  EAS↓
    "rs2814778",   # DARC     Duffy-null antigen        AFR↑↑ others↓
    "rs3827760",   # EDAR     hair / teeth morphology   EAS↑  AFR↓  EUR↓
    "rs671",       # ALDH2    alcohol metabolism        EAS↑  others↓
    "rs16891982",  # SLC45A2  skin pigmentation         EUR↑  AFR↓  EAS↓
    "rs17822931",  # ABCC11   earwax / body odour       EAS↑  AFR↓  EUR↓
    # Medium Fst
    "rs12913832",  # HERC2    eye colour                EUR↑
    "rs4988235",   # LCT      lactase persistence       EUR↑
    "rs182549",    # LCT region
    "rs6025",      # F5       Factor V Leiden           EUR↑
    "rs1800562",   # HFE      hemochromatosis           EUR↑
    "rs2476601",   # PTPN22   autoimmune                EUR↑
    "rs1805008",   # MC1R     red hair                  EUR↑
    "rs1805007",   # MC1R     red hair
    "rs2228479",   # MC1R     V92M
    "rs1042602",   # TYR      pigmentation
    "rs1800414",   # OCA2     East-Asian pigmentation   EAS↑
    "rs1800401",   # OCA2     pigmentation
    "rs28777",     # SLC45A2  region
    "rs7495174",   # OCA2     region
    "rs1129038",   # HERC2    region
    # Drug-metabolism markers with notable population variation
    "rs762551",    # CYP1A2   caffeine
    "rs4244285",   # CYP2C19  clopidogrel / PPIs
    "rs3892097",   # CYP2D6   drug metabolism
    "rs1799853",   # CYP2C9   warfarin
    "rs1057910",   # CYP2C9*3 warfarin
    # Additional markers with ancestry signal
    "rs429358",    # APOE     e4  (AFR > EUR > EAS)
    "rs7412",      # APOE     e2
    "rs1801133",   # MTHFR    C677T
    "rs9939609",   # FTO      obesity
    "rs7903146",   # TCF7L2   T2D
    "rs2187668",   # HLA-DQA1 celiac (EUR)
    "rs1815739",   # ACTN3    power / endurance
    "rs4680",      # COMT     dopamine clearance
    "rs6265",      # BDNF     neuroplasticity
]

# ── Population mappings ────────────────────────────────────────────────────
# gnomAD v4 top-level ancestry labels (excluding sex-stratified and sub-cohorts)
GNOMAD_POPS = {
    "afr": "African / African-American",
    "amr": "Latino / Admixed American",
    "ami": "Amish",
    "asj": "Ashkenazi Jewish",
    "eas": "East Asian",
    "fin": "Finnish",
    "mid": "Middle Eastern",
    "nfe": "Non-Finnish European",
    "sas": "South Asian",
}

# Collapse to 6 display superpopulations
SUPERPOP_MAP = {
    "afr": "AFR",
    "amr": "AMR",
    "ami": "EUR",   # Amish are European-American
    "asj": "EUR",   # Ashkenazi Jewish closest to European
    "eas": "EAS",
    "fin": "EUR",
    "mid": "MID",
    "nfe": "EUR",
    "sas": "SAS",
}

SUPERPOP_LABELS = {
    "AFR": "African",
    "AMR": "Admixed American",
    "EAS": "East Asian",
    "EUR": "European",
    "MID": "Middle Eastern",
    "SAS": "South Asian",
}

SUPERPOP_COLORS = {
    "AFR": "#e53935",
    "AMR": "#ff9800",
    "EAS": "#4caf50",
    "EUR": "#1565c0",
    "MID": "#00897b",
    "SAS": "#9c27b0",
}

GNOMAD_API  = "https://gnomad.broadinstitute.org/api"
_DATASET    = "gnomad_r4"
_CACHE_FILE = "aim_frequencies.json"
_MIN_AN     = 200   # minimum allele number to trust a population frequency


# ── gnomAD v4 two-step query ───────────────────────────────────────────────

def _gql(query: str) -> dict:
    payload = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        GNOMAD_API, data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read()).get("data") or {}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                time.sleep(wait)
            else:
                raise
    return {}


def _best_variant_id(rsid: str) -> str | None:
    """Resolve rsID → best SNP variantId via variant_search."""
    data = _gql('{ variant_search(query: "%s", dataset: %s) { variant_id } }' % (rsid, _DATASET))
    hits = data.get("variant_search") or []
    snps = [h["variant_id"] for h in hits
            if len((h["variant_id"] or "").split("-")) == 4
            and len(h["variant_id"].split("-")[2]) == 1
            and len(h["variant_id"].split("-")[3]) == 1]
    if not snps:
        return None
    # Prefer non-strand-ambiguous allele pairs (avoid A/T and G/C)
    non_ambig = [v for v in snps if set(v.split("-")[2:4]) not in ({"A","T"}, {"G","C"})]
    return (non_ambig or snps)[0]


def _fetch_populations(variant_id: str) -> dict:
    """Fetch per-population AC/AN for a specific variantId."""
    data = _gql(
        '{ variant(variantId: "%s", dataset: %s) '
        '{ genome { populations { id ac an } } } }' % (variant_id, _DATASET)
    )
    variant = data.get("variant") or {}
    pops = (variant.get("genome") or {}).get("populations") or []
    result = {}
    for p in pops:
        pid = p["id"]
        # Skip sex-stratified, HGDP, 1KG sub-cohorts, and "remaining"
        if ":" in pid or "_" in pid or pid in ("XX", "XY", "remaining"):
            continue
        if pid not in GNOMAD_POPS:
            continue
        an, ac = p.get("an") or 0, p.get("ac") or 0
        if an >= _MIN_AN:
            result[pid] = ac / an
    return result


def _query_gnomad(rsid: str) -> dict | None:
    """Return {rsid, variantId, ref, alt, pops: {pop_id: af}} or None."""
    try:
        variant_id = _best_variant_id(rsid)
        if not variant_id:
            return None
        time.sleep(0.2)
        pops = _fetch_populations(variant_id)
        if not pops:
            return None
        parts = variant_id.split("-")
        ref, alt = parts[2], parts[3]
        return {"rsid": rsid, "variantId": variant_id, "ref": ref, "alt": alt, "pops": pops}
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return None


# ── Public API ─────────────────────────────────────────────────────────────

def download_aim_frequencies(data_dir: Path) -> Path:
    """Query gnomAD v4 for all AIM frequencies and cache to JSON.

    Safe to re-run — only fetches markers not already in the cache.
    """
    import sys
    ancestry_dir = Path(data_dir) / "ancestry"
    ancestry_dir.mkdir(exist_ok=True)
    cache_path = ancestry_dir / _CACHE_FILE

    existing: dict = {}
    if cache_path.exists():
        with open(cache_path) as f:
            existing = json.load(f)

    to_fetch = [r for r in ANCESTRY_AIMS if r not in existing]
    if not to_fetch:
        print(f"All {len(ANCESTRY_AIMS)} markers cached at {cache_path}", file=sys.stderr)
        return cache_path

    print(f"Querying gnomAD v4 for {len(to_fetch)} markers…", file=sys.stderr)
    found = 0
    for i, rsid in enumerate(to_fetch):
        print(f"  [{i+1}/{len(to_fetch)}] {rsid}", end="  ", flush=True, file=sys.stderr)
        result = _query_gnomad(rsid)
        time.sleep(0.2)  # stay polite
        if result:
            existing[rsid] = result
            found += 1
            print(f"✓  ({result['variantId']})", file=sys.stderr)
        else:
            print("—  not found in gnomAD v4", file=sys.stderr)
        time.sleep(1.5)   # stay well within gnomAD rate limits

    with open(cache_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n{found}/{len(to_fetch)} markers saved → {cache_path}", file=sys.stderr)
    return cache_path


def load_aim_frequencies(data_dir: Path) -> dict:
    """Load cached AIM frequencies. Raises FileNotFoundError if not downloaded yet."""
    cache_path = Path(data_dir) / "ancestry" / _CACHE_FILE
    if not cache_path.exists():
        raise FileNotFoundError(
            "Ancestry reference not found.\n"
            "Run the download cell in section 7 of the workshop notebook first."
        )
    with open(cache_path) as f:
        return json.load(f)


def compute_ancestry(genome_by_rsid: dict, frequencies: dict) -> tuple[dict, int]:
    """Hardy-Weinberg log-likelihood ancestry inference.

    For each AIM present in the genome, compute log P(genotype | population)
    under HWE, accumulate across all markers, then normalise via softmax and
    aggregate to superpopulations.

    Returns (superpop_probs: dict[str, float], n_markers_used: int).
    """
    log_likes: dict[str, float] = {p: 0.0 for p in GNOMAD_POPS}
    n_used = 0

    for rsid, freq_data in frequencies.items():
        if rsid not in genome_by_rsid:
            continue

        gt  = genome_by_rsid[rsid]["genotype"]
        ref = freq_data["ref"]
        alt = freq_data["alt"]

        if len(gt) == 2:
            if not all(a in (ref, alt) for a in gt):
                continue
            dosage = sum(1 for a in gt if a == alt)
        elif len(gt) == 1:
            if gt not in (ref, alt):
                continue
            dosage = 1 if gt == alt else 0
        else:
            continue

        for pop_id, af in freq_data["pops"].items():
            if pop_id not in log_likes:
                continue
            af = max(1e-6, min(1 - 1e-6, float(af)))
            if dosage == 0:
                ll = 2 * math.log(1 - af)
            elif dosage == 1:
                ll = math.log(2) + math.log(af) + math.log(1 - af)
            else:
                ll = 2 * math.log(af)
            log_likes[pop_id] += ll

        n_used += 1

    if n_used == 0:
        return {sp: 1 / len(SUPERPOP_LABELS) for sp in SUPERPOP_LABELS}, 0

    # Softmax over gnomAD populations
    max_ll = max(log_likes.values())
    exp_ll = {p: math.exp(ll - max_ll) for p, ll in log_likes.items()}
    total  = sum(exp_ll.values())
    probs  = {p: v / total for p, v in exp_ll.items()}

    # Aggregate to superpopulations
    super_probs: dict[str, float] = {}
    for pop_id, prob in probs.items():
        sp = SUPERPOP_MAP.get(pop_id)
        if sp:
            super_probs[sp] = super_probs.get(sp, 0.0) + prob

    return super_probs, n_used
