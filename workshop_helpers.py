"""Visualization and pipeline helpers for the analyze-your-dna workshop notebook."""

import csv
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MAG_COLORS  = ["#4caf50", "#8bc34a", "#ffc107", "#ff7043", "#e53935"]
CAT_COLORS  = {
    "pathogenic":        "#e53935",
    "likely_pathogenic": "#ff7043",
    "risk_factor":       "#ffc107",
    "drug_response":     "#42a5f5",
    "protective":        "#66bb6a",
    "other":             "#b0bec5",
}
LEVEL_ORDER  = {"1A": 0, "1B": 1, "2A": 2, "2B": 3, "3": 4, "4": 5}
LEVEL_COLORS = ["#1565c0", "#1976d2", "#42a5f5", "#90caf9", "#b0bec5", "#eceff1"]
CHROM_ORDER  = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
BARS         = {0: "○○○○", 1: "●○○○", 2: "●●○○", 3: "●●●○", 4: "●●●●", 5: "●●●●!", 6: "●●●●!!"}



def plot_ancestry_composition(super_probs: dict, n_markers: int = 0, subject_name: str = ""):
    """Horizontal bar chart of ancestry composition probabilities."""
    from analyze_dna.ancestry import SUPERPOP_COLORS, SUPERPOP_LABELS

    sorted_pops = sorted(super_probs.items(), key=lambda x: -x[1])
    labels = [SUPERPOP_LABELS[p] for p, _ in sorted_pops]
    values = [v * 100 for _, v in sorted_pops]
    colors = [SUPERPOP_COLORS[p] for p, _ in sorted_pops]

    fig, ax = plt.subplots(figsize=(10, max(3, len(sorted_pops) * 0.7)))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.8, height=0.55)

    for bar, val in zip(bars, values[::-1]):
        if val >= 0.5:
            ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=12, fontweight="bold",
                    color="#212121")

    ax.set_xlim(0, 115)
    ax.set_xlabel("Estimated ancestry (%)", fontsize=11)
    title = f"Ancestry Composition — {subject_name}" if subject_name else "Ancestry Composition"
    subtitle = f"{n_markers} markers · method: Hardy-Weinberg log-likelihood · reference: gnomAD v3" if n_markers else ""
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    top_pop, top_pct = sorted_pops[0]
    print(f"Primary ancestry: {SUPERPOP_LABELS[top_pop]} ({top_pct*100:.1f}%)")
    for pop, pct in sorted_pops[1:]:
        if pct * 100 >= 1.0:
            print(f"                  {SUPERPOP_LABELS[pop]} ({pct*100:.1f}%)")


_TH = [("background-color", "#37474f"), ("color", "white"),
       ("font-size", "13px"), ("padding", "6px 10px"), ("text-align", "left")]
_TD = [("border-bottom", "1px solid #eceff1")]
_CELL = {"text-align": "left", "font-size": "13px", "padding": "4px 10px"}


def _base_style(df):
    from IPython.display import display
    styled = (df.style
               .set_properties(**_CELL)
               .set_table_styles([{"selector": "th", "props": _TH},
                                  {"selector": "td", "props": _TD}])
               .hide(axis="index"))
    display(styled)


def display_snp_lookups(genome_by_rsid, snps):
    import pandas as pd
    from IPython.display import display

    rows = []
    for rsid, label in snps:
        data = genome_by_rsid.get(rsid)
        rows.append({
            "rsID":        rsid,
            "Description": label,
            "Chr":         data["chromosome"] if data else "—",
            "Position":    data["position"]   if data else "—",
            "Genotype":    data["genotype"]   if data else "not found",
            "_found":      data is not None,
        })

    df = pd.DataFrame(rows)

    def style_row(row):
        if not row["_found"]:
            base = "background-color: #f5f5f5; color: #9e9e9e;"
        else:
            base = "background-color: #e8f5e9; color: #1b5e20;"
        bold = base + " font-weight: bold;"
        return [bold if col == "Genotype" else base for col in row.index]

    styled = (df.style.apply(style_row, axis=1)
               .set_properties(**_CELL)
               .set_table_styles([{"selector": "th", "props": _TH},
                                  {"selector": "td", "props": _TD}])
               .hide(axis="index")
               .hide(subset=["_found"], axis="columns"))
    display(styled)


def display_pathogenic_findings(clinvar_findings):
    import pandas as pd
    from IPython.display import display

    CAT_BG = {"pathogenic": "#fde8e8", "likely_pathogenic": "#fff3e0"}
    CAT_FG = {"pathogenic": "#b71c1c", "likely_pathogenic": "#e65100"}
    STAR_ICONS = {0: "☆☆☆☆", 1: "⭐☆☆☆", 2: "⭐⭐☆☆", 3: "⭐⭐⭐☆", 4: "⭐⭐⭐⭐"}

    rows = []
    for cat in ("pathogenic", "likely_pathogenic"):
        for f in clinvar_findings[cat]:
            rows.append({
                "Significance": cat.replace("_", " ").title(),
                "Gene":         f["gene"] or "—",
                "rsID":         f["rsid"],
                "Genotype":     f["genotype"],
                "Evidence":     STAR_ICONS.get(f["gold_stars"], "?"),
                "Status":       zygosity_status(f),
                "Condition":    (f["traits"].split(";")[0][:60] if f["traits"] else "—"),
                "_cat":         cat,
            })

    if not rows:
        print("No pathogenic / likely pathogenic variants found."); return

    rows.sort(key=lambda r: (r["_cat"] != "pathogenic", -list(STAR_ICONS).index(
        next(k for k, v in STAR_ICONS.items() if v == r["Evidence"]))))

    df = pd.DataFrame(rows)

    def style_row(row):
        bg  = CAT_BG.get(row["_cat"], "#ffffff")
        fg  = CAT_FG.get(row["_cat"], "#000000")
        base = f"background-color: {bg}; color: {fg};"
        bold = base + " font-weight: bold;"
        return [bold if col in ("Significance", "Gene") else base for col in row.index]

    styled = (df.style.apply(style_row, axis=1)
               .set_properties(**_CELL)
               .set_table_styles([{"selector": "th", "props": _TH},
                                  {"selector": "td", "props": _TD}])
               .hide(axis="index")
               .hide(subset=["_cat"], axis="columns"))
    display(styled)


def display_drug_interactions(drug_interactions, levels=("1A", "1B")):
    import pandas as pd
    from IPython.display import display

    LVL_BG = {"1A": "#e3f2fd", "1B": "#e8eaf6", "2A": "#f3e5f5",
               "2B": "#fce4ec", "3": "#f5f5f5", "4": "#fafafa"}
    LVL_FG = {"1A": "#0d47a1", "1B": "#1a237e", "2A": "#4a148c",
               "2B": "#880e4f", "3": "#424242", "4": "#757575"}

    filtered = sorted(
        [d for d in drug_interactions if d["level"] in levels],
        key=lambda x: LEVEL_ORDER.get(x["level"], 99)
    )
    if not filtered:
        print(f"No level {'/'.join(levels)} drug interactions found."); return

    rows = [{
        "Level":      d["level"],
        "Gene":       d["gene"] or "—",
        "rsID":       d["rsid"],
        "Genotype":   d["genotype"],
        "Drug(s)":    d["drugs"][:40] if d["drugs"] else "—",
        "Category":   d["category"],
        "Annotation": d["annotation"][:120] + ("…" if len(d["annotation"]) > 120 else ""),
    } for d in filtered]

    df = pd.DataFrame(rows)

    def style_row(row):
        lvl  = row["Level"]
        bg   = LVL_BG.get(lvl, "#ffffff")
        fg   = LVL_FG.get(lvl, "#000000")
        base = f"background-color: {bg}; color: {fg};"
        bold = base + " font-weight: bold;"
        return [bold if col in ("Level", "Gene") else base for col in row.index]

    styled = (df.style.apply(style_row, axis=1)
               .set_properties(**_CELL)
               .set_table_styles([{"selector": "th", "props": _TH},
                                  {"selector": "td", "props": _TD}])
               .hide(axis="index"))
    display(styled)


def _fmt(ax, title, xlabel=None, ylabel=None, grid_axis="y"):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, fontsize=11)
    if grid_axis: ax.grid(axis=grid_axis, alpha=0.3)


# ── Pipeline functions ─────────────────────────────────────────────────────

def load_genome(genome_path):
    """Parse a 23andMe TSV into rsid and position lookup dicts."""
    genome_by_rsid, genome_by_position, no_calls = {}, {}, 0
    with open(genome_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            rsid, chrom, pos, genotype = parts[0], parts[1], parts[2], parts[3]
            if genotype == "--":
                no_calls += 1
                continue
            genome_by_rsid[rsid] = {"chromosome": chrom, "position": pos, "genotype": genotype}
            genome_by_position[f"{chrom}:{pos}"] = {"rsid": rsid, "genotype": genotype}
    return genome_by_rsid, genome_by_position, no_calls


def lookup_snp(genome_by_rsid, rsid, label=""):
    data = genome_by_rsid.get(rsid)
    tag  = f" ({label})" if label else ""
    if data:
        print(f"{rsid}{tag:38s}  chr{data['chromosome']}:{data['position']}  {data['genotype']}")
    else:
        print(f"{rsid}{tag:38s}  not found")


def scan_clinvar(clinvar_path, genome_by_position):
    """Scan ClinVar for variants present in the genome. Skips indels."""
    findings = {k: [] for k in ["pathogenic", "likely_pathogenic", "risk_factor",
                                 "drug_response", "protective", "other"]}
    stats = {"total": 0, "matched": 0, "pathogenic": 0, "likely_pathogenic": 0}
    with open(clinvar_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats["total"] += 1
            pos_key = f"{row['chrom']}:{row['pos']}"
            if pos_key not in genome_by_position:
                continue
            stats["matched"] += 1
            ref, alt = row["ref"], row["alt"]
            if len(ref) != 1 or len(alt) != 1:   # skip indels
                continue
            user = genome_by_position[pos_key]
            gt   = user["genotype"]
            sig  = row["clinical_significance"].lower()
            if not (alt in gt) or gt == ref + ref:
                continue
            finding = {
                "rsid":                  user["rsid"],
                "gene":                  row.get("symbol", ""),
                "chrom":                 row["chrom"],
                "pos":                   row["pos"],
                "ref":                   ref,
                "alt":                   alt,
                "genotype":              gt,
                "is_homozygous":         gt == alt + alt,
                "clinical_significance": row["clinical_significance"],
                "gold_stars":            int(row["gold_stars"]) if row.get("gold_stars") else 0,
                "traits":                row.get("all_traits", ""),
                "inheritance":           row.get("inheritance_modes", ""),
                "review_status":         row.get("review_status", ""),
            }
            if "pathogenic" in sig and "likely" not in sig and "conflict" not in sig:
                findings["pathogenic"].append(finding); stats["pathogenic"] += 1
            elif "likely pathogenic" in sig:
                findings["likely_pathogenic"].append(finding); stats["likely_pathogenic"] += 1
            elif "risk factor" in sig:
                findings["risk_factor"].append(finding)
            elif "drug response" in sig:
                findings["drug_response"].append(finding)
            elif "protective" in sig:
                findings["protective"].append(finding)
            elif "association" in sig or "affects" in sig:
                findings["other"].append(finding)
    return findings, stats


def find_drug_interactions(genome, pharmgkb):
    interactions = []
    for rsid, info in pharmgkb.items():
        if rsid not in genome:
            continue
        genotype = genome[rsid]["genotype"]
        annotation = info["genotypes"].get(genotype) or info["genotypes"].get(genotype[::-1])
        if annotation:
            interactions.append({
                "rsid": rsid, "gene": info["gene"], "drugs": info["drugs"],
                "genotype": genotype, "level": info["level"],
                "category": info["category"], "annotation": annotation,
            })
    return interactions


def zygosity_status(finding):
    inheritance = finding["inheritance"].lower()
    if finding["is_homozygous"]:
        return "AFFECTED (homozygous)"
    if "recessive" in inheritance:
        return "CARRIER (heterozygous, recessive)"
    if "dominant" in inheritance:
        return "AFFECTED (heterozygous, dominant)"
    return "HETEROZYGOUS (inheritance unclear)"


# ── Section 1 ──────────────────────────────────────────────────────────────

def plot_chromosome_distribution(genome_by_rsid):
    counts_map = Counter(v["chromosome"] for v in genome_by_rsid.values())
    chroms = [c for c in CHROM_ORDER if c in counts_map]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(chroms, [counts_map[c] for c in chroms], color="steelblue", edgecolor="white", linewidth=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _fmt(ax, "SNPs per chromosome in your 23andMe data", xlabel="Chromosome", ylabel="SNPs genotyped")
    plt.tight_layout(); plt.show()


# ── Section 3 ──────────────────────────────────────────────────────────────

def plot_curated_impact(findings):
    mag_counts = Counter(f["magnitude"] for f in findings)
    max_mag    = max(mag_counts) if mag_counts else 4
    labels_map = {0: "Normal", 1: "Low", 2: "Moderate", 3: "High", 4: "Critical", 5: "Critical+", 6: "Critical++"}
    buckets    = list(range(max_mag + 1))
    values     = [mag_counts.get(i, 0) for i in buckets]
    colors     = [MAG_COLORS[min(i, len(MAG_COLORS) - 1)] for i in buckets]
    labels     = [f"{labels_map.get(i, str(i))} ({i})" for i in buckets]
    fig, ax    = plt.subplots(figsize=(max(8, len(buckets) * 1.4), 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.set_ylim(0, max(values) * 1.25 if values else 10)
    _fmt(ax, "Curated SNP findings by impact level", ylabel="Number of findings")
    plt.tight_layout(); plt.show()


def print_high_impact_findings(findings, min_magnitude=2):
    import pandas as pd
    from IPython.display import display

    high = sorted([f for f in findings if f["magnitude"] >= min_magnitude],
                  key=lambda x: (-x["magnitude"], x["category"]))
    if not high:
        print(f"No findings with magnitude ≥ {min_magnitude}."); return

    MAG_LABELS = {0: "○ normal", 1: "● low", 2: "●● moderate",
                  3: "●●● high", 4: "●●●● critical", 5: "●●●●! critical+", 6: "●●●●!! critical++"}
    MAG_BG = {
        0: "#e8f5e9", 1: "#f9fbe7", 2: "#fff8e1",
        3: "#fff3e0", 4: "#fbe9e7", 5: "#f3e5f5", 6: "#ede7f6",
    }
    MAG_FG = {
        0: "#2e7d32", 1: "#558b2f", 2: "#e65100",
        3: "#bf360c", 4: "#b71c1c", 5: "#6a1b9a", 6: "#4527a0",
    }

    rows = [{
        "Impact":      MAG_LABELS.get(f["magnitude"], str(f["magnitude"])),
        "Category":    f["category"],
        "Gene":        f["gene"],
        "rsID":        f["rsid"],
        "Genotype":    f["genotype"],
        "Description": f["description"],
        "_mag":        f["magnitude"],
    } for f in high]

    df = pd.DataFrame(rows)

    def style_row(row):
        mag  = row["_mag"]
        bg   = MAG_BG.get(mag, "#ffffff")
        fg   = MAG_FG.get(mag, "#000000")
        base = f"background-color: {bg}; color: {fg};"
        bold = base + " font-weight: bold;"
        return [bold if col in ("Impact", "Gene") else base for col in row.index]

    styled = (
        df.style
        .apply(style_row, axis=1)
        .set_properties(**_CELL)
        .set_table_styles([{"selector": "th", "props": _TH},
                           {"selector": "td", "props": _TD}])
        .hide(axis="index")
        .hide(subset=["_mag"], axis="columns")
    )
    display(styled)


# ── Section 4 ──────────────────────────────────────────────────────────────

def plot_clinvar_categories(clinvar_findings):
    labels = {
        "pathogenic": "Pathogenic", "likely_pathogenic": "Likely Pathogenic",
        "risk_factor": "Risk Factor", "drug_response": "Drug Response",
        "protective": "Protective", "other": "Other association",
    }
    keys   = list(labels.keys())
    values = [len(clinvar_findings[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh([labels[k] for k in keys], values,
                   color=[CAT_COLORS[k] for k in keys], edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=11)
    ax.set_xlim(0, max(values) * 1.18 if any(values) else 10)
    _fmt(ax, "ClinVar findings in your genome", xlabel="Number of variants found", grid_axis="x")
    plt.tight_layout(); plt.show()


# ── Section 5 ──────────────────────────────────────────────────────────────

def plot_pharmgkb_levels(drug_interactions):
    levels = ["1A", "1B", "2A", "2B", "3", "4"]
    counts = [sum(1 for d in drug_interactions if d["level"] == l) for l in levels]
    desc   = ["Clinical guideline", "Clinical annotation", "Moderate evidence",
              "Moderate (preliminary)", "Low evidence", "Preliminary"]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([f"{l}\n{d}" for l, d in zip(levels, desc)], counts,
                  color=LEVEL_COLORS, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.set_ylim(0, max(counts) * 1.25 if any(counts) else 10)
    _fmt(ax, "PharmGKB drug interactions by evidence level", ylabel="Number of annotations")
    plt.tight_layout(); plt.show()


# ── Section 6 ──────────────────────────────────────────────────────────────

def plot_summary_dashboard(genome_by_rsid, curated_findings, clinvar_findings, drug_interactions):
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Genome at a glance (text panel)
    ax0 = fig.add_subplot(gs[0, 0]); ax0.axis("off")
    n_path = len(clinvar_findings["pathogenic"]) + len(clinvar_findings["likely_pathogenic"])
    lines  = [
        ("Total SNPs genotyped",       f"{len(genome_by_rsid):,}"),
        ("Curated SNPs analysed",       f"{len(curated_findings)}"),
        ("High-impact curated (≥3)",    f"{sum(1 for f in curated_findings if f['magnitude'] >= 3)}"),
        ("ClinVar pathogenic/likely",    f"{n_path}"),
        ("Drug interactions (all)",      f"{len(drug_interactions)}"),
        ("Drug interactions (level 1A)", f"{sum(1 for d in drug_interactions if d['level'] == '1A')}"),
    ]
    ax0.text(0.5, 1.02, "Genome at a glance", ha="center", va="top",
             fontsize=12, fontweight="bold", transform=ax0.transAxes)
    y = 0.92
    for label, val in lines:
        ax0.text(0.05, y, label, fontsize=10, transform=ax0.transAxes, va="top")
        ax0.text(0.97, y, val, fontsize=10, transform=ax0.transAxes, va="top",
                 ha="right", fontweight="bold", color="#1565c0")
        y -= 0.15
    ax0.axhline(y=0.07, color="#ddd", linewidth=0.8, xmin=0, xmax=1)

    # Curated impact distribution
    ax1 = fig.add_subplot(gs[0, 1])
    mag_counts = Counter(f["magnitude"] for f in curated_findings)
    vals = [mag_counts.get(i, 0) for i in range(5)]
    bars = ax1.bar(["0","1","2","3","4"], vals, color=MAG_COLORS, edgecolor="white")
    ax1.bar_label(bars, padding=2, fontsize=9)
    ax1.set_ylim(0, max(vals) * 1.3 if vals else 5)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Curated findings\nby impact level", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Magnitude (0–4)", fontsize=9)

    # ClinVar categories
    ax2 = fig.add_subplot(gs[0, 2])
    cat_keys   = ["pathogenic","likely_pathogenic","risk_factor","drug_response","protective","other"]
    cat_labels = ["Pathogenic","Likely Path.","Risk Factor","Drug Resp.","Protective","Other"]
    cat_vals   = [len(clinvar_findings[k]) for k in cat_keys]
    bars2 = ax2.barh(cat_labels[::-1], cat_vals[::-1],
                     color=[CAT_COLORS[k] for k in cat_keys][::-1], edgecolor="white")
    ax2.bar_label(bars2, padding=3, fontsize=9)
    ax2.set_xlim(0, max(cat_vals) * 1.25 if any(cat_vals) else 5)
    ax2.grid(axis="x", alpha=0.3)
    ax2.set_title("ClinVar findings\nby category", fontsize=11, fontweight="bold")

    # Het / hom donut
    ax3 = fig.add_subplot(gs[1, 0])
    het = sum(1 for v in genome_by_rsid.values()
              if len(v["genotype"]) == 2 and v["genotype"][0] != v["genotype"][1])
    hom = sum(1 for v in genome_by_rsid.values()
              if len(v["genotype"]) == 2 and v["genotype"][0] == v["genotype"][1])
    wedges, _ = ax3.pie([hom, het], colors=["#42a5f5","#ff7043"],
                        startangle=90, wedgeprops=dict(width=0.5, edgecolor="white"))
    pct = het / (het + hom) * 100 if (het + hom) else 0
    ax3.text(0, 0, f"{pct:.1f}%\nhet", ha="center", va="center", fontsize=12, fontweight="bold")
    ax3.legend(wedges, [f"Hom ({hom:,})", f"Het ({het:,})"],
               loc="lower center", fontsize=9, frameon=False, ncol=2)
    ax3.set_title("Genotype\nzygosity", fontsize=11, fontweight="bold")

    # PharmGKB evidence levels
    ax4 = fig.add_subplot(gs[1, 1])
    levels    = ["1A","1B","2A","2B","3","4"]
    lv_counts = [sum(1 for d in drug_interactions if d["level"] == l) for l in levels]
    bars4 = ax4.bar(levels, lv_counts, color=LEVEL_COLORS, edgecolor="white")
    ax4.bar_label(bars4, padding=2, fontsize=9)
    ax4.set_ylim(0, max(lv_counts) * 1.3 if lv_counts else 5)
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_title("PharmGKB drug interactions\nby evidence level", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Evidence level", fontsize=9)

    # Top ClinVar genes
    ax5 = fig.add_subplot(gs[1, 2])
    gene_counts = Counter(
        f["gene"] for flist in clinvar_findings.values()
        for f in flist if f["gene"]
    )
    top = gene_counts.most_common(10)
    if top:
        genes5, cnts5 = zip(*top)
        bars5 = ax5.barh(list(genes5)[::-1], list(cnts5)[::-1], color="#5c6bc0", edgecolor="white")
        ax5.bar_label(bars5, padding=3, fontsize=9)
        ax5.set_xlim(0, max(cnts5) * 1.25)
    ax5.grid(axis="x", alpha=0.3)
    ax5.set_title("Top 10 genes\n(ClinVar findings)", fontsize=11, fontweight="bold")

    fig.suptitle("Genetic Analysis — Summary Dashboard", fontsize=15, fontweight="bold", y=1.01)
    plt.show()
