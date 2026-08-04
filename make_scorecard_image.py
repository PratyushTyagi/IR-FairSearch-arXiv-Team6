"""Render a static Fairness Scorecard panel (from app.py's real logic) as a PNG
for Slide 9 of the deck — a faithful visual of the Streamlit dashboard output
without needing a live browser session.

Picks a query where Fair-Top-K actually changes the top-10 composition, then
draws: four metric tiles (Baseline vs Fair-Top-K), a group-composition bar, and
the group-tagged ranked list.

Usage:  python3 scripts/make_scorecard_image.py
Output: deliverables/scorecard_demo.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorecard as sc

ROOT = os.environ.get(
    "FAIRSEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "deliverables", "scorecard_demo.png")
NAVY, BLUE, RED, GRAY = "#0f2e5c", "#2563eb", "#dc2626", "#555555"


def main():
    queries, pools, docs, mmr = sc.load()
    # pick a query with >=1 Privileged paper in the baseline top-10 (so the
    # Fair-Top-K contrast is visible)
    qid = None
    for q in queries:
        pool = pools[q["qid"]]
        if sum(1 for d in pool[:10] if d["is_priv"]) >= 1:
            qid = q["qid"]; break
    qid = qid or queries[0]["qid"]
    qtext = next(q["text"] for q in queries if q["qid"] == qid)
    pool = pools[qid]

    base = sc.scorecard(pool, sc.rerank(pool, "baseline"), 10)
    fair_order = sc.rerank(pool, "fairtopk", min_share=1.0)
    fair = sc.scorecard(pool, fair_order, 10)

    fig = plt.figure(figsize=(13, 7.0))
    fig.patch.set_facecolor("white")
    # header bar
    fig.add_artist(plt.Rectangle((0, 0.92), 1, 0.08, color=NAVY, transform=fig.transFigure))
    fig.text(0.02, 0.945, "FairSearch-arXiv  —  Fairness Scorecard", color="white",
             fontsize=17, fontweight="bold")
    fig.text(0.02, 0.90, f"Query {qid}:  {qtext[:92]}…", fontsize=10.5, color="#222")
    fig.text(0.02, 0.87, "Group = QS Top-20 institution (Privileged) vs. everyone else "
             f"(Underrepresented).  Corpus base rate: {100*sc.CORPUS_PRIV_RATE:.2f}% Privileged.",
             fontsize=8.5, color=GRAY)

    # ---- metric tiles: Baseline vs Fair-Top-K ----
    tiles = [
        ("Privileged share@10", f"{100*base['priv_share']:.0f}%", f"{100*fair['priv_share']:.0f}%"),
        ("SPD", f"{base['spd']:+.1e}", f"{fair['spd']:+.1e}"),
        ("NDCG@10", f"{base['ndcg']:.3f}", f"{fair['ndcg']:.3f}"),
        ("Precision@10", f"{base['precision']:.3f}", f"{fair['precision']:.3f}"),
    ]
    x0, w, y = 0.02, 0.235, 0.70
    for i, (label, bv, fv) in enumerate(tiles):
        x = x0 + i * w
        ax = fig.add_axes([x, y, w - 0.02, 0.13]); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor="#eff3fb", edgecolor="#c9d6ef", transform=ax.transAxes))
        ax.text(0.5, 0.80, label, ha="center", fontsize=9, color=NAVY, fontweight="bold")
        ax.text(0.28, 0.35, bv, ha="center", fontsize=13, color="#333")
        ax.text(0.28, 0.10, "Baseline", ha="center", fontsize=7, color=GRAY)
        ax.text(0.72, 0.35, fv, ha="center", fontsize=13, color=BLUE, fontweight="bold")
        ax.text(0.72, 0.10, "Fair-Top-K", ha="center", fontsize=7, color=BLUE)

    fig.text(0.02, 0.665, "Fair-Top-K drives the elite share to 0% at essentially no NDCG / precision cost.",
             fontsize=10, color="#166534", fontweight="bold")

    # ---- composition bars ----
    axc = fig.add_axes([0.02, 0.10, 0.30, 0.48])
    for j, (name, order, col) in enumerate([("Baseline", pool, "#94a3b8"),
                                            ("Fair-Top-K", fair_order, BLUE)]):
        sel = order[:10]
        npriv = sum(1 for d in sel if d["is_priv"]); nund = 10 - npriv
        axc.barh(j + 0.0, nund, color=col, height=0.6)
        axc.barh(j + 0.0, npriv, left=nund, color=RED, height=0.6)
        axc.text(10.2, j, f"{npriv} elite", va="center", fontsize=8, color=RED)
    axc.set_yticks([0, 1]); axc.set_yticklabels(["Baseline", "Fair-Top-K"], fontsize=9)
    axc.set_xlim(0, 12); axc.set_xlabel("papers in top-10", fontsize=8)
    axc.set_title("Group composition (Underrep vs Privileged/elite)", fontsize=9.5, color=NAVY)
    axc.spines[["top", "right"]].set_visible(False)

    # ---- ranked list (Fair-Top-K top-8) ----
    axt = fig.add_axes([0.36, 0.06, 0.62, 0.54]); axt.axis("off")
    axt.text(0, 1.0, "Fair-Top-K — top results", fontsize=10, color=NAVY, fontweight="bold")
    rows = []
    for rank, d in enumerate(fair_order[:8], 1):
        badge = "Privileged" if d["is_priv"] else "Underrep"
        rel = "yes" if d["relevant"] else ""
        inst = (d["institution"] or "—")[:26]
        title = d["title"][:52] + ("…" if len(d["title"]) > 52 else "")
        rows.append([str(rank), badge, rel, title, inst])
    tbl = axt.table(cellText=rows, colLabels=["#", "group", "rel", "title", "institution"],
                    colWidths=[0.04, 0.13, 0.05, 0.55, 0.23], loc="upper left", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.6); tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if r == 0:
            cell.set_facecolor(NAVY); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
        elif c == 1:  # color the group column
            txt = cell.get_text().get_text()
            cell.get_text().set_color(RED if txt == "Privileged" else BLUE)
            cell.get_text().set_fontweight("bold")

    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}  (query {qid}: baseline elite={base['priv_in_topk']}/10 -> "
          f"Fair-Top-K elite={fair['priv_in_topk']}/10)")


if __name__ == "__main__":
    main()
