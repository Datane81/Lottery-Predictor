"""
Number Predictor (1-59) - statistical analysis app
---------------------------------------------------
Install:  pip install streamlit pandas numpy openpyxl
Run:      streamlit run number_predictor.py

Models: Frequency, Recency-weighted frequency, Overdue (gap ratio), Common pairs,
        Bayesian (discounted Beta-Binomial), Markov chain, Gap-hazard.
They are combined into a weighted ensemble. A walk-forward backtest shows whether
any model actually beats random guessing on YOUR data.
"""
import math
import random
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

MAX_N = 59
DB_PATH = Path(__file__).with_name("numbers.db")

# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------
def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS draws("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, draw_date TEXT, numbers TEXT NOT NULL)"
        )


def load_draws():
    with db() as c:
        rows = c.execute("SELECT id, draw_date, numbers FROM draws ORDER BY id").fetchall()
    return [(r[0], r[1], [int(x) for x in r[2].split(",")]) for r in rows]


def insert_draws(items, replace=False):
    with db() as c:
        if replace:
            c.execute("DELETE FROM draws")
        c.executemany(
            "INSERT INTO draws(draw_date, numbers) VALUES (?, ?)",
            [(d, ",".join(map(str, n))) for d, n in items],
        )


def delete_last():
    with db() as c:
        c.execute("DELETE FROM draws WHERE id = (SELECT MAX(id) FROM draws)")


def clear_all():
    with db() as c:
        c.execute("DELETE FROM draws")


# ----------------------------------------------------------------------------
# Parsing uploaded data
# ----------------------------------------------------------------------------
DATE_RE = re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")


def parse_text(text, per_draw=3, newest_first=False):
    out = []
    for line in text.splitlines():
        m = DATE_RE.search(line)
        date = m.group(0) if m else None
        rest = TIME_RE.sub(" ", DATE_RE.sub(" ", line))
        nums = [int(x) for x in re.findall(r"\d+", rest)]
        nums = list(dict.fromkeys(n for n in nums if 1 <= n <= MAX_N))
        if per_draw > 0:
            if len(nums) < per_draw:
                continue
            nums = nums[:per_draw]
        elif len(nums) < 2:
            continue
        out.append((date, nums))
    if newest_first:
        out.reverse()
    return out


def file_to_text(upload):
    name = upload.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload, header=None).to_csv(index=False, header=False)
    return upload.getvalue().decode("utf-8", errors="ignore")


def to_matrix(draws):
    H = np.zeros((len(draws), MAX_N))
    for i, (_, _, nums) in enumerate(draws):
        H[i, [n - 1 for n in nums]] = 1
    return H


# ----------------------------------------------------------------------------
# Mathematical models  (each returns a raw score vector of length 59)
# ----------------------------------------------------------------------------
def z(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def m_frequency(H, **_):
    return H.sum(0)


def m_recent(H, half_life=30, **_):
    N = len(H)
    w = 0.5 ** ((N - 1 - np.arange(N)) / half_life)
    return w @ H


def current_gaps(H):
    N = len(H)
    seen = H.sum(0)
    last_idx = (H * np.arange(N)[:, None]).max(0)
    return np.where(seen > 0, N - 1 - last_idx, N)


def m_overdue(H, **_):
    N = len(H)
    avg_gap = N / np.maximum(H.sum(0), 1)
    return current_gaps(H) / avg_gap  # >1 means "later than usual"


def pair_matrix(H):
    C = H.T @ H
    np.fill_diagonal(C, 0)
    return C


def m_pairs(H, top_k=25, **_):
    C = pair_matrix(H)
    iu = np.triu_indices(MAX_N, 1)
    vals = C[iu]
    idx = np.argsort(vals)[-min(top_k, len(vals)):]
    part = np.zeros(MAX_N)
    for i in idx:
        part[iu[0][i]] += vals[i]
        part[iu[1][i]] += vals[i]
    last = np.flatnonzero(H[-1])
    aff = C[last].sum(0) if len(last) else np.zeros(MAX_N)
    return z(part) + z(aff)


def bayes_posterior(H, prior_strength=20, decay=0.99):
    N = len(H)
    w = decay ** (N - 1 - np.arange(N))
    hits, trials = w @ H, w.sum()
    p0 = max(H.sum() / (N * MAX_N), 1e-9)
    a, b = prior_strength * p0, prior_strength * (1 - p0)
    al, be = a + hits, b + trials - hits
    mean = al / (al + be)
    sd = np.sqrt(al * be / ((al + be) ** 2 * (al + be + 1)))
    return mean, sd


def m_bayes(H, prior_strength=20, decay=0.99, **_):
    return bayes_posterior(H, prior_strength, decay)[0]


def m_markov(H, alpha=0.5, **_):
    if len(H) < 2:
        return np.zeros(MAX_N)
    T = H[:-1].T @ H[1:] + alpha
    T = T / T.sum(1, keepdims=True)
    last = np.flatnonzero(H[-1])
    return T[last].mean(0) if len(last) else np.zeros(MAX_N)


def m_hazard(H, smooth=2.0, **_):
    N = len(H)
    base = H.sum() / (N * MAX_N)
    out = np.zeros(MAX_N)
    for n in range(MAX_N):
        pos = np.flatnonzero(H[:, n])
        cur = N - 1 - pos[-1] if len(pos) else N
        gaps = np.diff(pos) if len(pos) > 1 else np.array([])
        out[n] = ((gaps == cur + 1).sum() + smooth * base) / ((gaps > cur).sum() + smooth)
    return out


MODELS = {
    "Frequency": (m_frequency, "How often each number appeared overall"),
    "Recent trend": (m_recent, "Frequency with recent draws weighted more"),
    "Overdue": (m_overdue, "Current gap vs the number's own average gap"),
    "Common pairs": (m_pairs, "Membership in top pairs + affinity with the last draw"),
    "Bayesian": (m_bayes, "Discounted Beta-Binomial posterior probability"),
    "Markov chain": (m_markov, "Transition probabilities from the last draw"),
    "Gap hazard": (m_hazard, "Empirical chance of appearing given the current gap"),
}


def all_z(H, params):
    return {name: z(fn(H, **params)) for name, (fn, _) in MODELS.items()}


def ensemble(zs, weights):
    tot = sum(weights.values()) or 1
    return sum(weights[n] * zs[n] for n in zs) / tot


def chi_square(H):
    c = H.sum(0)
    E = c.sum() / MAX_N
    chi = ((c - E) ** 2 / E).sum()
    df = MAX_N - 1
    zz = ((chi / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return chi, 0.5 * math.erfc(zz / math.sqrt(2))


def backtest(H, weights, params, n_test, picks=3, min_hist=30):
    N = len(H)
    start = max(min_hist, N - n_test)
    res = {n: [] for n in list(MODELS) + ["ENSEMBLE"]}
    base = []
    for t in range(start, N):
        zs = all_z(H[:t], params)
        scores = dict(zs)
        scores["ENSEMBLE"] = ensemble(zs, weights)
        for name, s in scores.items():
            res[name].append(H[t, np.argsort(s)[-picks:]].sum())
        base.append(picks * H[t].sum() / MAX_N)
    rb = float(np.mean(base))
    rows = [{"Model": k, "Avg hits": np.mean(v), "Random baseline": rb,
             "Edge": np.mean(v) - rb} for k, v in res.items()]
    return pd.DataFrame(rows), len(base)


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def show_balls(nums):
    html = "".join(
        "<span style='display:inline-flex;width:88px;height:88px;border-radius:50%;"
        "background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;font-size:38px;"
        "font-weight:700;align-items:center;justify-content:center;margin:8px;"
        f"box-shadow:0 4px 12px rgba(0,0,0,.3)'>{n}</span>"
        for n in nums
    )
    st.markdown(f"<div style='text-align:center'>{html}</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Number Predictor", page_icon="🔮", layout="wide")
    init_db()
    draws = load_draws()
    H = to_matrix(draws) if draws else np.zeros((0, MAX_N))

    st.title("🔮 Number Predictor (1–59)")
    st.caption("Frequency · Overdue · Pairs · Bayesian · Markov chain · Gap hazard · Ensemble")

    # ---- Sidebar: model weights ----
    st.sidebar.header("⚙️ Model weights")
    st.sidebar.caption("0 = ignore a model, 5 = strongest influence.")
    weights = {n: st.sidebar.slider(n, 0.0, 5.0, 1.0, 0.5, help=d)
               for n, (_, d) in MODELS.items()}
    with st.sidebar.expander("Advanced parameters"):
        params = {
            "half_life": st.slider("Recent trend half-life (draws)", 5, 200, 30),
            "decay": st.slider("Bayesian memory decay", 0.90, 1.0, 0.99, 0.005),
            "prior_strength": st.slider("Bayesian prior strength", 1, 100, 20),
            "alpha": st.slider("Markov smoothing", 0.0, 5.0, 0.5, 0.1),
            "top_k": st.slider("Top pairs considered", 5, 100, 25),
        }
    st.sidebar.metric("Draws in database", len(draws))

    tab_data, tab_charts, tab_pred, tab_test = st.tabs(
        ["📥 Data", "📊 Charts", "🔮 Prediction", "🧪 Backtest"]
    )

    # ======================= DATA =======================
    with tab_data:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Upload your draw history")
            st.write("CSV, TXT or Excel — one draw per row, e.g. `2024-05-01, 7, 23, 41`. "
                     "A date column is optional.")
            up = st.file_uploader("Choose a file", type=["csv", "txt", "xlsx", "xls"])
            o1, o2, o3 = st.columns(3)
            per_draw = o1.number_input("Numbers per draw (0 = all)", 0, 20, 3)
            order = o2.radio("File order", ["Oldest first", "Newest first"])
            mode = o3.radio("Import mode", ["Add to database", "Replace database"])
            if up is not None:
                items = parse_text(file_to_text(up), int(per_draw), order == "Newest first")
                st.write(f"Found **{len(items)}** valid draws.")
                if items:
                    st.dataframe(pd.DataFrame(
                        [{"Date": d, "Numbers": ", ".join(map(str, n))} for d, n in items[:10]]),
                        hide_index=True)
                    if st.button("✅ Import into database", type="primary"):
                        insert_draws(items, replace=mode.startswith("Replace"))
                        st.success("Imported!")
                        st.rerun()
        with c2:
            st.subheader("Add a draw manually")
            dt = st.text_input("Date (optional)")
            cols = st.columns(3)
            vals = [cols[i].number_input(f"No. {i+1}", 1, MAX_N, i + 1, key=f"m{i}") for i in range(3)]
            if st.button("➕ Add draw"):
                if len(set(vals)) < 3:
                    st.error("The 3 numbers must be different.")
                else:
                    insert_draws([(dt or None, [int(v) for v in vals])])
                    st.rerun()
            st.divider()
            if st.button("🎲 Load 300 random demo draws"):
                insert_draws([(None, random.sample(range(1, MAX_N + 1), 3)) for _ in range(300)],
                             replace=True)
                st.rerun()

        if draws:
            st.subheader("Stored draws (latest first)")
            df = pd.DataFrame([{"#": i, "Date": d, "Numbers": ", ".join(map(str, n))}
                               for i, d, n in draws[::-1][:100]])
            st.dataframe(df, hide_index=True, use_container_width=True)
            b1, b2, b3 = st.columns(3)
            b1.download_button("⬇️ Export CSV", pd.DataFrame(
                [{"date": d, "numbers": " ".join(map(str, n))} for _, d, n in draws]
            ).to_csv(index=False), "draws.csv")
            if b2.button("↩️ Delete latest draw"):
                delete_last()
                st.rerun()
            if b3.checkbox("Unlock 'clear all'") and st.button("🗑️ Clear database"):
                clear_all()
                st.rerun()

    if len(draws) < 10:
        for t in (tab_charts, tab_pred, tab_test):
            with t:
                st.info("Add at least 10 draws in the **Data** tab to unlock this section.")
        return

    # ======================= CHARTS =======================
    with tab_charts:
        win = st.slider("Analyse the last N draws", 10, len(H), len(H))
        Hw = H[-win:]
        idx = list(range(1, MAX_N + 1))
        a, b = st.columns(2)
        with a:
            st.subheader("Frequency")
            st.bar_chart(pd.DataFrame({"Times drawn": Hw.sum(0)}, index=idx))
        with b:
            st.subheader("Most overdue (draws since last seen)")
            st.bar_chart(pd.DataFrame({"Draws since last seen": current_gaps(Hw)}, index=idx))
        c, d = st.columns(2)
        with c:
            st.subheader("Top 15 pairs")
            C = pair_matrix(Hw)
            iu = np.triu_indices(MAX_N, 1)
            order_ = np.argsort(C[iu])[::-1][:15]
            st.dataframe(pd.DataFrame({
                "Pair": [f"{iu[0][i]+1} & {iu[1][i]+1}" for i in order_],
                "Times together": [int(C[iu][i]) for i in order_]}), hide_index=True)
        with d:
            st.subheader("Is the data random?")
            chi, p = chi_square(Hw)
            st.metric("Chi-square p-value", f"{p:.3f}")
            st.caption("p > 0.05 → frequencies are consistent with pure chance "
                       "(no number is genuinely 'hot').")

    # ======================= PREDICTION =======================
    with tab_pred:
        zs = all_z(H, params)
        score = ensemble(zs, weights)
        top = np.argsort(score)[::-1]
        pick = sorted(int(n) + 1 for n in top[:3])
        st.subheader("Most likely next 3 numbers (ensemble)")
        show_balls(pick)
        st.caption(f"Based on {len(draws)} draws. Last draw: "
                   f"{', '.join(map(str, draws[-1][2]))}")
        st.subheader("Top 10 candidates and each model's vote")
        tbl = pd.DataFrame({m: zs[m][top[:10]].round(2) for m in zs},
                           index=[int(n) + 1 for n in top[:10]])
        tbl.insert(0, "ENSEMBLE SCORE", score[top[:10]].round(2))
        tbl.index.name = "Number"
        st.dataframe(tbl, use_container_width=True)
        st.markdown("**Each model's own top 3**")
        st.dataframe(pd.DataFrame({
            "Model": list(zs),
            "Top 3": [", ".join(str(int(n) + 1) for n in np.argsort(v)[::-1][:3]) for v in zs.values()]
        }), hide_index=True)
        mean, sd = bayes_posterior(H, params["prior_strength"], params["decay"])
        with st.expander("Bayesian posterior for the top 3"):
            st.dataframe(pd.DataFrame({
                "Number": pick,
                "P(appears in a draw)": [f"{mean[n-1]:.3%}" for n in pick],
                "± std": [f"{sd[n-1]:.3%}" for n in pick]}), hide_index=True)

    # ======================= BACKTEST =======================
    with tab_test:
        st.write("Replays history: for each past draw the models only see earlier draws, "
                 "predict 3 numbers, and are scored on how many hit.")
        n_test = st.slider("Number of past draws to test", 20, min(500, len(H) - 30) if len(H) > 50 else 20,
                           min(100, max(20, len(H) - 30)))
        if st.button("▶️ Run backtest", type="primary"):
            with st.spinner("Running…"):
                res, n = backtest(H, weights, params, n_test)
            st.write(f"Tested on **{n}** draws.")
            st.dataframe(res.round(4), hide_index=True, use_container_width=True)
            st.bar_chart(res.set_index("Model")[["Edge"]])
            st.caption("Edge = average hits minus what random picking would score. "
                       "Edges near 0 mean the model is no better than chance.")

    st.divider()
    st.caption("⚠️ If the numbers are truly random, each draw is independent and no model can "
               "predict it. This app finds statistical patterns and shows honestly whether they "
               "beat chance. It's for analysis/entertainment, not a guarantee.")


if __name__ == "__main__":
    main()
