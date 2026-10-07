"""
Simulasi Pendapatan Sewa Media Layanan Digital - Wi-Fi Publik (Captive Portal)
Jalankan:  streamlit run app.py

Model (accrual / kategori aktual):
  pendapatan bulan m = sum_{layanan, kategori} AP_aktif x sebaran_kategori x tarif
  - Jumlah AP per kategori tarif untuk tiap layanan diisi pengguna (kondisi dasar).
  - Pertumbuhan login, musiman, dan pengali what-if menggeser sebaran kategori
    (login tiap AP diasumsikan tersebar log-uniform di dalam rentang kategorinya).
"""
import math
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# DATA DASAR (sesuai data infrastruktur & skema fee)
# ----------------------------------------------------------------------------
TIERS = ["Basic", "Reguler", "Premium", "Ultimate", "Unlimited"]
DEFAULT_TARIF = [20_000, 30_000, 70_000, 120_000, 150_000]
TIER_RANGE = ["≤ 1.000", "> 1.000 – 3.000", "> 3.000 – 5.000", "> 5.000 – 10.000", "> 10.000"]
TIER_MIN_LOGIN = [0, 1_001, 3_001, 5_001, 10_001]  # batas bawah login sesi/bulan per kategori

# Batas (login sesi/bulan) untuk memodelkan sebaran di dalam kategori: Basic 100-1.000, ..., Unlimited 10.000-30.000
LOGIN_EDGES = [100, 1_000, 3_000, 5_000, 10_000, 30_000]

GROUPS = [
    {"name": "Telkom – Internet Publik", "ap": 30, "zona": "Publik"},
    {"name": "Telkom – Kecamatan Berdaya", "ap": 150, "zona": "Publik"},
    {"name": "Icon Plus – Internet Desa", "ap": 171, "zona": "Publik"},
    {"name": "Icon Plus – Internet Sekolah", "ap": 10, "zona": "Pendidikan"},
    {"name": "ION Network – Internet Sekolah", "ap": 99, "zona": "Pendidikan"},
    {"name": "ION Network – Internet Publik", "ap": 10, "zona": "Publik"},
]

# PRESET = ASUMSI ILUSTRATIF (bukan data riil). Urutan kolom: Basic, Reguler, Premium, Ultimate, Unlimited (%)
PRESETS = {
    "Pesimis": [
        [60, 25, 10, 5, 0],
        [70, 25, 5, 0, 0],
        [85, 12, 3, 0, 0],
        [50, 30, 15, 5, 0],
        [55, 30, 10, 5, 0],
        [40, 30, 20, 10, 0],
    ],
    "Realistis": [
        [30, 30, 20, 15, 5],
        [45, 35, 15, 5, 0],
        [60, 30, 8, 2, 0],
        [20, 30, 30, 15, 5],
        [25, 35, 25, 10, 5],
        [20, 30, 25, 15, 10],
    ],
    "Optimis": [
        [10, 20, 25, 30, 15],
        [25, 35, 25, 12, 3],
        [40, 35, 15, 8, 2],
        [10, 20, 30, 25, 15],
        [10, 25, 30, 25, 10],
        [5, 15, 25, 30, 25],
    ],
}

MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
MONTHS_LONG = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
               "September", "Oktober", "November", "Desember"]

# Contoh pola musiman (ILUSTRATIF) - pengali volume login per bulan kalender
SEASON_EXAMPLE = {
    "Publik":     [1.05, 0.95, 1.00, 1.05, 1.00, 1.10, 1.15, 1.05, 1.00, 1.00, 1.00, 1.10],
    "Pendidikan": [0.95, 1.00, 1.05, 1.05, 1.05, 0.70, 0.60, 1.05, 1.10, 1.10, 1.10, 0.80],
}


# ----------------------------------------------------------------------------
# FUNGSI HITUNG
# ----------------------------------------------------------------------------
def rp(x: float) -> str:
    """Format Rupiah ala Indonesia: Rp 1.234.567"""
    return "Rp " + f"{x:,.0f}".replace(",", ".")


def juta(x: float) -> str:
    return f"Rp {x / 1e6:,.1f} jt".replace(",", "X").replace(".", ",").replace("X", ".")


def redistribute(dist, f: float) -> np.ndarray:
    """
    Geser sebaran kategori bila volume login semua AP dikali f.
    Login di dalam tiap kategori diasumsikan log-uniform pada LOGIN_EDGES.
    """
    dist = np.asarray(dist, dtype=float)
    if abs(f - 1.0) < 1e-12:
        return dist.copy()
    src = np.log(LOGIN_EDGES)
    tgt = [-math.inf] + list(np.log([1_000, 3_000, 5_000, 10_000])) + [math.inf]
    lf = math.log(f)
    out = np.zeros(5)
    for i in range(5):
        lo, hi = src[i] + lf, src[i + 1] + lf
        width = hi - lo
        for j in range(5):
            overlap = max(0.0, min(hi, tgt[j + 1]) - max(lo, tgt[j]))
            out[j] += dist[i] * overlap / width
    return out


def counts_from_pct(total: int, pct) -> list:
    """Ubah persentase menjadi jumlah AP bulat yang totalnya tepat = total (largest remainder)."""
    raw = [total * p / 100 for p in pct]
    base = [int(math.floor(v)) for v in raw]
    rem = int(total - sum(base))
    order = sorted(range(len(raw)), key=lambda j: raw[j] - base[j], reverse=True)
    for j in order[:max(rem, 0)]:
        base[j] += 1
    return base


def month_labels(start_month: int, start_year: int):
    labels = []
    for m in range(12):
        idx = (start_month - 1 + m)
        year = start_year + idx // 12
        labels.append(f"{MONTHS_SHORT[idx % 12]} {year}")
    return labels


def simulate(groups_df: pd.DataFrame, tarif, p: dict) -> pd.DataFrame:
    """
    groups_df: kolom Layanan, Zona, AP, dan fraksi f0..f4 (jumlah = 1)
    p: parameter (start_month, start_year, active_start, ramp_full, active_override,
       uptime, growth, traffic_mult, season, tax, collection)
    Mengembalikan tabel panjang: idx, Bulan, Layanan, Zona, Kategori, AP, Pendapatan
    """
    labels = month_labels(p["start_month"], p["start_year"])
    rows = []
    for m in range(12):
        cal = (p["start_month"] - 1 + m) % 12
        if p.get("active_override") is not None:
            act = p["active_override"]
        elif p["ramp_full"] <= 1:
            act = 1.0
        else:
            act = p["active_start"] + (1 - p["active_start"]) * min(1.0, m / (p["ramp_full"] - 1))
        growth_f = (1 + p["growth"]) ** m
        for _, g in groups_df.iterrows():
            f = p["traffic_mult"] * growth_f * p["season"][g["Zona"]][cal]
            dist = redistribute([g[f"f{j}"] for j in range(5)], f)
            ap_act = g["AP"] * act * p["uptime"]
            for j, t in enumerate(TIERS):
                ap_j = ap_act * dist[j]
                rev = ap_j * tarif[j] * (1 - p["tax"]) * p["collection"]
                rows.append({"idx": m, "Bulan": labels[m], "Layanan": g["Layanan"], "Zona": g["Zona"],
                             "Kategori": t, "AP": ap_j, "Pendapatan": rev})
    return pd.DataFrame(rows)


def ap_needed(tm_gross: float, tarif_base: float, tarif_up: float, n_ap: float):
    """Jumlah AP di kategori 'up' bila sisa AP di kategori 'base' agar sewa bruto bulanan = tm_gross."""
    if tm_gross <= tarif_base * n_ap:
        return 0.0
    x = (tm_gross - tarif_base * n_ap) / (tarif_up - tarif_base)
    return x


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Simulasi Pendapatan Wi-Fi Publik", page_icon="📶", layout="wide")
    st.title("📶 Simulasi Pendapatan Sewa Media Layanan Digital")
    st.caption("Wi-Fi publik & captive portal · perhitungan berdasarkan kategori aktual (accrual) · "
               "preset yang tersedia adalah **asumsi ilustratif**, bukan data riil.")

    # ---------------- SIDEBAR ----------------
    sb = st.sidebar
    sb.header("Parameter Umum")
    target = sb.number_input("Target pendapatan setahun (Rp)", min_value=0, value=300_000_000, step=10_000_000)
    c1, c2 = sb.columns(2)
    start_month = c1.selectbox("Bulan mulai", list(range(1, 13)), index=0, format_func=lambda i: MONTHS_LONG[i - 1])
    start_year = c2.number_input("Tahun", min_value=2024, max_value=2100, value=2027, step=1)

    sb.subheader("Faktor realitas")
    active_start = sb.slider("AP aktif pada bulan pertama (%)", 0, 100, 100, 5,
                             help="Dipakai bersama 'bulan semua AP aktif'. Naik linear sampai 100%.")
    ramp_full = sb.slider("Bulan ketika seluruh AP sudah aktif", 1, 12, 1,
                          help="1 = seluruh AP aktif sejak bulan pertama.")
    uptime = sb.slider("AP beroperasi (uptime rata-rata, %)", 50, 100, 98, 1,
                       help="AP yang mati tidak ditagih (Pasal 6 ayat 7).")
    growth = sb.number_input("Pertumbuhan volume login per bulan (%)", min_value=-20.0, max_value=50.0,
                             value=0.0, step=0.5)
    traffic_mult = sb.slider("Pengali volume login (what-if)", 0.25, 3.0, 1.0, 0.05,
                             help="1,0 = sesuai sebaran yang Anda isi. 1,5 = login 50% lebih ramai.")
    collection = sb.slider("Tingkat penagihan terealisasi (%)", 0, 100, 100, 1)
    tax = sb.number_input("Pajak yang termasuk dalam tarif (%)", min_value=0.0, max_value=30.0, value=0.0, step=0.5,
                          help="Dikurangkan dari pendapatan bila tarif sudah termasuk pajak. Isi 0 bila tidak ada.")
    use_season = sb.checkbox("Pakai contoh pola musiman", value=False,
                             help="Sekolah turun saat libur semester, wisata naik saat libur. Dapat diedit di tab Input.")

    with sb.expander("Tarif per AP per bulan (dapat diubah)"):
        tarif = []
        for i, t in enumerate(TIERS):
            tarif.append(st.number_input(f"{t}  ({TIER_RANGE[i]} login/bln)", min_value=0,
                                         value=DEFAULT_TARIF[i], step=5_000, key=f"tarif_{t}"))

    tabs = st.tabs(["⚙️ Input Skenario", "📊 Hasil", "🎯 Target Solver", "🔍 Sensitivitas"])

    # ---------------- TAB 1: INPUT ----------------
    with tabs[0]:
        st.subheader("Jumlah AP per kategori tarif untuk tiap layanan")
        st.write("Isi **jumlah AP** pada tiap kategori. Jumlah AP setiap layanan otomatis sama dengan total kelima "
                 "kategori. Contoh: 30 AP = Basic 10, Reguler 5, Premium 5, Ultimate 5, Unlimited 5.")
        preset = st.radio("Isi awal (asumsi ilustratif)", ["Pesimis", "Realistis", "Optimis"],
                          index=1, horizontal=True,
                          help="Preset mengubah persentase asumsi menjadi jumlah AP sesuai jumlah AP tiap layanan. "
                               "Setelah itu seluruh angka bebas Anda ubah.")
        base = pd.DataFrame({
            "Layanan": [g["name"] for g in GROUPS],
            "Zona": [g["zona"] for g in GROUPS],
        })
        counts = [counts_from_pct(g["ap"], PRESETS[preset][i]) for i, g in enumerate(GROUPS)]
        for j, t in enumerate(TIERS):
            base[f"{t} (AP)"] = [counts[i][j] for i in range(len(GROUPS))]

        colcfg = {f"{t} (AP)": st.column_config.NumberColumn(min_value=0, step=1, format="%d") for t in TIERS}
        edited = st.data_editor(base, key=f"editor_ap_{preset}", hide_index=True,
                                disabled=["Layanan", "Zona"], column_config=colcfg)

        cnt = edited[[f"{t} (AP)" for t in TIERS]].astype(float).fillna(0.0).clip(lower=0.0)
        ap_sum = cnt.sum(axis=1)
        ref_ap = pd.Series([g["ap"] for g in GROUPS], index=edited.index, dtype=float)

        summary = pd.DataFrame({
            "Layanan": edited["Layanan"],
            "Jumlah AP (hasil input)": ap_sum.round().astype(int),
            "Data infrastruktur": ref_ap.astype(int),
            "Selisih": (ap_sum - ref_ap).round().astype(int),
        })
        st.dataframe(summary, hide_index=True)
        total_ap = float(ap_sum.sum())
        if (summary["Selisih"] != 0).any():
            st.warning(f"Jumlah AP hasil input ({total_ap:,.0f}) berbeda dari data infrastruktur "
                       f"({ref_ap.sum():,.0f}). Tidak masalah bila memang untuk simulasi penambahan/pengurangan AP."
                       .replace(",", "."))
        else:
            st.success(f"Jumlah AP tiap layanan sesuai data infrastruktur. Total {total_ap:,.0f} AP.".replace(",", "."))

        groups_df = edited[["Layanan", "Zona"]].copy()
        groups_df["AP"] = ap_sum.values
        denom = np.where(ap_sum.values > 0, ap_sum.values, 1.0)
        for j in range(5):
            groups_df[f"f{j}"] = np.where(ap_sum.values > 0, cnt.iloc[:, j].values / denom,
                                          1.0 if j == 0 else 0.0)

        # pola musiman
        if use_season:
            st.subheader("Pola musiman (pengali volume login per bulan kalender)")
            st.caption("1,0 = normal. Contoh ilustratif; silakan sesuaikan dengan kondisi lokal "
                       "(libur sekolah, Lebaran, libur panjang, event wisata).")
            sdf = pd.DataFrame(SEASON_EXAMPLE, index=MONTHS_SHORT).T
            sdf = st.data_editor(sdf, key="season_editor")
            season = {z: [float(v) for v in sdf.loc[z].values] for z in ["Publik", "Pendidikan"]}
        else:
            season = {"Publik": [1.0] * 12, "Pendidikan": [1.0] * 12}

    params = {
        "start_month": int(start_month), "start_year": int(start_year),
        "active_start": active_start / 100, "ramp_full": int(ramp_full), "active_override": None,
        "uptime": uptime / 100, "growth": growth / 100, "traffic_mult": float(traffic_mult),
        "season": season, "tax": tax / 100, "collection": collection / 100,
    }
    df = simulate(groups_df, tarif, params)

    monthly = (df.groupby(["idx", "Bulan"], sort=True)
                 .agg(AP_aktif=("AP", "sum"), Pendapatan=("Pendapatan", "sum")).reset_index())
    monthly["Kumulatif"] = monthly["Pendapatan"].cumsum()
    monthly["% Target (kumulatif)"] = monthly["Kumulatif"] / target * 100 if target > 0 else 0.0
    tier_pivot = (df.pivot_table(index="idx", columns="Kategori", values="AP", aggfunc="sum")[TIERS])
    total = float(monthly["Pendapatan"].sum())
    gap = total - target
    reached = monthly.loc[monthly["Kumulatif"] >= target, "Bulan"]
    ap_months = float(monthly["AP_aktif"].sum())

    # ---------------- TAB 2: HASIL ----------------
    with tabs[1]:
        st.subheader("Ringkasan")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total pendapatan setahun", juta(total))
        k2.metric("Capaian target", f"{(total / target * 100) if target else 0:.1f}%")
        k3.metric("Rata-rata per bulan", juta(total / 12))
        k4.metric("Selisih terhadap target", juta(gap),
                  delta=("+" if gap >= 0 else "-") + juta(abs(gap)), delta_color="normal")
        if target > 0:
            if len(reached):
                st.success(f"Target kumulatif tercapai pada **{reached.iloc[0]}**.")
            else:
                st.error(f"Target belum tercapai dalam 12 bulan. Kekurangan {rp(-gap)}.")

        # grafik 1: pendapatan per bulan vs target bulanan
        fig = go.Figure()
        fig.add_bar(x=monthly["Bulan"], y=monthly["Pendapatan"] / 1e6, name="Pendapatan",
                    marker_color="#4A90D9")
        fig.add_scatter(x=monthly["Bulan"], y=[target / 12 / 1e6] * 12, mode="lines",
                        name="Target rata-rata/bulan", line=dict(color="#D64545", dash="dash"))
        fig.update_layout(title="Pendapatan per bulan (Rp juta)", yaxis_title="Rp juta",
                          template="plotly_white", height=380, legend=dict(orientation="h"))
        st.plotly_chart(fig)

        # grafik 2: kumulatif vs target
        fig2 = go.Figure()
        fig2.add_scatter(x=monthly["Bulan"], y=monthly["Kumulatif"] / 1e6, mode="lines+markers",
                         name="Kumulatif", line=dict(color="#2E7D32"))
        fig2.add_scatter(x=monthly["Bulan"], y=[target / 1e6] * 12, mode="lines", name="Target setahun",
                         line=dict(color="#D64545", dash="dash"))
        fig2.update_layout(title="Pendapatan kumulatif vs target (Rp juta)", yaxis_title="Rp juta",
                           template="plotly_white", height=380, legend=dict(orientation="h"))
        st.plotly_chart(fig2)

        # donut: per kategori & per layanan
        d1, d2 = st.columns(2)
        by_tier = df.groupby("Kategori")["Pendapatan"].sum().reindex(TIERS)
        figp = go.Figure(go.Pie(labels=by_tier.index, values=by_tier.values, hole=0.5, sort=False))
        figp.update_layout(title="Kontribusi per kategori tarif", height=360, template="plotly_white")
        d1.plotly_chart(figp)
        by_grp = df.groupby("Layanan")["Pendapatan"].sum().reindex([g["name"] for g in GROUPS])
        figg = go.Figure(go.Pie(labels=by_grp.index, values=by_grp.values, hole=0.5, sort=False))
        figg.update_layout(title="Kontribusi per layanan / pengelola", height=360, template="plotly_white")
        d2.plotly_chart(figg)

        # stacked bar per layanan
        piv_g = df.pivot_table(index=["idx", "Bulan"], columns="Layanan", values="Pendapatan", aggfunc="sum")
        piv_g = piv_g.reset_index()
        figs = go.Figure()
        for g in [g["name"] for g in GROUPS]:
            figs.add_bar(x=piv_g["Bulan"], y=piv_g[g] / 1e6, name=g)
        figs.update_layout(barmode="stack", title="Pendapatan per bulan per layanan (Rp juta)",
                           yaxis_title="Rp juta", template="plotly_white", height=420,
                           legend=dict(orientation="h"))
        st.plotly_chart(figs)

        # stacked: komposisi AP per kategori per bulan
        figt = go.Figure()
        for t in TIERS:
            figt.add_bar(x=monthly["Bulan"], y=tier_pivot[t].values, name=t)
        figt.update_layout(barmode="stack", title="Komposisi AP aktif per kategori tarif",
                           yaxis_title="Jumlah AP (ekspektasi)", template="plotly_white", height=380,
                           legend=dict(orientation="h"))
        st.plotly_chart(figt)

        # tabel bulanan
        st.subheader("Tabel pendapatan per bulan")
        tbl = monthly[["Bulan", "AP_aktif"]].copy()
        for t in TIERS:
            tbl[f"AP {t}"] = tier_pivot[t].values
        tbl["Pendapatan"] = monthly["Pendapatan"]
        tbl["Kumulatif"] = monthly["Kumulatif"]
        tbl["% Target kumulatif"] = monthly["% Target (kumulatif)"]
        show = tbl.copy()
        for c in ["AP_aktif"] + [f"AP {t}" for t in TIERS]:
            show[c] = show[c].map(lambda v: f"{v:,.1f}".replace(",", "X").replace(".", ",").replace("X", "."))
        show["Pendapatan"] = show["Pendapatan"].map(rp)
        show["Kumulatif"] = show["Kumulatif"].map(rp)
        show["% Target kumulatif"] = show["% Target kumulatif"].map(lambda v: f"{v:.1f}%")
        show = show.rename(columns={"AP_aktif": "AP aktif"})
        st.dataframe(show, hide_index=True)
        st.caption("Jumlah AP per kategori adalah nilai ekspektasi (bisa berupa pecahan) hasil sebaran persentase.")

        st.subheader("Rekap per layanan")
        rg = df.groupby("Layanan").agg(Pendapatan=("Pendapatan", "sum")).reindex([g["name"] for g in GROUPS])
        rg["Jumlah AP"] = groups_df.set_index("Layanan")["AP"]
        rg["Pendapatan/AP/bulan"] = rg["Pendapatan"] / rg["Jumlah AP"].replace(0, np.nan) / 12
        rg["Kontribusi"] = rg["Pendapatan"] / total * 100 if total > 0 else 0
        rg_show = rg.copy().reset_index()
        rg_show["Pendapatan"] = rg_show["Pendapatan"].map(rp)
        rg_show["Pendapatan/AP/bulan"] = rg_show["Pendapatan/AP/bulan"].map(lambda v: rp(v) if pd.notna(v) else "-")
        rg_show["Kontribusi"] = rg_show["Kontribusi"].map(lambda v: f"{v:.1f}%")
        rg_show["Jumlah AP"] = rg_show["Jumlah AP"].map(lambda v: f"{v:,.0f}".replace(",", "."))
        st.dataframe(rg_show, hide_index=True)

        # ekspor Excel
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            pd.DataFrame({
                "Parameter": ["Target setahun (Rp)", "Bulan mulai", "AP aktif bulan pertama (%)",
                              "Bulan semua AP aktif", "Uptime (%)", "Pertumbuhan login/bulan (%)",
                              "Pengali volume login", "Tingkat penagihan (%)", "Pajak dalam tarif (%)",
                              "Total pendapatan (Rp)", "Capaian target (%)"] +
                             [f"Tarif {t} (Rp)" for t in TIERS],
                "Nilai": [target, f"{MONTHS_LONG[start_month - 1]} {start_year}", active_start, ramp_full,
                          uptime, growth, traffic_mult, collection, tax, round(total),
                          round(total / target * 100, 2) if target else 0] + list(tarif),
            }).to_excel(xw, sheet_name="Parameter", index=False)
            edited.to_excel(xw, sheet_name="Input AP", index=False)
            tbl.to_excel(xw, sheet_name="Bulanan", index=False)
            rg.reset_index().to_excel(xw, sheet_name="Per Layanan", index=False)
            df.to_excel(xw, sheet_name="Detail", index=False)
        st.download_button("📥 Unduh hasil (Excel)", buf.getvalue(), file_name="simulasi_pendapatan_wifi.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ---------------- TAB 3: TARGET SOLVER ----------------
    with tabs[2]:
        st.subheader("Berapa AP yang harus dipenuhi untuk mencapai target?")
        net_factor = (1 - params["tax"]) * params["collection"]
        if net_factor <= 0:
            st.error("Tingkat penagihan atau faktor pajak membuat pendapatan bersih 0. Periksa parameter.")
        elif total_ap <= 0:
            st.error("Jumlah AP masih 0.")
        else:
            tm = target / 12 / net_factor                      # sewa bruto per bulan yang dibutuhkan
            n_ap = total_ap * params["uptime"]                  # AP beroperasi (setahun penuh)
            s1, s2, s3 = st.columns(3)
            s1.metric("Target per bulan", juta(target / 12))
            s2.metric("Sewa bruto/bulan yang dibutuhkan", juta(tm),
                      help="Setelah memperhitungkan pajak dan tingkat penagihan.")
            s3.metric("Rata-rata tarif per AP dibutuhkan", rp(tm / n_ap),
                      help=f"Dengan {n_ap:,.0f} AP beroperasi sepanjang tahun.".replace(",", "."))

            st.markdown("**Bandingkan dengan skenario saat ini**")
            need_avg = target / ap_months if ap_months > 0 else 0
            cur_avg = total / ap_months if ap_months > 0 else 0
            v1, v2 = st.columns(2)
            v1.metric("Pendapatan per AP-bulan saat ini", rp(cur_avg))
            diff = cur_avg - need_avg
            v2.metric("Dibutuhkan agar target tercapai", rp(need_avg),
                      delta=("+" if diff >= 0 else "-") + rp(abs(diff)) + " dari kebutuhan", delta_color="normal")

            st.markdown("---")
            st.markdown("**1. Bila seluruh AP berada di satu kategori yang sama**")
            rows = []
            for j, t in enumerate(TIERS):
                need = math.ceil(tm / tarif[j]) if tarif[j] > 0 else float("inf")
                rows.append({"Kategori": t, "Tarif/AP/bulan": rp(tarif[j]),
                             "AP dibutuhkan": f"{need:,}".replace(",", "."),
                             "Layak?": "✅ ya" if need <= n_ap else "❌ melebihi jumlah AP"})
            st.dataframe(pd.DataFrame(rows), hide_index=True)

            st.markdown("**2. Kombinasi: sisa AP di kategori dasar, sebagian naik ke kategori lebih tinggi**")
            m1, m2 = st.columns(2)
            base_t = m1.selectbox("Kategori dasar (sisa AP)", TIERS[:-1], index=0)
            up_opts = TIERS[TIERS.index(base_t) + 1:]
            up_t = m2.selectbox("Kategori yang harus dicapai", up_opts, index=min(1, len(up_opts) - 1))
            bi, ui = TIERS.index(base_t), TIERS.index(up_t)
            x = ap_needed(tm, tarif[bi], tarif[ui], n_ap)
            if x <= 0:
                st.success(f"Seluruh AP di {base_t} saja sudah memenuhi target.")
            elif x > n_ap:
                st.error(f"Tidak mungkin: bahkan bila seluruh AP berada di {up_t}, target belum tercapai.")
            else:
                min_login_total = x * TIER_MIN_LOGIN[ui] + (n_ap - x) * TIER_MIN_LOGIN[bi]
                r1, r2, r3 = st.columns(3)
                r1.metric(f"AP di {up_t} yang dibutuhkan", f"{math.ceil(x):,}".replace(",", "."))
                r2.metric("Porsi dari AP beroperasi", f"{x / n_ap * 100:.0f}%")
                r3.metric(f"Login minimum per AP {up_t}", f"{TIER_MIN_LOGIN[ui]:,}/bln".replace(",", "."),
                          help=f"≈ {TIER_MIN_LOGIN[ui] / 30:,.0f} login per hari.".replace(",", "."))
                st.caption(f"Sisanya (±{n_ap - x:,.0f} AP) di {base_t}. Total login sesi minimum yang diperlukan "
                           f"≈ {min_login_total:,.0f} per bulan (≈ {min_login_total / 30:,.0f} per hari)."
                           .replace(",", "."))

            st.markdown("**Matriks kebutuhan AP (kategori dasar × kategori tujuan)**")
            mat = {}
            for bt in TIERS[:-1]:
                rowd = {}
                for ut in TIERS[1:]:
                    b, u = TIERS.index(bt), TIERS.index(ut)
                    if u <= b:
                        rowd[ut] = "—"
                        continue
                    xx = ap_needed(tm, tarif[b], tarif[u], n_ap)
                    if xx <= 0:
                        rowd[ut] = "tercapai"
                    elif xx > n_ap:
                        rowd[ut] = "tidak mungkin"
                    else:
                        rowd[ut] = f"{math.ceil(xx)} AP ({xx / n_ap * 100:.0f}%)"
                mat[f"Sisa di {bt}"] = rowd
            st.dataframe(pd.DataFrame(mat).T)

            st.markdown("---")
            st.markdown("**3. Menutup selisih skenario saat ini**")
            if gap >= 0:
                st.success("Skenario saat ini sudah memenuhi target. Tidak ada selisih yang perlu ditutup.")
            else:
                short_m = (-gap) / 12 / net_factor
                st.write(f"Kekurangan {rp(-gap)} per tahun (≈ {rp(short_m)} sewa bruto per bulan). "
                         "Jumlah AP Basic yang perlu naik kategori (efek setahun penuh):")
                rows = []
                for j in range(1, 5):
                    rows.append({"Naik ke": TIERS[j],
                                 "Tambahan pendapatan/AP/bulan": rp(tarif[j] - tarif[0]),
                                 "AP yang perlu naik": f"{math.ceil(short_m / (tarif[j] - tarif[0])):,}".replace(",", ".")})
                st.dataframe(pd.DataFrame(rows), hide_index=True)

    # ---------------- TAB 4: SENSITIVITAS ----------------
    with tabs[3]:
        st.subheader("Sensitivitas pendapatan setahun")
        st.write("Seberapa besar pendapatan bila **volume login** berbeda dari asumsi dan **persentase AP yang aktif** "
                 "berubah. Parameter lain mengikuti sidebar; ramp-up diabaikan (AP aktif dianggap tetap sepanjang tahun).")
        mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
        actives = [40, 60, 80, 100]
        z, txt = [], []
        for mu in mults:
            zr, tr = [], []
            for a in actives:
                pp = dict(params)
                pp["traffic_mult"] = params["traffic_mult"] * mu
                pp["active_override"] = a / 100
                rev = float(simulate(groups_df, tarif, pp)["Pendapatan"].sum())
                ratio = rev / target if target > 0 else 0
                zr.append(ratio)
                tr.append(f"{rev / 1e6:,.0f} jt<br>({ratio * 100:.0f}%)")
            z.append(zr)
            txt.append(tr)
        figh = go.Figure(go.Heatmap(
            z=z, x=[f"{a}% AP aktif" for a in actives], y=[f"login ×{mu:g}" for mu in mults],
            text=txt, texttemplate="%{text}", zmid=1.0, zmin=0.0, zmax=2.0,
            colorscale=[[0, "#D64545"], [0.5, "#FFF3C4"], [1, "#2E7D32"]], showscale=False))
        figh.update_layout(title="Pendapatan setahun (Rp juta) dan % terhadap target — hijau = target tercapai",
                           template="plotly_white", height=460)
        st.plotly_chart(figh)

        # kurva: pendapatan vs pengali login (AP aktif sesuai slider uptime, tanpa ramp-up)
        xs = np.round(np.arange(0.25, 3.01, 0.25), 2)
        ys = []
        for mu in xs:
            pp = dict(params)
            pp["traffic_mult"] = params["traffic_mult"] * float(mu)
            pp["active_override"] = 1.0
            ys.append(float(simulate(groups_df, tarif, pp)["Pendapatan"].sum()) / 1e6)
        figc = go.Figure()
        figc.add_scatter(x=xs, y=ys, mode="lines+markers", name="Pendapatan setahun", line=dict(color="#4A90D9"))
        figc.add_scatter(x=xs, y=[target / 1e6] * len(xs), mode="lines", name="Target",
                         line=dict(color="#D64545", dash="dash"))
        figc.update_layout(title="Pendapatan setahun vs pengali volume login (seluruh AP aktif)",
                           xaxis_title="Pengali volume login", yaxis_title="Rp juta",
                           template="plotly_white", height=380, legend=dict(orientation="h"))
        st.plotly_chart(figc)
        above = [float(m) for m, y in zip(xs, ys) if y >= target / 1e6]
        if above:
            st.info(f"Dengan sebaran saat ini, target tercapai bila volume login minimal sekitar "
                    f"**×{above[0]:g}** dari asumsi (seluruh AP aktif, resolusi 0,25).")
        else:
            st.warning("Target tidak tercapai pada pengali login sampai ×3. Periksa sebaran atau jumlah AP.")


if __name__ == "__main__":
    main()
