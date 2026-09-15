# -*- coding: utf-8 -*-
"""
CIC-Darknet2020 veri seti uzerinde denetimli ogrenme analizi
=============================================================
Makale: "Karanlik Ag ve VPN Trafigi Tespiti: Denetimli Ogrenme Modellerinde
         Performans, Ozellik Onemi ve Yorumlanabilirlik Analizi"

Bu betik makaledeki TUM deneysel sonuclari ve Sekil 2-5'i uretir:
  1) Veri on isleme (kimlik sutunlari, NaN/inf, sifir varyans temizligi)
  2) Ikili siniflandirma  : Normal (Non-Tor + NonVPN) vs Karanlik (Tor + VPN)
  3) Dort sinifli gorev   : Tor, Non-Tor, VPN, NonVPN
  4) Alti model           : LR, GNB, k-NN, KA (DT), RO (RF), HGA (HistGB)
  5) Sinif bazli metrikler ve karisiklik matrisi (RO, dort sinif)
  6) Oznitelik onemi      : Gini + permutasyon (8.000 orneklik alt kume, 5 tekrar)
  7) Oznitelik indirgeme  : Gini'ye gore ilk 10 oznitelik ile RO'nun yeniden egitimi

Veri seti : CIC-Darknet2020 (https://www.unb.ca/cic/datasets/darknet2020.html)
            CSV dosyasini bu betikle ayni klasore "Darknet.CSV" adiyla koyunuz.
Cikti     : results.json (tum metrikler) ve figs/ klasorunde 4 adet PNG sekil.
Not       : Tum rastgelelik RNG = 42 tohumu ile sabitlenmistir.
"""
import json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RNG = 42                      # tekrarlanabilirlik icin sabit tohum
VERI_YOLU = "Darknet.CSV"     # CIC-Darknet2020 CSV dosyasi
FIG = "figs"                  # sekillerin kaydedilecegi klasor
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

t0 = time.time()
df = pd.read_csv(VERI_YOLU, low_memory=False)
print("Ham boyut:", df.shape, flush=True)

# ---------------------------------------------------------------
# 1) VERI ON ISLEME
# ---------------------------------------------------------------
# Kimlik niteligindeki sutunlar (veri sizintisi riski) ve uygulama
# duzeyi etiket (Label.1) kapsam disidir. Src/Dst Port ve Protocol,
# literatürdeki yaygin uygulamayla uyumlu olarak korunmustur.
drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label.1"]
df = df.drop(columns=drop_cols)

# Akis hizi turevi sutunlardaki sonsuz degerler NaN'a cevrilir,
# eksik satirlar silinir (toplam 50 kayit, %0,03).
df = df.replace([np.inf, -np.inf], np.nan)
n_nan = df.isna().any(axis=1).sum()
df = df.dropna().reset_index(drop=True)
print("NaN/inf nedeniyle silinen satir:", n_nan, "kalan:", len(df), flush=True)

# Tum kayitlarda ayni degeri tasiyan (sifir varyansli) oznitelikler elenir.
feat_cols = [c for c in df.columns if c != "Label"]
zero_var = [c for c in feat_cols if df[c].nunique() <= 1]
print("Sifir varyansli sutun sayisi:", len(zero_var), zero_var, flush=True)
df = df.drop(columns=zero_var)
feat_cols = [c for c in df.columns if c != "Label"]
print("Kalan oznitelik sayisi:", len(feat_cols), flush=True)

# Hedef degiskenler: dort sinifli etiket ve ikili ust sinif.
# Tor + VPN -> Karanlik; Non-Tor + NonVPN -> Normal (Lashkari vd. 2020 ile uyumlu)
y4 = df["Label"].values
y2 = np.where(np.isin(y4, ["Tor", "VPN"]), "Karanlik", "Normal")
X = df[feat_cols].astype(np.float64).values

results = {"n_final": int(len(df)), "n_nan": int(n_nan),
           "zero_var": zero_var, "n_feat": len(feat_cols),
           "class_counts": df["Label"].value_counts().to_dict()}

# ---------------------------------------------------------------
# 2) MODELLER
# ---------------------------------------------------------------
# Adil kiyas icin kapsamli hiperparametre aramasi yapilmamis,
# yaygin varsayilan ayarlar kullanilmistir (Tablo 4).
models_def = {
    "LR":  ("Lojistik Regresyon", lambda: LogisticRegression(max_iter=1000, n_jobs=-1)),
    "GNB": ("Naive Bayes", lambda: GaussianNB()),
    "KNN": ("k-En Yakin Komsu (k=5)", lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    "DT":  ("Karar Agaci", lambda: DecisionTreeClassifier(random_state=RNG)),
    "RF":  ("Rastgele Orman", lambda: RandomForestClassifier(n_estimators=100, random_state=RNG, n_jobs=-1)),
    "HGB": ("Gradyan Artirma (Hist.)", lambda: HistGradientBoostingClassifier(random_state=RNG)),
}
# Olcek duyarli modeller: standartlastirma yalnizca bunlara uygulanir.
NEEDS_SCALE = {"LR", "KNN", "GNB"}

def run_task(y, task_name):
    """Verilen hedef icin %70/%30 katmanli ayrimla alti modeli egit ve olc."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30,
                                          random_state=RNG, stratify=y)
    # Standartlastirma istatistikleri YALNIZCA egitim kumesinden hesaplanir.
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    out, fitted = {}, {}
    for key, (name, ctor) in models_def.items():
        m = ctor()
        a, b = (Xtr_s, Xte_s) if key in NEEDS_SCALE else (Xtr, Xte)
        t = time.time(); m.fit(a, ytr); tr_time = time.time() - t
        t = time.time(); yp = m.predict(b); te_time = time.time() - t
        out[key] = {
            "name": name,
            "acc":  accuracy_score(yte, yp),
            "prec": precision_score(yte, yp, average="weighted", zero_division=0),
            "rec":  recall_score(yte, yp, average="weighted", zero_division=0),
            "f1":   f1_score(yte, yp, average="weighted", zero_division=0),
            "f1_macro": f1_score(yte, yp, average="macro", zero_division=0),
            "train_s": tr_time, "test_s": te_time,
        }
        fitted[key] = (m, b)
        print(f"[{task_name}] {name}: acc={out[key]['acc']:.4f} "
              f"f1w={out[key]['f1']:.4f} f1m={out[key]['f1_macro']:.4f} "
              f"({tr_time:.1f}s)", flush=True)
    return out, fitted, (Xtr, Xte, ytr, yte, sc)

# ---------------------------------------------------------------
# 3) IKILI SINIFLANDIRMA (Tablo 5)
# ---------------------------------------------------------------
res2, fit2, data2 = run_task(y2, "IKILI")
results["binary"] = res2

# ROC-AUC (Karanlik sinifi pozitif kabul edilerek)
Xtr, Xte, ytr, yte, sc = data2
for key in models_def:
    m, b = fit2[key]
    try:
        proba = m.predict_proba(b)
        pos_idx = list(m.classes_).index("Karanlik")
        results["binary"][key]["auc"] = roc_auc_score(
            (yte == "Karanlik").astype(int), proba[:, pos_idx])
    except Exception as e:
        print(key, "auc hesaplanamadi:", e)
print("AUC'ler hesaplandi", flush=True)

# ---------------------------------------------------------------
# 4) DORT SINIFLI SINIFLANDIRMA (Tablo 6)
# ---------------------------------------------------------------
res4, fit4, data4 = run_task(y4, "4-SINIF")
results["multi"] = res4

# Sinif bazli metrikler ve karisiklik matrisi: RO modeli (Tablo 7, Sekil 4)
Xtr4, Xte4, ytr4, yte4, sc4 = data4
rf4, b4 = fit4["RF"]
yp4 = rf4.predict(b4)
classes = ["Non-Tor", "NonVPN", "VPN", "Tor"]
per_class = {}
for c in classes:
    per_class[c] = {
        "prec": precision_score(yte4, yp4, labels=[c], average="macro", zero_division=0),
        "rec":  recall_score(yte4, yp4, labels=[c], average="macro", zero_division=0),
        "f1":   f1_score(yte4, yp4, labels=[c], average="macro", zero_division=0),
        "support": int((yte4 == c).sum()),
    }
results["per_class_rf"] = per_class
cm = confusion_matrix(yte4, yp4, labels=classes)
results["cm"] = cm.tolist()
print("Sinif bazli metrikler tamam", flush=True)

# ---------------------------------------------------------------
# 5) OZNITELIK ONEMI (Tablo 8, Sekil 5)
# ---------------------------------------------------------------
# (a) Gini (safsizlik) temelli onem: RO modelinden dogrudan alinir.
imp = pd.Series(rf4.feature_importances_, index=feat_cols).sort_values(ascending=False)
results["rf_importance_top15"] = imp.head(15).to_dict()

# (b) Permutasyon temelli onem: test kumesinden 8.000 orneklik alt kume,
#     5 tekrar, makro F1 olcutu (model-agnostik yaklasim).
rng = np.random.RandomState(RNG)
sub = rng.choice(len(Xte4), size=min(8000, len(Xte4)), replace=False)
t = time.time()
pi = permutation_importance(rf4, Xte4[sub], yte4[sub], n_repeats=5,
                            random_state=RNG, n_jobs=-1, scoring="f1_macro")
print("Permutasyon suresi:", time.time() - t, flush=True)
pi_s = pd.Series(pi.importances_mean, index=feat_cols).sort_values(ascending=False)
results["perm_importance_top15"] = pi_s.head(15).to_dict()

# ---------------------------------------------------------------
# 6) OZNITELIK INDIRGEME: Gini'ye gore ilk 10 oznitelik ile RO
# ---------------------------------------------------------------
top10 = list(imp.head(10).index)
idx10 = [feat_cols.index(c) for c in top10]
rf10 = RandomForestClassifier(n_estimators=100, random_state=RNG, n_jobs=-1)
rf10.fit(Xtr4[:, idx10], ytr4)
yp10 = rf10.predict(Xte4[:, idx10])
results["rf_top10"] = {
    "acc": accuracy_score(yte4, yp10),
    "f1": f1_score(yte4, yp10, average="weighted", zero_division=0),
    "f1_macro": f1_score(yte4, yp10, average="macro", zero_division=0),
    "features": top10,
}
print("Top-10 RF dogruluk:", results["rf_top10"]["acc"], flush=True)

# ===============================================================
# SEKILLER
# ===============================================================
os.makedirs(FIG, exist_ok=True)

# --- Sekil 2 (makale): sinif dagilimi cubuk grafigi -------------
fig, ax = plt.subplots(figsize=(6.2, 3.4))
counts = df["Label"].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
for b_, v in zip(bars, counts.values):
    # Turkce binlik ayraci icin nokta kullanilir (110.394 gibi)
    ax.text(b_.get_x() + b_.get_width()/2, v + 1500,
            f"{v:,}".replace(",", "."), ha="center", fontsize=9)
ax.set_ylabel("Akış sayısı")
ax.set_xlabel("Trafik sınıfı")
ax.set_ylim(0, counts.max()*1.12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/sekil1_dagilim.png", bbox_inches="tight")
plt.close(fig)

# --- Sekil 3 (makale): iki gorevde agirlikli F1 kiyasi ----------
fig, ax = plt.subplots(figsize=(7.2, 3.8))
keys = ["GNB", "LR", "KNN", "DT", "HGB", "RF"]      # artan basari sirasi
names = [models_def[kk][0] for kk in keys]
f1_b = [res2[kk]["f1"]*100 for kk in keys]
f1_m = [res4[kk]["f1"]*100 for kk in keys]
xpos = np.arange(len(keys)); w = 0.38
ax.bar(xpos - w/2, f1_b, w, label="İkili sınıflandırma", color="#4C72B0")
ax.bar(xpos + w/2, f1_m, w, label="Dört sınıflı sınıflandırma", color="#C44E52")
ax.set_xticks(xpos)
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
ax.set_ylabel("Ağırlıklı F1 skoru (%)")
ax.set_ylim(0, 105)
for i, (a_, b_) in enumerate(zip(f1_b, f1_m)):     # cubuk ustu etiketler
    ax.text(i - w/2, a_ + 1, f"{a_:.1f}", ha="center", fontsize=7.5)
    ax.text(i + w/2, b_ + 1, f"{b_:.1f}", ha="center", fontsize=7.5)
ax.legend(loc="lower right", fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/sekil2_karsilastirma.png", bbox_inches="tight")
plt.close(fig)

# --- Sekil 4 (makale): karisiklik matrisi (RO, dort sinif) ------
fig, ax = plt.subplots(figsize=(5.4, 4.6))
cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100  # satir yuzdesi
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(classes); ax.set_yticklabels(classes)
ax.set_xlabel("Tahmin edilen sınıf"); ax.set_ylabel("Gerçek sınıf")
for i in range(4):
    for j in range(4):
        col = "white" if cmn[i, j] > 50 else "black"   # koyu hucrede beyaz yazi
        ax.text(j, i, f"{cm[i, j]:,}".replace(",", ".") + f"\n(%{cmn[i, j]:.1f})",
                ha="center", va="center", fontsize=8, color=col)
fig.colorbar(im, ax=ax, label="Satır yüzdesi (%)")
fig.tight_layout()
fig.savefig(f"{FIG}/sekil3_karisiklik.png", bbox_inches="tight")
plt.close(fig)

# --- Sekil 5 (makale): Gini ve permutasyon onemleri yan yana ----
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
imp.head(15)[::-1].plot.barh(ax=axes[0], color="#4C72B0")
axes[0].set_title("(a) Gini önem düzeyi", fontsize=10)
axes[0].set_xlabel("Önem düzeyi")
pi_s.head(15)[::-1].plot.barh(ax=axes[1], color="#C44E52")
axes[1].set_title("(b) Permütasyon önem düzeyi", fontsize=10)
axes[1].set_xlabel("Makro F1 düşüşü")
for a_ in axes:
    a_.tick_params(axis="y", labelsize=8)
    a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/sekil4_onem.png", bbox_inches="tight")
plt.close(fig)

# Tum sayisal sonuclar JSON olarak kaydedilir (makaledeki tablolarin kaynagi).
with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("TOPLAM SURE:", time.time() - t0, flush=True)
print("BITTI", flush=True)
