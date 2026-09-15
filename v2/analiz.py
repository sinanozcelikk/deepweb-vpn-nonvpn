# -*- coding: utf-8 -*-
"""
CIC-Darknet2020 veri seti uzerinde denetimli ogrenme analizi
=============================================================
Makale: "Karanlik Ag ve VPN Trafigi Tespiti: Denetimli Ogrenme Modellerinde
         Performans, Ozellik Onemi ve Yorumlanabilirlik Analizi"

Bu betik makaledeki TUM deneysel sonuclari ve Sekil 2-7'yi uretir:

  A) Veri on isleme (kimlik sutunlari, NaN/inf, sifir varyans temizligi)
  B) Ikili siniflandirma  : Normal (Non-Tor + NonVPN) vs Karanlik (Tor + VPN)
                            + ROC-AUC ve PR-AUC                      -> Tablo 5
  C) Dort sinifli gorev   : Tor, Non-Tor, VPN, NonVPN                -> Tablo 6
                            + RO ve HGA icin sinif bazli metrikler   -> Tablo 10
  D) McNemar testleri     : model ciftleri, iki gorev                -> Tablo 8
  E) 5 katli katmanli capraz dogrulama, iki gorev, alti model        -> Tablo 7
  F) Sinif dengesizligi   : maliyet duyarli ogrenme ve SMOTE         -> Tablo 11
  G) Grup bazli ayrim     : kaynak-hedef IP cifti ile veri sizintisi
                            denetimi                                 -> Tablo 12
  H) Oznitelik onemi      : Gini + permutasyon + SHAP                -> Tablo 13
  I) Siralama uyumu       : Spearman, Kendall, ortusme               -> Tablo 14
  J) Oznitelik indirgeme  : ilk 10 oznitelik ile yeniden egitim      -> Tablo 15
  K) Hiperparametre eniyilemesi (rastgele arama)                     -> Tablo 9

Veri seti : CIC-Darknet2020 (https://www.unb.ca/cic/datasets/darknet2020.html)
            CSV dosyasini bu betikle ayni klasore "Darknet.CSV" adiyla koyunuz.
Cikti     : results.json (tum metrikler) ve figs/ klasorunde 6 adet PNG sekil
            (Sekil 1 ayri betikle uretilir: sekil_akis.py).
Not       : Tum rastgelelik RNG = 42 tohumu ile sabitlenmistir.
Kullanim  : python analiz.py
"""
import gc
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupShuffleSplit, RandomizedSearchCV,
                                     StratifiedKFold, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings("ignore")

RNG = 42                      # tekrarlanabilirlik icin sabit tohum
VERI_YOLU = "Darknet.CSV"     # CIC-Darknet2020 CSV dosyasi
FIG = "figs"                  # sekillerin kaydedilecegi klasor
CIKTI = "results.json"
SINIF = ["Non-Tor", "NonVPN", "VPN", "Tor"]

plt.rcParams.update({"font.size": 10, "figure.dpi": 200, "font.family": "DejaVu Sans"})
MAVI, KIRMIZI, YESIL, MOR = "#4C72B0", "#C44E52", "#55A868", "#8172B2"

T0 = time.time()
R = {}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def kaydet():
    """Sonuclari her asamada diske yazar; uzun kosularda ilerleme kaybolmaz."""
    with open(CIKTI, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, default=float)


# "python analiz.py --devam" cagrisi, results.json dosyasinda zaten bulunan
# asamalari atlar ve yarim kalan kosuyu kaldigi yerden surdurur. Bellegi
# gereken ara ciktilar ARA klasorunde saklanir.
DEVAM = "--devam" in sys.argv
ARA = "ara"
os.makedirs(ARA, exist_ok=True)
if DEVAM and os.path.exists(CIKTI):
    with open(CIKTI, encoding="utf-8") as f:
        R = json.load(f)


def tamam(*anahtarlar):
    """Verilen anahtarlarin tamami onceki kosuda uretildiyse asama atlanir."""
    return DEVAM and all(k in R for k in anahtarlar)


# ===============================================================
# A) VERI ON ISLEME
# ===============================================================
log("A) Veri okunuyor:", VERI_YOLU)
df = pd.read_csv(VERI_YOLU, low_memory=False)
log("   Ham boyut:", df.shape)

# Grup anahtari: kaynak-hedef IP cifti. Grup bazli ayrimda (asama G)
# ayni iletisim ciftine ait akislarin tek kumede kalmasi icin kullanilir.
grup = (df["Src IP"].astype(str) + "|" + df["Dst IP"].astype(str)).values

# Kimlik niteligindeki sutunlar (veri sizintisi riski) ve uygulama duzeyi
# etiket (Label.1) kapsam disidir. Src/Dst Port ve Protocol, literaturdeki
# yaygin uygulamayla uyumlu olarak korunmustur.
df = df.drop(columns=["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label.1"])

# Akis hizi turevi sutunlardaki sonsuz degerler NaN'a cevrilir, eksik
# satirlar silinir (toplam 50 kayit, %0,03).
df = df.replace([np.inf, -np.inf], np.nan)
gecerli = ~df.isna().any(axis=1).values
n_nan = int((~gecerli).sum())
df = df[gecerli].reset_index(drop=True)
grup = grup[gecerli]
log("   NaN/inf nedeniyle silinen satir:", n_nan, "| kalan:", len(df))

# Tum kayitlarda ayni degeri tasiyan (sifir varyansli) oznitelikler elenir.
feat = [c for c in df.columns if c != "Label"]
sifir_varyans = [c for c in feat if df[c].nunique() <= 1]
df = df.drop(columns=sifir_varyans)
feat = [c for c in df.columns if c != "Label"]
log("   Sifir varyansli sutun:", len(sifir_varyans), "| kalan oznitelik:", len(feat))

# Hedef degiskenler: dort sinifli etiket ve ikili ust sinif.
# Tor + VPN -> Karanlik; Non-Tor + NonVPN -> Normal
y4 = df["Label"].values
y2 = np.where(np.isin(y4, ["Tor", "VPN"]), "Karanlik", "Normal")
X = np.ascontiguousarray(df[feat].to_numpy(dtype=np.float64))
sinif_dagilim = df["Label"].value_counts().to_dict()
del df
gc.collect()

R["veri"] = {
    "n_ham": 158616,
    "n_final": int(X.shape[0]),
    "n_nan": n_nan,
    "n_oznitelik": len(feat),
    "sifir_varyansli_sutunlar": sifir_varyans,
    "oznitelikler": feat,
    "sinif_dagilim": {k: int(v) for k, v in sinif_dagilim.items()},
    "ikili_dagilim": {k: int(v) for k, v in pd.Series(y2).value_counts().items()},
    "grup_sayisi": int(pd.Series(grup).nunique()),
}
kaydet()
log("   Nihai matris:", X.shape)

# ===============================================================
# MODEL TANIMLARI
# ===============================================================
# Adil kiyas icin kapsamli hiperparametre aramasi yapilmamis, yaygin
# varsayilan ayarlar kullanilmistir (Tablo 4). Bu tercihin sonuclari ne
# olcude sinirladigi asama K'da ayrica olculmektedir.
MODELLER = {
    "LR":  ("Lojistik Regresyon", lambda: LogisticRegression(max_iter=1000)),
    "GNB": ("Gaussian Naive Bayes", lambda: GaussianNB()),
    "KNN": ("k-En Yakin Komsu (k=5)", lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    "DT":  ("Karar Agaci", lambda: DecisionTreeClassifier(random_state=RNG)),
    "RF":  ("Rastgele Orman", lambda: RandomForestClassifier(
        n_estimators=100, random_state=RNG, n_jobs=-1)),
    "HGB": ("Gradyan Artirma (Hist.)", lambda: HistGradientBoostingClassifier(random_state=RNG)),
}
# Olcek duyarli modeller: standartlastirma yalnizca bunlara uygulanir.
OLCEKLI = {"LR", "KNN", "GNB"}


def sinif_bazli(yte, yp, siniflar=SINIF):
    """Her sinif icin kesinlik, duyarlilik, F1 ve test orneklem sayisi."""
    return {c: {"kesinlik": precision_score(yte, yp, labels=[c], average="macro", zero_division=0),
                "duyarlilik": recall_score(yte, yp, labels=[c], average="macro", zero_division=0),
                "f1": f1_score(yte, yp, labels=[c], average="macro", zero_division=0),
                "n": int((yte == c).sum())}
            for c in siniflar if (yte == c).sum() > 0}


def olc(yte, yp):
    """Agirlikli ve makro ortalamali temel basarim olcutleri."""
    return {"dogruluk": accuracy_score(yte, yp),
            "kesinlik": precision_score(yte, yp, average="weighted", zero_division=0),
            "duyarlilik": recall_score(yte, yp, average="weighted", zero_division=0),
            "f1_agirlikli": f1_score(yte, yp, average="weighted", zero_division=0),
            "f1_makro": f1_score(yte, yp, average="macro", zero_division=0)}


def egit_olc(Xtr, Xte, ytr, yte, etiket, pozitif=None):
    """Alti modeli egitir, olcer; tahminleri ve (varsa) olasiliklari dondurur.

    Bellek disiplini: her model olculdukten sonra silinir; ayni anda birden
    fazla topluluk modeli bellekte tutulmaz.
    """
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    sonuc, tahmin, olasilik = {}, {}, {}
    for k, (ad, kur) in MODELLER.items():
        m = kur()
        a, b = (Xtr_s, Xte_s) if k in OLCEKLI else (Xtr, Xte)
        t = time.time(); m.fit(a, ytr); egitim_s = time.time() - t
        t = time.time(); yp = m.predict(b); test_s = time.time() - t
        sonuc[k] = {"ad": ad, **olc(yte, yp), "egitim_s": egitim_s, "test_s": test_s}
        tahmin[k] = yp
        if pozitif is not None:
            try:
                pr = m.predict_proba(b)[:, list(m.classes_).index(pozitif)]
                yb = (yte == pozitif).astype(int)
                sonuc[k]["roc_auc"] = roc_auc_score(yb, pr)
                sonuc[k]["pr_auc"] = average_precision_score(yb, pr)
                olasilik[k] = pr
            except Exception as e:   # olasilik uretemeyen model olursa
                log("   olasilik hesaplanamadi:", k, e)
        log(f"   {etiket} {ad}: dogruluk={sonuc[k]['dogruluk']:.4f} "
            f"makroF1={sonuc[k]['f1_makro']:.4f} ({egitim_s:.1f}s)")
        del m
        gc.collect()
    del Xtr_s, Xte_s, sc
    gc.collect()
    return sonuc, tahmin, olasilik


# ===============================================================
# B) IKILI SINIFLANDIRMA (Tablo 5)
# ===============================================================
ROC_ARA = os.path.join(ARA, "roc_pr.npz")
tahmin2 = None
if tamam("ikili") and os.path.exists(ROC_ARA):
    log("B) Ikili siniflandirma - onceki kosudan alindi")
    _ara = np.load(ROC_ARA, allow_pickle=True)
    pr_hgb, yb_hgb = _ara["pr"], _ara["y"]
else:
    log("B) Ikili siniflandirma (Normal / Karanlik)")
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X, y2, test_size=0.30, random_state=RNG, stratify=y2)
    R["ikili"], tahmin2, olasilik2 = egit_olc(
        Xtr2, Xte2, ytr2, yte2, "IKILI", pozitif="Karanlik")
    pr_hgb, yb_hgb = olasilik2["HGB"], (yte2 == "Karanlik").astype(int)   # Sekil 3 icin
    np.savez(ROC_ARA, pr=pr_hgb, y=yb_hgb)
    del Xtr2, Xte2, ytr2, olasilik2
    gc.collect()
    kaydet()

# ===============================================================
# C) DORT SINIFLI SINIFLANDIRMA (Tablo 6, Tablo 10)
# ===============================================================
Xtr4, Xte4, ytr4, yte4 = train_test_split(
    X, y4, test_size=0.30, random_state=RNG, stratify=y4)
tahmin4 = None
if tamam("dort_sinif", "karisiklik_HGA"):
    log("C) Dort sinifli siniflandirma - onceki kosudan alindi")
    cm_hgb = np.array(R["karisiklik_HGA"])
else:
    log("C) Dort sinifli siniflandirma")
    R["dort_sinif"], tahmin4, _ = egit_olc(Xtr4, Xte4, ytr4, yte4, "4-SINIF")

    R["sinif_bazli_RO"] = sinif_bazli(yte4, tahmin4["RF"])
    R["sinif_bazli_HGA"] = sinif_bazli(yte4, tahmin4["HGB"])
    cm_hgb = confusion_matrix(yte4, tahmin4["HGB"], labels=SINIF)
    R["karisiklik_HGA"] = cm_hgb.tolist()
    R["karisiklik_RO"] = confusion_matrix(yte4, tahmin4["RF"], labels=SINIF).tolist()
    kaydet()
    log("   Sinif bazli metrikler ve karisiklik matrisleri tamam")

# ===============================================================
# D) McNEMAR TESTLERI (Tablo 8)
# ===============================================================
log("D) McNemar testleri")


def mcnemar(y, p1, p2):
    """Sureklilik duzeltmeli ki-kare; uyusmazlik 25'in altindaysa tam binom."""
    d1, d2 = (p1 == y), (p2 == y)
    b = int(np.sum(d1 & ~d2))    # 1. model dogru, 2. model yanlis
    c = int(np.sum(~d1 & d2))    # 1. model yanlis, 2. model dogru
    if b + c == 0:
        return {"b": b, "c": c, "ki_kare": 0.0, "p": 1.0}
    ki2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.chi2.sf(ki2, 1))
    if b + c < 25:
        p = float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))
    return {"b": b, "c": c, "ki_kare": float(ki2), "p": p}


CIFTLER = [("HGB", "RF"), ("HGB", "DT"), ("RF", "DT"), ("HGB", "KNN"),
           ("DT", "KNN"), ("KNN", "LR"), ("LR", "GNB")]
if tahmin2 is None or tahmin4 is None:
    log("   McNemar - onceki kosudan alindi")
else:
    R["mcnemar_dort_sinif"] = {f"{a}-{b}": mcnemar(yte4, tahmin4[a], tahmin4[b])
                               for a, b in CIFTLER}
    R["mcnemar_ikili"] = {f"{a}-{b}": mcnemar(yte2, tahmin2[a], tahmin2[b])
                          for a, b in CIFTLER}
    for k, v in R["mcnemar_dort_sinif"].items():
        log(f"   4-sinif {k}: b={v['b']} c={v['c']} ki2={v['ki_kare']:.2f} p={v['p']:.3e}")
    kaydet()
del tahmin2, tahmin4
gc.collect()

# ===============================================================
# E) 5 KATLI KATMANLI CAPRAZ DOGRULAMA (Tablo 7)
# ===============================================================
log("E) 5 katli katmanli capraz dogrulama")


def capraz_dogrula(y, etiket, tor_olc=False, n_kat=5):
    skf = StratifiedKFold(n_splits=n_kat, shuffle=True, random_state=RNG)
    kayit = {k: {"dogruluk": [], "f1_agirlikli": [], "f1_makro": [], "tor": []}
             for k in MODELLER}
    for i, (tr, te) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X[tr])
        Xs_tr, Xs_te = sc.transform(X[tr]), sc.transform(X[te])
        for k in MODELLER:
            m = MODELLER[k][1]()
            a, b = (Xs_tr, Xs_te) if k in OLCEKLI else (X[tr], X[te])
            m.fit(a, y[tr]); yp = m.predict(b)
            kayit[k]["dogruluk"].append(accuracy_score(y[te], yp))
            kayit[k]["f1_agirlikli"].append(f1_score(y[te], yp, average="weighted", zero_division=0))
            kayit[k]["f1_makro"].append(f1_score(y[te], yp, average="macro", zero_division=0))
            if tor_olc:
                kayit[k]["tor"].append(
                    recall_score(y[te], yp, labels=["Tor"], average="macro", zero_division=0))
            del m
            gc.collect()
        del Xs_tr, Xs_te, sc
        gc.collect()
        log(f"   {etiket} kat {i + 1}/{n_kat} tamamlandi")

    out = {}
    for k, v in kayit.items():
        out[k] = {"ad": MODELLER[k][0]}
        for olcut in ("dogruluk", "f1_agirlikli", "f1_makro"):
            out[k][olcut + "_ort"] = float(np.mean(v[olcut]))
            out[k][olcut + "_ss"] = float(np.std(v[olcut], ddof=1))
        out[k]["dogruluk_katlar"] = [float(x) for x in v["dogruluk"]]
        if tor_olc:
            out[k]["tor_duyarlilik_ort"] = float(np.mean(v["tor"]))
            out[k]["tor_duyarlilik_ss"] = float(np.std(v["tor"], ddof=1))
        log(f"   {etiket} {k}: dogruluk={out[k]['dogruluk_ort']:.4f}"
            f"±{out[k]['dogruluk_ss']:.4f} makroF1={out[k]['f1_makro_ort']:.4f}"
            f"±{out[k]['f1_makro_ss']:.4f}")
    return out


if tamam("cd_ikili"):
    log("   CD-IKILI - onceki kosudan alindi")
else:
    R["cd_ikili"] = capraz_dogrula(y2, "CD-IKILI")
    kaydet()
if tamam("cd_dort_sinif"):
    log("   CD-4SINIF - onceki kosudan alindi")
else:
    R["cd_dort_sinif"] = capraz_dogrula(y4, "CD-4SINIF", tor_olc=True)
    kaydet()

# ===============================================================
# F) SINIF DENGESIZLIGI: MALIYET DUYARLI OGRENME VE SMOTE (Tablo 11)
# ===============================================================
if tamam("dengeleme"):
    log("F) Sinif dengesizligi - onceki kosudan alindi")
else:
    log("F) Sinif dengesizligine yonelik yaklasimlar")
    deng = {}


    def deng_olc(yp):
        d = olc(yte4, yp)
        for c in SINIF:
            d[f"{c}_kesinlik"] = precision_score(yte4, yp, labels=[c], average="macro", zero_division=0)
            d[f"{c}_duyarlilik"] = recall_score(yte4, yp, labels=[c], average="macro", zero_division=0)
            d[f"{c}_f1"] = f1_score(yte4, yp, labels=[c], average="macro", zero_division=0)
        return d


    # (1) maliyet duyarli ogrenme: sinif agirligi
    for ad, model, olcekli in [
            ("RO_sinif_agirlikli", RandomForestClassifier(
                n_estimators=100, random_state=RNG, n_jobs=-1, class_weight="balanced"), False),
            ("KA_sinif_agirlikli", DecisionTreeClassifier(
                random_state=RNG, class_weight="balanced"), False),
            ("LR_sinif_agirlikli", LogisticRegression(
                max_iter=1000, class_weight="balanced"), True)]:
        if olcekli:
            sc = StandardScaler().fit(Xtr4)
            model.fit(sc.transform(Xtr4), ytr4)
            yp = model.predict(sc.transform(Xte4))
            del sc
        else:
            model.fit(Xtr4, ytr4)
            yp = model.predict(Xte4)
        deng[ad] = deng_olc(yp)
        log(f"   {ad}: makroF1={deng[ad]['f1_makro']:.4f} "
            f"Tor_duyarlilik={deng[ad]['Tor_duyarlilik']:.4f}")
        del model
        gc.collect()

    # HGA sinif agirligi parametresi almadigi icin ornek agirligi verilir.
    sw = compute_sample_weight("balanced", ytr4)
    m = HistGradientBoostingClassifier(random_state=RNG).fit(Xtr4, ytr4, sample_weight=sw)
    deng["HGA_ornek_agirlikli"] = deng_olc(m.predict(Xte4))
    log(f"   HGA_ornek_agirlikli: makroF1={deng['HGA_ornek_agirlikli']['f1_makro']:.4f} "
        f"Tor_duyarlilik={deng['HGA_ornek_agirlikli']['Tor_duyarlilik']:.4f}")
    del m, sw
    gc.collect()

    # (2) SMOTE: yalnizca egitim kumesine uygulanir, test kumesi ozgun birakilir.
    t = time.time()
    Xs, ys = SMOTE(random_state=RNG, k_neighbors=5).fit_resample(Xtr4, ytr4)
    R["smote_egitim_dagilim"] = {k: int(v) for k, v in pd.Series(ys).value_counts().items()}
    log(f"   SMOTE sonrasi egitim kumesi: {Xs.shape} ({time.time() - t:.0f}s)")
    m = HistGradientBoostingClassifier(random_state=RNG).fit(Xs, ys)
    deng["HGA_SMOTE"] = deng_olc(m.predict(Xte4))
    log(f"   HGA_SMOTE: makroF1={deng['HGA_SMOTE']['f1_makro']:.4f} "
        f"Tor_duyarlilik={deng['HGA_SMOTE']['Tor_duyarlilik']:.4f}")
    del m, Xs, ys
    gc.collect()
    R["dengeleme"] = deng
    kaydet()

# ===============================================================
# G) GRUP BAZLI AYRIM (Tablo 12)
# ===============================================================
if tamam("grup_dort_sinif", "grup_ikili", "karisiklik_HGA_grup"):
    log("G) Grup bazli ayrim - onceki kosudan alindi")
    cm_hgb_grup = np.array(R["karisiklik_HGA_grup"])
else:
    log("G) Grup bazli ayrim (kaynak-hedef IP cifti)")
    gtr, gte = next(GroupShuffleSplit(
        n_splits=1, test_size=0.30, random_state=RNG).split(X, y4, groups=grup))
    R["grup_ayrim_bilgi"] = {
        "n_egitim": int(len(gtr)), "n_test": int(len(gte)),
        "ortak_grup_sayisi": int(len(set(grup[gtr]) & set(grup[gte]))),
        "egitim_dagilim": {k: int(v) for k, v in pd.Series(y4[gtr]).value_counts().items()},
        "test_dagilim": {k: int(v) for k, v in pd.Series(y4[gte]).value_counts().items()},
    }
    log("   egitim:", len(gtr), "test:", len(gte),
        "| ortak grup:", R["grup_ayrim_bilgi"]["ortak_grup_sayisi"])

    g4, gtahmin4, _ = egit_olc(X[gtr], X[gte], y4[gtr], y4[gte], "GRUP-4SINIF")
    for k in MODELLER:
        g4[k]["sinif_bazli"] = sinif_bazli(y4[gte], gtahmin4[k])
    R["grup_dort_sinif"] = g4
    cm_hgb_grup = confusion_matrix(y4[gte], gtahmin4["HGB"], labels=SINIF)
    R["karisiklik_HGA_grup"] = cm_hgb_grup.tolist()
    del gtahmin4
    gc.collect()
    kaydet()

    R["grup_ikili"], _, _ = egit_olc(X[gtr], X[gte], y2[gtr], y2[gte],
                                     "GRUP-IKILI", pozitif="Karanlik")
    # Bundan sonraki asamalar yalnizca dort sinifli ayrimi kullanir; tam veri
    # matrisi bellekten dusurulur (bellegi sinirli makinelerde onemli).
del X, y2, y4
gc.collect()
kaydet()

# ===============================================================
# H) OZNITELIK ONEMI: GINI + PERMUTASYON (Tablo 13)
# ===============================================================
rng = np.random.RandomState(RNG)
if tamam("onem_gini", "onem_permutasyon"):
    log("H) Oznitelik onemi - onceki kosudan alindi")
    gini = pd.Series(R["onem_gini"])
    perm = pd.Series(R["onem_permutasyon"])
    # Cekilis sirasi korunur: SHAP alt kumesi ayni olsun diye rng ilerletilir.
    rng.choice(len(Xte4), size=min(8000, len(Xte4)), replace=False)
else:
    log("H) Oznitelik onemi (Gini ve permutasyon)")
    rf = RandomForestClassifier(n_estimators=100, random_state=RNG, n_jobs=-1).fit(Xtr4, ytr4)
    gini = pd.Series(rf.feature_importances_, index=feat).sort_values(ascending=False)
    R["onem_gini"] = gini.to_dict()
    log("   Gini ilk 5:", [f"{k} {v:.4f}" for k, v in list(gini.head(5).items())])

    # Permutasyon onemi: test kumesinden 8.000 orneklik alt kume, 5 tekrar,
    # makro F1 olcutu. Ayni RandomState nesnesi asama J'de SHAP alt kumesi icin
    # de kullanilir; cekilis sirasi tekrarlanabilirlik acisindan onemlidir.
    alt = rng.choice(len(Xte4), size=min(8000, len(Xte4)), replace=False)
    t = time.time()
    pi = permutation_importance(rf, Xte4[alt], yte4[alt], n_repeats=5,
                                random_state=RNG, scoring="f1_macro")
    log(f"   Permutasyon suresi: {time.time() - t:.0f}s")
    perm = pd.Series(pi.importances_mean, index=feat).sort_values(ascending=False)
    R["onem_permutasyon"] = perm.to_dict()
    R["onem_permutasyon_ss"] = pd.Series(pi.importances_std, index=feat).to_dict()
    log("   Permutasyon ilk 5:", [f"{k} {v:.4f}" for k, v in list(perm.head(5).items())])
    del rf, pi
    gc.collect()
    kaydet()

# ===============================================================
# J-1) OZNITELIK INDIRGEME: GINI ILK 10 (Tablo 15)
# ===============================================================
gini_top10 = list(gini.head(10).index)
idx_gini = [feat.index(c) for c in gini_top10]
if tamam("RO_gini_ilk10", "HGA_gini_ilk10"):
    log("J) Oznitelik indirgeme (Gini) - onceki kosudan alindi")
else:
    log("J) Oznitelik indirgeme")
    for ad, model in [("RO_gini_ilk10", RandomForestClassifier(
                          n_estimators=100, random_state=RNG, n_jobs=-1)),
                      ("HGA_gini_ilk10", HistGradientBoostingClassifier(random_state=RNG))]:
        model.fit(Xtr4[:, idx_gini], ytr4)
        yp = model.predict(Xte4[:, idx_gini])
        R[ad] = {**olc(yte4, yp), "oznitelikler": gini_top10,
                 "tor_duyarlilik": recall_score(yte4, yp, labels=["Tor"],
                                                average="macro", zero_division=0)}
        log(f"   {ad}: dogruluk={R[ad]['dogruluk']:.4f} makroF1={R[ad]['f1_makro']:.4f}")
        del model
        gc.collect()
    kaydet()

# ===============================================================
# I-1) SHAP ANALIZI (HGA, dort sinif)
# ===============================================================
if tamam("onem_shap", "onem_shap_sinif_bazli", "HGA_shap_ilk10"):
    log("I) SHAP analizi - onceki kosudan alindi")
    shp = pd.Series(R["onem_shap"])
    siniflar_hgb = R["shap_siniflar"]
    shap_top10 = R["HGA_shap_ilk10"]["oznitelikler"]
else:
    log("I) SHAP analizi (HGA)")
    hgb = HistGradientBoostingClassifier(random_state=RNG).fit(Xtr4, ytr4)
    aciklayici = shap.TreeExplainer(hgb)
    alt_shap = rng.choice(len(Xte4), size=min(3000, len(Xte4)), replace=False)
    t = time.time()
    sv = np.array(aciklayici.shap_values(Xte4[alt_shap]))   # (n, oznitelik, sinif)
    log(f"   SHAP sekli: {sv.shape} ({time.time() - t:.0f}s)")
    siniflar_hgb = list(hgb.classes_)
    R["shap_ornek_sayisi"] = int(len(alt_shap))
    R["shap_siniflar"] = siniflar_hgb

    shap_kuresel = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
    shp = pd.Series(shap_kuresel, index=feat).sort_values(ascending=False)
    R["onem_shap"] = shp.to_dict()
    if sv.ndim == 3:
        R["onem_shap_sinif_bazli"] = {
            c: pd.Series(np.abs(sv[:, :, i]).mean(axis=0), index=feat).to_dict()
            for i, c in enumerate(siniflar_hgb)}
    log("   SHAP ilk 5:", [f"{k} {v:.3f}" for k, v in list(shp.head(5).items())])
    # Sekil 7 sinif bazli sozlukten uretilir; buyuk SHAP dizisi serbest birakilir.
    del hgb, aciklayici, sv
    gc.collect()

    # SHAP ilk 10 oznitelik ile indirgeme (Tablo 15)
    shap_top10 = list(shp.head(10).index)
    idx_shap = [feat.index(c) for c in shap_top10]
    m = HistGradientBoostingClassifier(random_state=RNG).fit(Xtr4[:, idx_shap], ytr4)
    yp = m.predict(Xte4[:, idx_shap])
    R["HGA_shap_ilk10"] = {**olc(yte4, yp), "oznitelikler": shap_top10,
                           "tor_duyarlilik": recall_score(yte4, yp, labels=["Tor"],
                                                          average="macro", zero_division=0)}
    log(f"   HGA_shap_ilk10: dogruluk={R['HGA_shap_ilk10']['dogruluk']:.4f} "
        f"makroF1={R['HGA_shap_ilk10']['f1_makro']:.4f}")
    del m
    gc.collect()
    kaydet()

# ===============================================================
# I-2) SIRALAMA UYUMU (Tablo 14)
# ===============================================================
if tamam("siralama_uyumu", "siralama_ortusme"):
    log("I) Siralama uyumu - onceki kosudan alindi")
else:
    log("I) Onem yontemleri arasindaki siralama uyumu")
    a_ = np.array([gini[c] for c in feat])
    b_ = np.array([perm[c] for c in feat])
    c_ = np.array([shp[c] for c in feat])


    def uyum(u, v):
        sp, kt = stats.spearmanr(u, v), stats.kendalltau(u, v)
        return {"spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
                "kendall": float(kt.statistic), "kendall_p": float(kt.pvalue)}


    R["siralama_uyumu"] = {"gini_permutasyon": uyum(a_, b_),
                           "gini_shap": uyum(a_, c_),
                           "permutasyon_shap": uyum(b_, c_)}
    g10, p10, s10 = (set(gini.head(10).index), set(perm.head(10).index), set(shp.head(10).index))
    g15, p15 = set(gini.head(15).index), set(perm.head(15).index)
    R["siralama_ortusme"] = {
        "gini_permutasyon_ilk10": len(g10 & p10),
        "gini_permutasyon_ilk15": len(g15 & p15),
        "gini_shap_ilk10": len(g10 & s10),
        "permutasyon_shap_ilk10": len(p10 & s10),
        "uc_yontemde_ortak_ilk10": sorted(g10 & p10 & s10),
        "gini_ilk10": gini_top10,
        "permutasyon_ilk10": list(perm.head(10).index),
        "shap_ilk10": shap_top10,
    }
    log("   Uyum:", R["siralama_uyumu"])
    log("   Ortusme:", R["siralama_ortusme"]["uc_yontemde_ortak_ilk10"])
    kaydet()

# ===============================================================
# SEKILLER (Sekil 2-7)
# ===============================================================
log("Sekiller uretiliyor")
os.makedirs(FIG, exist_ok=True)

# --- Sekil 2: sinif dagilimi -----------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.4))
degerler = [R["veri"]["sinif_dagilim"][c] for c in SINIF]
cubuklar = ax.bar(SINIF, degerler, color=[MAVI, YESIL, KIRMIZI, MOR])
for cb, v in zip(cubuklar, degerler):
    # Turkce binlik ayraci icin nokta kullanilir (110.394 gibi)
    ax.text(cb.get_x() + cb.get_width() / 2, v + max(degerler) * 0.02,
            f"{v:,}".replace(",", "."), ha="center", fontsize=9)
ax.set_ylabel("Akış sayısı"); ax.set_xlabel("Trafik sınıfı")
ax.set_ylim(0, max(degerler) * 1.14)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/sekil2_dagilim.png", bbox_inches="tight")
plt.close(fig)

# --- Sekil 3: HGA icin ROC ve kesinlik-duyarlilik egrileri -----
fpr, tpr, _ = roc_curve(yb_hgb, pr_hgb)
kesinlik, duyarlilik, _ = precision_recall_curve(yb_hgb, pr_hgb)
taban = yb_hgb.mean()
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.5))
axs[0].plot(fpr, tpr, color=MAVI, lw=1.6,
            label=f"HGA (AUC = {R['ikili']['HGB']['roc_auc']:.4f})")
axs[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
axs[0].set_xlabel("Yanlış pozitif oranı"); axs[0].set_ylabel("Doğru pozitif oranı")
axs[0].set_title("(a) ROC eğrisi", fontsize=10)
axs[0].legend(fontsize=8, loc="lower right")
axs[1].plot(duyarlilik, kesinlik, color=KIRMIZI, lw=1.6,
            label=f"HGA (PR-AUC = {R['ikili']['HGB']['pr_auc']:.4f})")
axs[1].axhline(taban, ls="--", color="#999999", lw=1, label=f"Rastgele ({taban:.3f})")
axs[1].set_xlabel("Duyarlılık"); axs[1].set_ylabel("Kesinlik")
axs[1].set_title("(b) Kesinlik-duyarlılık eğrisi", fontsize=10)
axs[1].legend(fontsize=8, loc="lower left"); axs[1].set_ylim(0, 1.04)
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/sekil3_roc_pr.png", bbox_inches="tight")
plt.close(fig)

# --- Sekil 4: capraz dogrulamada agirlikli F1 kiyasi -----------
sira = ["GNB", "LR", "KNN", "DT", "RF", "HGB"]          # artan basari sirasi
adlar = {"GNB": "Gaussian NB", "LR": "Lojistik Reg.", "KNN": "k-NN",
         "DT": "Karar Ağacı", "RF": "Rastgele Orman", "HGB": "HGA"}
fig, ax = plt.subplots(figsize=(7.4, 3.9))
xpos = np.arange(len(sira)); w = 0.38
b1 = [R["cd_ikili"][k]["f1_agirlikli_ort"] * 100 for k in sira]
e1 = [R["cd_ikili"][k]["f1_agirlikli_ss"] * 100 for k in sira]
b2 = [R["cd_dort_sinif"][k]["f1_agirlikli_ort"] * 100 for k in sira]
e2 = [R["cd_dort_sinif"][k]["f1_agirlikli_ss"] * 100 for k in sira]
ax.bar(xpos - w / 2, b1, w, yerr=e1, capsize=3, label="İkili sınıflandırma", color=MAVI)
ax.bar(xpos + w / 2, b2, w, yerr=e2, capsize=3, label="Dört sınıflı sınıflandırma", color=KIRMIZI)
ax.set_xticks(xpos); ax.set_xticklabels([adlar[k] for k in sira],
                                        rotation=18, ha="right", fontsize=8.5)
ax.set_ylabel("Ağırlıklı F1 skoru (%)"); ax.set_ylim(0, 108)
for i, (u, v) in enumerate(zip(b1, b2)):
    ax.text(i - w / 2, u + 2.4, f"{u:.1f}", ha="center", fontsize=7.4)
    ax.text(i + w / 2, v + 2.4, f"{v:.1f}", ha="center", fontsize=7.4)
ax.legend(loc="lower right", fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/sekil4_f1.png", bbox_inches="tight")
plt.close(fig)

# --- Sekil 5: HGA karisiklik matrisleri (iki ayrim duzeni) -----
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2))
for ax_, mtx, baslik in [(axs[0], cm_hgb, "(a) Katmanlı rastgele ayrım"),
                         (axs[1], cm_hgb_grup, "(b) Grup bazlı ayrım")]:
    yuzde = mtx.astype(float) / mtx.sum(axis=1, keepdims=True) * 100   # satir yuzdesi
    im = ax_.imshow(yuzde, cmap="Blues", vmin=0, vmax=100)
    ax_.set_xticks(range(4)); ax_.set_yticks(range(4))
    ax_.set_xticklabels(SINIF, fontsize=8.5); ax_.set_yticklabels(SINIF, fontsize=8.5)
    ax_.set_xlabel("Tahmin edilen sınıf", fontsize=9)
    ax_.set_ylabel("Gerçek sınıf", fontsize=9)
    ax_.set_title(baslik, fontsize=10)
    for i in range(4):
        for j in range(4):
            renk = "white" if yuzde[i, j] > 50 else "black"   # koyu hucrede beyaz yazi
            ax_.text(j, i, f"{mtx[i, j]:,}".replace(",", ".") + f"\n(%{yuzde[i, j]:.1f})",
                     ha="center", va="center", fontsize=7.2, color=renk)
fig.colorbar(im, ax=axs, label="Satır yüzdesi (%)", fraction=0.03)
fig.savefig(f"{FIG}/sekil5_karisiklik.png", bbox_inches="tight")
plt.close(fig)

# --- Sekil 6: uc onem yontemi yan yana ------------------------
fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.4))
for ax_, seri, baslik, renk, xetiket in [
        (axs[0], gini, "(a) Gini önem düzeyi", MAVI, "Safsızlık azalması"),
        (axs[1], perm, "(b) Permütasyon önem düzeyi", KIRMIZI, "Makro F1 düşüşü"),
        (axs[2], shp, "(c) SHAP önem düzeyi", YESIL, "Ortalama |SHAP|")]:
    seri.head(15)[::-1].plot.barh(ax=ax_, color=renk)
    ax_.set_title(baslik, fontsize=10); ax_.set_xlabel(xetiket, fontsize=9)
    ax_.tick_params(axis="y", labelsize=7.6)
    ax_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/sekil6_onem.png", bbox_inches="tight")
plt.close(fig)

# --- Sekil 7: sinif bazli SHAP isi haritasi -------------------
ilk12 = list(shp.head(12).index)
sinif_shap = R["onem_shap_sinif_bazli"]
mat = np.vstack([[sinif_shap[c][o] for o in ilk12] for c in siniflar_hgb])
matn = mat / mat.max(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(9.0, 3.6))
im = ax.imshow(matn, cmap="YlOrRd", aspect="auto")
ax.set_yticks(range(len(siniflar_hgb))); ax.set_yticklabels(siniflar_hgb, fontsize=9)
ax.set_xticks(range(len(ilk12))); ax.set_xticklabels(ilk12, rotation=38, ha="right", fontsize=7.8)
for i in range(len(siniflar_hgb)):
    for j in range(len(ilk12)):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.6,
                color="white" if matn[i, j] > 0.6 else "black")
fig.colorbar(im, ax=ax, label="Sınıf içi göreli katkı", fraction=0.03)
ax.set_xlabel("Öznitelik", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/sekil7_shap_sinif.png", bbox_inches="tight")
plt.close(fig)

# ===============================================================
# K) HIPERPARAMETRE ENIYILEMESI (Tablo 9)
# ===============================================================
log("K) Hiperparametre eniyilemesi (rastgele arama, 3 kat, makro F1)")
ARAMA = {
    "HGA": (HistGradientBoostingClassifier(random_state=RNG),
            {"learning_rate": [0.05, 0.1, 0.2], "max_iter": [100, 200],
             "max_leaf_nodes": [31, 63, 127], "min_samples_leaf": [20, 50],
             "l2_regularization": [0.0, 1.0]}, 6),
    "KA":  (DecisionTreeClassifier(random_state=RNG),
            {"max_depth": [None, 20, 30, 40], "min_samples_split": [2, 5, 10],
             "min_samples_leaf": [1, 2, 4], "criterion": ["gini", "entropy"]}, 6),
    "RO":  (RandomForestClassifier(random_state=RNG, n_jobs=-1),
            {"n_estimators": [100], "max_depth": [None, 25],
             "min_samples_split": [2, 5], "min_samples_leaf": [1, 2],
             "max_features": ["sqrt", 0.3]}, 4),
}
hp = dict(R.get("hiperparametre", {})) if DEVAM else {}
for ad, (kestirici, izgara, n_deneme) in ARAMA.items():
    if ad in hp:
        log(f"   {ad} - onceki kosudan alindi")
        continue
    t = time.time()
    arama = RandomizedSearchCV(kestirici, izgara, n_iter=n_deneme, cv=3,
                               scoring="f1_macro", random_state=RNG, n_jobs=1, refit=True)
    arama.fit(Xtr4, ytr4)
    yp = arama.best_estimator_.predict(Xte4)
    hp[ad] = {"en_iyi_parametreler": {k: (str(v) if v is None else v)
                                      for k, v in arama.best_params_.items()},
              "cd_makro_f1": float(arama.best_score_),
              **olc(yte4, yp),
              "tor_duyarlilik": recall_score(yte4, yp, labels=["Tor"],
                                             average="macro", zero_division=0),
              "sure_s": time.time() - t, "n_deneme": n_deneme}
    log(f"   {ad}: {arama.best_params_}")
    log(f"      -> dogruluk={hp[ad]['dogruluk']:.4f} makroF1={hp[ad]['f1_makro']:.4f} "
        f"({hp[ad]['sure_s']:.0f}s)")
    del arama
    gc.collect()
    R["hiperparametre"] = hp
    kaydet()

del Xtr4, Xte4, ytr4
gc.collect()

R["toplam_sure_s"] = time.time() - T0
kaydet()
log("Sekiller:", sorted(os.listdir(FIG)))
log(f"Tum sonuclar {CIKTI} dosyasina yazildi.")
log("BITTI")
