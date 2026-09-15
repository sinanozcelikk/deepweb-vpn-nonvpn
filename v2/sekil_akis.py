# -*- coding: utf-8 -*-
"""
Makaledeki Sekil 1'i (onerilen yontemin genel is akisi) ureten betik.
Veri gerektirmez; kutu-ok semasi matplotlib patch nesneleriyle cizilir.
Cikti: figs/sekil1_akis.png
Kullanim: python sekil_akis.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

MAVI, KIRMIZI = "#4C72B0", "#C44E52"
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.2)
ax.axis("off")


def kutu(x, y, w, h, metin, fc="#EAF1FA", ec=MAVI, fs=8.2):
    """Yuvarlatilmis kose kutu + ortalanmis metin."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, metin, ha="center", va="center", fontsize=fs)


def ok(x1, y1, x2, y2):
    """Iki nokta arasinda dolu uclu ok."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, color="#555555", lw=1.1))


# 1. sira (soldan saga): veri -> on isleme -> nihai matris -> gorevler
kutu(0.10, 5.95, 2.15, 1.00, "CIC-Darknet2020\n(158.616 akış,\n85 sütun)")
kutu(2.70, 5.95, 2.35, 1.00, "Veri Ön İşleme\n(kimlik sütunları, NaN/∞,\nsıfır varyans temizliği)", fs=7.6)
kutu(5.50, 5.95, 1.80, 1.00, "64 öznitelik\n158.566 akış")
kutu(7.75, 5.95, 2.15, 1.00, "İki görev\nİkili (Normal/Karanlık)\nDört sınıflı", fs=7.8)
ok(2.25, 6.45, 2.70, 6.45)
ok(5.05, 6.45, 5.50, 6.45)
ok(7.30, 6.45, 7.75, 6.45)
ok(8.80, 5.95, 8.80, 5.25)

# 2. sira (sagdan sola): model egitimi -> dogrulama duzeni -> degerlendirme
kutu(6.95, 4.20, 2.95, 1.00, "Model Eğitimi\nLR, GNB, k-NN, KA,\nRO, HGA", fs=8.2)
kutu(3.60, 4.20, 2.90, 1.00, "Doğrulama Düzeni\n%70/%30 ayrım + 5 katlı\nÇD + grup bazlı ayrım", fs=7.6)
kutu(0.25, 4.20, 2.90, 1.00, "Değerlendirme\nDoğruluk, F1 (ağırlıklı/makro),\nROC-AUC, PR-AUC", fs=7.4)
ok(6.95, 4.70, 6.50, 4.70)
ok(3.60, 4.70, 3.15, 4.70)
ok(1.70, 4.20, 1.70, 3.50)

# 3. sira (soldan saga): istatistiksel sinama -> dengeleme -> hiperparametre
kutu(0.25, 2.45, 2.90, 1.00, "İstatistiksel Sınama\nMcNemar testi\n(model çiftleri)", fs=8.0)
kutu(3.60, 2.45, 2.90, 1.00, "Sınıf Dengesizliği\nMaliyet duyarlı öğrenme\nve SMOTE", fs=8.0)
kutu(6.95, 2.45, 2.95, 1.00, "Hiperparametre\nOptimizasyonu\n(rastgele arama)", fs=8.0)
ok(3.15, 2.95, 3.60, 2.95)
ok(6.50, 2.95, 6.95, 2.95)
ok(8.40, 2.45, 8.40, 1.75)

# 4. sira (sagdan sola): onem analizi -> siralama uyumu -> indirgeme
kutu(6.95, 0.55, 2.95, 1.00, "Öznitelik Önemi\nGini, permütasyon,\nSHAP", fs=8.2)
kutu(3.60, 0.55, 2.90, 1.00, "Sıralama Uyumu\nSpearman / Kendall\nve örtüşme", fs=8.2)
kutu(0.25, 0.55, 2.90, 1.00, "Öznitelik İndirgeme\nve Yorumlanabilirlik",
     fc="#FDECEA", ec=KIRMIZI, fs=8.2)
ok(6.95, 1.05, 6.50, 1.05)
ok(3.60, 1.05, 3.15, 1.05)

os.makedirs("figs", exist_ok=True)
fig.savefig("figs/sekil1_akis.png", bbox_inches="tight")
print("figs/sekil1_akis.png kaydedildi")
