# -*- coding: utf-8 -*-
"""
Makaledeki Sekil 1'i (onerilen yontemin genel is akisi) ureten betik.
Veri gerektirmez; kutu-ok semasi matplotlib patch nesneleriyle cizilir.
Cikti: figs/sekil0_akis.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")

def box(x, y, w, h, text, fc="#EAF1FA", ec="#4C72B0", fs=8.6):
    """Yuvarlatilmis kose kutu + ortalanmis metin."""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                       fc=fc, ec=ec, lw=1.3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)

def arrow(x1, y1, x2, y2):
    """Iki nokta arasinda dolu uclu ok."""
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=13, color="#555555", lw=1.2)
    ax.add_patch(a)

# Ust sira: veri -> on isleme -> nihai matris -> ayrim
box(0.15, 4.3, 2.2, 1.0, "CIC-Darknet2020\n(158.616 akış,\n85 sütun)")
box(2.85, 4.3, 2.4, 1.0, "Veri Ön İşleme\n(kimlik sütunları, NaN/∞,\nsıfır varyans temizliği)")
box(5.75, 4.3, 1.9, 1.0, "64 öznitelik\n158.566 akış")
box(8.05, 4.3, 1.8, 1.0, "Katmanlı Ayrım\n%70 eğitim\n%30 test")
arrow(2.35, 4.8, 2.85, 4.8)
arrow(5.25, 4.8, 5.75, 4.8)
arrow(7.65, 4.8, 8.05, 4.8)
arrow(8.95, 4.3, 8.95, 3.55)

# Orta sira (sagdan sola): egitim -> gorevler -> degerlendirme
box(6.9, 2.6, 2.95, 0.95, "Model Eğitimi\nLR, GNB, k-NN, DT,\nRO, Gradyan Artırma")
box(3.6, 2.6, 2.8, 0.95, "İki görev:\nİkili (Normal/Karanlık)\nDört sınıflı (Tor, VPN,\nNon-Tor, NonVPN)", fs=8.0)
box(0.35, 2.6, 2.75, 0.95, "Değerlendirme\nDoğruluk, Kesinlik,\nDuyarlılık, F1, AUC")
arrow(6.9, 3.07, 6.4, 3.07)
arrow(3.6, 3.07, 3.1, 3.07)
arrow(1.7, 2.6, 1.7, 1.85)

# Alt sira: onem analizi -> indirgeme -> yorumlanabilirlik
box(0.35, 0.75, 2.9, 1.0, "Öznitelik Önemi\nGini ve permütasyon\nanalizi")
box(3.75, 0.75, 2.9, 1.0, "Öznitelik İndirgeme\nİlk 10 öznitelik ile\nyeniden eğitim")
box(7.1, 0.75, 2.6, 1.0, "Yorumlanabilirlik\nve literatürle\nkarşılaştırma", fc="#FDECEA", ec="#C44E52")
arrow(3.25, 1.25, 3.75, 1.25)
arrow(6.65, 1.25, 7.1, 1.25)

os.makedirs("figs", exist_ok=True)
fig.savefig("figs/sekil0_akis.png", bbox_inches="tight")
print("figs/sekil0_akis.png kaydedildi")
