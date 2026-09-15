# Karanlık Ağ ve VPN Trafiği Tespiti — Analiz Kodları

Bu depo, *"Karanlık Ağ ve VPN Trafiği Tespiti: Denetimli Öğrenme Modellerinde
Performans, Özellik Önemi ve Yorumlanabilirlik Analizi"* başlıklı makalenin
tüm deneysel sonuçlarını, tablolarını ve şekillerini üreten Python kodlarını
içermektedir.

## Veri Seti

Çalışmada kamuya açık **CIC-Darknet2020** veri seti kullanılmıştır:
https://www.unb.ca/cic/datasets/darknet2020.html

Veri seti telif nedeniyle depoya eklenmemiştir. İndirdiğiniz CSV dosyasını
`Darknet.CSV` adıyla bu klasöre koyunuz (158.616 akış, 85 sütun).

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `analiz.py` | Ana deney betiği. Makaledeki Tablo 2 ve Tablo 5–15'in tüm değerlerini `results.json` dosyasına yazar; Şekil 2–7'yi `figs/` klasörüne üretir. |
| `sekil_akis.py` | Makaledeki Şekil 1'i (yöntemin iş akışı şeması) üretir. Veri gerektirmez. |
| `requirements.txt` | Gerekli Python paketleri. |

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python sekil_akis.py    # Şekil 1 (iş akışı)
python analiz.py        # Tüm deneyler + Şekil 2-7 + results.json
```

Koşu yarıda kesilirse, tamamlanmış aşamalar `results.json` dosyasında saklandığı
için baştan başlamaya gerek yoktur:

```bash
python analiz.py --devam   # tamamlanan aşamaları atlar, kaldığı yerden sürdürür
```

`--devam` kipinde yalnızca eksik aşamalar hesaplanır; ara çıktılar `ara/`
klasöründe tutulur.

### Çalışma süresi

Tek çekirdekli bir makinede tüm deneyler yaklaşık **32 dakika** sürmektedir.
Çok çekirdekli bir masaüstünde bu süre belirgin biçimde kısalır; Rastgele Orman,
k-NN ve permütasyon analizi `n_jobs=-1` ile paralel çalışmaktadır. En uzun
adımlar sırasıyla hiperparametre araması (~11 dk), beş katlı çapraz doğrulama
(~13 dk) ve grup bazlı ayrımdır (~4 dk).

## Betiğin Aşamaları

`analiz.py` dosya içinde A–K harfleriyle bölümlenmiştir. Her aşamanın makaledeki
karşılığı aşağıdadır.

| Aşama | İçerik | Makale karşılığı |
|---|---|---|
| A | Veri ön işleme | Tablo 2, Tablo 3 |
| B | İkili sınıflandırma + ROC-AUC + PR-AUC | Tablo 5, Şekil 3 |
| C | Dört sınıflı sınıflandırma + sınıf bazlı metrikler | Tablo 6, Tablo 10, Şekil 5(a) |
| D | McNemar testleri (yedi model çifti, iki görev) | Tablo 8 |
| E | Beş katlı katmanlı çapraz doğrulama | Tablo 7, Şekil 4 |
| F | Maliyet duyarlı öğrenme ve SMOTE | Tablo 11 |
| G | Grup bazlı ayrım (kaynak-hedef IP çifti) | Tablo 12, Şekil 5(b) |
| H | Gini ve permütasyon öznitelik önemi | Tablo 13, Şekil 6(a,b) |
| I | SHAP analizi ve sıralama uyumu | Tablo 13, Tablo 14, Şekil 6(c), Şekil 7 |
| J | Öznitelik indirgeme (ilk 10 öznitelik) | Tablo 15 |
| K | Hiperparametre eniyilemesi (rastgele arama) | Tablo 9 |

## Deney Düzeni (Özet)

- **Ön işleme:** Flow ID, Src/Dst IP, Timestamp ve uygulama etiketi (`Label.1`)
  çıkarılır; sonsuz/eksik değerli 50 kayıt silinir; varyansı sıfır olan
  15 öznitelik elenir. Kalan: **158.566 akış, 64 öznitelik**.
- **Görevler:** İkili (Normal / Karanlık) ve dört sınıflı
  (Tor, Non-Tor, VPN, NonVPN). Karanlık = Tor + VPN.
- **Modeller:** Lojistik Regresyon (`max_iter=1000`), Gaussian Naive Bayes,
  k-NN (`k=5`), Karar Ağacı, Rastgele Orman (`n_estimators=100`),
  Histogram Tabanlı Gradyan Artırma (scikit-learn varsayılanları).
- **Doğrulama düzenleri:** (1) katmanlı %70/%30 ayrım, (2) beş katlı katmanlı
  çapraz doğrulama, (3) kaynak-hedef IP çiftine göre grup bazlı ayrım
  (12.469 grup; eğitim ve test kümeleri arasında ortak çift yoktur).
- **Ölçekleme:** StandardScaler yalnızca LR, k-NN ve GNB için; istatistikler
  sadece eğitim kümesinden (çapraz doğrulamada her kat için ayrı) hesaplanır.
- **İstatistiksel sınama:** McNemar testi, süreklilik düzeltmeli ki-kare;
  uyuşmazlık sayısı 25'in altındaysa tam binom testi.
- **Sınıf dengesizliği:** Sınıf/örnek ağırlığı ile maliyet duyarlı öğrenme ve
  SMOTE (yalnızca eğitim kümesine uygulanır, test kümesi özgün bırakılır).
- **Yorumlanabilirlik:** Gini önemi (RO), permütasyon önemi (8.000 örneklik
  test alt kümesi, 5 tekrar, makro F1 ölçütü) ve SHAP (HGA modeli üzerinde
  ağaç tabanlı kesin algoritma, 3.000 akış, sınıf bazlı katkılar). Üç
  sıralama arasındaki uyum Spearman, Kendall tau ve üst sıra örtüşmesiyle
  nicel olarak ölçülür.
- **Öznitelik indirgeme:** Gini ve SHAP sıralamalarındaki ilk 10 öznitelikle
  modellerin yeniden eğitimi.

## Çıktılar

`results.json` dosyası tüm sayısal sonuçları içerir. Başlıca anahtarlar:
`veri`, `ikili`, `dort_sinif`, `sinif_bazli_RO`, `sinif_bazli_HGA`,
`karisiklik_HGA`, `mcnemar_dort_sinif`, `mcnemar_ikili`, `cd_ikili`,
`cd_dort_sinif`, `dengeleme`, `grup_dort_sinif`, `grup_ikili`, `onem_gini`,
`onem_permutasyon`, `onem_shap`, `onem_shap_sinif_bazli`, `siralama_uyumu`,
`siralama_ortusme`, `RO_gini_ilk10`, `HGA_gini_ilk10`, `HGA_shap_ilk10`,
`hiperparametre`.

`figs/` klasöründeki şekiller makaledeki numaralarla eşleşir:
`sekil1_akis.png`, `sekil2_dagilim.png`, `sekil3_roc_pr.png`, `sekil4_f1.png`,
`sekil5_karisiklik.png`, `sekil6_onem.png`, `sekil7_shap_sinif.png`.

## Tekrarlanabilirlik

Tüm rastgelelik kaynakları 42 tohumu ile sabitlenmiştir. Aynı veri dosyası ve
paket sürümleriyle makaledeki değerler birebir yeniden üretilebilir. Kontrol
için birkaç referans değer:

| Sonuç | Değer |
|---|---|
| İkili, HGA | %98,66 doğruluk; 0,9982 ROC-AUC; 0,9910 PR-AUC |
| Dört sınıf, HGA | %98,71 doğruluk; 0,9651 makro F1 |
| Çapraz doğrulama, HGA | ikili %98,64±0,02; dört sınıf %98,82±0,07 |
| Grup bazlı ayrım, HGA | ikili %97,64; dört sınıf %97,77 |
| McNemar, HGA–RO (dört sınıf) | b=264, c=148, χ²=32,10, p=1,5×10⁻⁸ |
| Hiperparametre eniyilemesi, HGA | %98,72 doğruluk; 0,9654 makro F1 |
| SHAP ilk 10 öznitelikli HGA | %98,62 doğruluk |
| Gini–permütasyon sıra uyumu | Spearman ρ = −0,007 (p = 0,954) |

Kullanılan ortam: Python 3.12, scikit-learn 1.8.0, shap 0.52, imbalanced-learn
0.14, pandas, numpy, scipy, matplotlib.
