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
| `analiz.py` | Ana deney betiği. Ön işleme, ikili ve dört sınıflı görevlerde 6 modelin eğitimi, AUC, sınıf bazlı metrikler, karışıklık matrisi, Gini + permütasyon öznitelik önemi, ilk 10 öznitelikle indirgenmiş model. Makaledeki Tablo 2 ve Tablo 5–8'in tüm değerlerini `results.json` dosyasına yazar; Şekil 2–5'i `figs/` klasörüne üretir. |
| `sekil_akis.py` | Makaledeki Şekil 1'i (yöntemin iş akışı şeması) üretir. Veri gerektirmez. |
| `requirements.txt` | Gerekli Python paketleri. |

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python sekil_akis.py    # Şekil 1 (iş akışı)
python analiz.py        # Tüm deneyler + Şekil 2-5 + results.json
```

`analiz.py` tipik bir masaüstü bilgisayarda yaklaşık 4 dakikada tamamlanır.
En uzun adımlar Rastgele Orman eğitimi (~30 sn/görev), k-NN tahmini (~26 sn/görev)
ve permütasyon önem analizidir (~35 sn).

## Deney Düzeni (Özet)

- **Ön işleme:** Flow ID, Src/Dst IP, Timestamp ve uygulama etiketi (`Label.1`)
  çıkarılır; sonsuz/eksik değerli 50 kayıt silinir; varyansı sıfır olan
  15 öznitelik elenir. Kalan: **158.566 akış, 64 öznitelik**.
- **Ayrım:** %70 eğitim / %30 test, katmanlı örnekleme, `random_state = 42`.
- **Ölçekleme:** StandardScaler yalnızca LR, k-NN ve GNB için; istatistikler
  sadece eğitim kümesinden hesaplanır.
- **Modeller:** Lojistik Regresyon (`max_iter=1000`), Gaussian Naive Bayes,
  k-NN (`k=5`), Karar Ağacı, Rastgele Orman (`n_estimators=100`),
  Histogram Tabanlı Gradyan Artırma (scikit-learn varsayılanları).
- **Görevler:** İkili (Normal / Karanlık) ve dört sınıflı
  (Tor, Non-Tor, VPN, NonVPN). Karanlık = Tor + VPN.
- **Yorumlanabilirlik:** Gini önemi (RO) ve permütasyon önemi
  (8.000 örneklik test alt kümesi, 5 tekrar, makro F1 ölçütü);
  Gini'ye göre ilk 10 öznitelikle RO'nun yeniden eğitimi.

## Tekrarlanabilirlik

Tüm rastgelelik kaynakları 42 tohumu ile sabitlenmiştir. Aynı veri dosyası ve
paket sürümleriyle makaledeki değerler birebir yeniden üretilebilir
(ör. ikili görevde HGA: %98,66 doğruluk, 0,9982 AUC; dört sınıflı görevde
HGA: %98,71 doğruluk; ilk 10 öznitelikli RO: %98,36 doğruluk).

Kullanılan ortam: Python 3.12, scikit-learn 1.8.0, pandas, numpy, matplotlib.
