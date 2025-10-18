# Specifikacija projekta: Klasifikacija emocija na slikama lica
## **Članovi tima**
- **Roman Minakov SV83/2023**
- **Vladimir Morozkin SV85/2023**
## 1. **Naziv teme**
Klasifikacija emocija na slikama lica
## **2. Definicija problema**
U savremenom društvu postoji potreba za automatizovanim sistemima koji mogu brzo i objektivno procenjivati emocionalno stanje osobe. Tradicionalne metode analize emocija zasnovane su na subjektivnom posmatranju, što ih čini sporim i nepouzdanim. Nedostatak automatskih rešenja otežava primenu analize emocija u oblastima gde je potrebna operativnost i skalabilnost, kao što su online obrazovanje, medicinska dijagnostika ili praćenje velikih grupa ljudi.
## **3. Motivacija i primene**
Prepoznavanje emocija sa lica ima praktičnu vrednost u različitim oblastima:
- Medicina i psihologija: sistemi mogu služiti kao pomoćni alat za stručnjake, beležeći i analizirajući emocionalne manifestacije pacijenata tokom vremena. Ovo pomaže boljoj proceni opšteg emocionalnog stanja i praćenju napretka terapije.  
- Bezbednost: tehnologije prepoznavanja emocija mogu u realnom vremenu identifikovati znakove agresije, panike ili drugih kritičnih emocionalnih stanja na mestima masovnog okupljanja.  
- Čovek-računar interfejsi (HCI): prepoznavanje emocija pomaže sistemima da se prilagode korisniku, čineći interakciju prirodnijom i intuitivnijom.
Razvijanje sistema za klasifikaciju emocija po slikama lica ima širok spektar primene i doprinosi stvaranju sigurnijih, efikasnijih i korisnijih tehnologija.
## **4. Skupovi podataka**
Koristićemo pristup sa više skupova podataka kako bismo poboljšali generalizaciju modela.

| **Skup podataka** | **FER2013** | **RAF-DB** |
| :--- | :--- | :--- |
| **Izvor** | [Kaggle Link](https://www.kaggle.com/datasets/msambare/fer2013) | [Kaggle Link](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset) |
| **Uzorci** | 35,887 | ~15,000 |
| **Rezolucija** | 48x48 (Nijanse sive) | ~100x100 (RGB) |
| **Okruženje** | Laboratorijski kontrolisano | Stvarni svet, sa interneta |
| **Ključni izazov** | Ozbiljna neravnoteža klasa | Veći kvalitet, naturalistički |
**Distribucija klasa:**

| **Emocija**     | **FER2013** | **RAF-DB** | **Primarni izazov**                      |
| :-------------- | :---------- | :--------- | :--------------------------------------- |
| **Bes**         | 4,953       | 1,619      | Srednja prevalenca                       |
| **Gađenje**     | 547         | 355        | **Ekstremna neravnoteža**                |
| **Strah**       | 5,121       | 877        | Često mešano sa iznenađenjem             |
| **Sreća**       | 8,989       | 5,957      | Najzastupljenije, lakše za klasifikaciju |
| **Tuga**        | 6,077       | 2,460      | Srednja prevalenca                       |
| **Iznenađenje** | 4,002       | 867        | Često mešano sa strahom                  |
| **Neutralno**   | 6,198       | 3,204      | Srednja prevalenca                       |
# **5. Predobrade podataka**
1.  **Ujednačavanje i čišćenje:**
    -   Mapiranje labela emocija iz oba skupa podataka na zajedničku šemu od 7 klasa.
    -   Provera i uklanjanje oštećenih ili pogrešno obeleženih slika (ako postoje).
2.  **Razdvajanje podataka:**
    -   **Početna podela:** Pre bilo kakve obrade, svaki skup podataka (FER2013 i RAF-DB) se **nezavisno** deli na skup za obučavanje, validacioni i test skup.
    -   **Proporcije:** 
        -   **Skup za obučavanje:** 70% originalnih podataka
        -   **Validacioni skup:** 15% originalnih podataka
        -   **Test skup:** 15% originalnih podataka
    -   **Kombinovanje:** Nakon podela, originalni podaci iz FER2013 i RAF-DB se kombinuju po skupovima:
        -   **Konačni Train skup:** FER2013_train + RAF-DB_train
        -   **Konačni Validation skup:** FER2013_val + RAF-DB_val
        -   **Konačni Test skup:** FER2013_test + RAF-DB_test
    -   **Cilj:** Ovo obezbeđuje da validacioni i test skupovi sadrže isključivo originalne, neaugmentisane podatke i predstavljaju čist merljiv benchmark za model.
3.  **Standardizacija:**
    -   **Promena veličine:** Sve slike u **sva tri skupa** će biti promenjene na ujednačenu veličinu od `128x128` piksela.
    -   **Konverzija boje:** Sive slike iz FER2013 skupa će biti konvertovane u RGB dodavanjem kanala kako bi se osigurala doslednost ulaza.
    -   **Normalizacija:** Vrednosti piksela će biti skalirane na opseg `[0, 1]` deljenjem sa `255.0`.
4.  **Strategija za balansiranje klasa:**
    -   **Primarna tehnika: Augmentacija za nadorkanjivanje.** Kako bismo rešili neravnotežu, agresivna augmentacija podataka se primenjuje **isključivo na manjinske klase unutar KONAČNOG SKUPA ZA OBUČAVANJE**.
    -   **Tehnike augmentacije:** Nasumično horizontalno preslikavanje, rotacija (±15°), pomeranje po širini/visini (±10%), zumiranje (±5%) i prilagođavanje osvetljenija (±20%). Ove transformacije se primenjuju u realnom vremenu tokom obuke ili se unapred generišu dodatni augmentisani uzorci samo za trening skup.
    -   **Važna napomena:** Validacioni i test skupovi **nikada** ne podležu augmentaciji. Oni se koriste u svojoj originalnoj, standardizovanoj formi kako bi evaluacija bila fer i korektna.
## **6. Metodologija**
### Faza 1: Osnovni model - Pojednostavljena CNN arhitektura (Baseline)
-   **Svrha:** Uspostavljanje performansne osnove (baseline) i verifikacija ispravnosti pipeline-a za obuku. Ovaj model se obučava brzo i služi kao referenta tačka za upoređivanje sa naprednijim ResNet modelom.
-   **Arhitektura:** Sekvencijalni model sa 2-3 konvoluciona bloka.
    -   **Blokovi:** Svaki blok se sastoji od:
        -   Jednog `Conv2D` sloja sa `ReLU` aktivacionom funkcijom. Broj filtera raste sa svakim blokom: **32 -> 64** (za 3 bloka: **32 -> 64 -> 128**).
        -   Jednog `MaxPooling2D` sloja za smanjenje prostorne dimenzionalnosti.
        -   `Dropout` (0.2 - 0.3) nakon pooling sloja za regularizaciju i smanjenje preprilagođavanja.
    -   **Glava:** `Flatten` -> `Dense(64, ReLU)` -> `Dropout(0.5)` -> `Dense(7, Softmax)`.
### Faza 2: Napredni model - Transfer učenja sa unapred obučenim ResNet50
-   **Arhitektura:**
    -   **Bazni model:** `ResNet50` unapred obučen na ImageNet, sa zamrznutim težinama. Ovo iskorišćava već naučene detektore karakteristika.
    -   **Glava:** Prilagođeni klasifikacioni slojevi na vrhu:
        -   `GlobalAveragePooling2D` -> `BatchNormalization` -> `Dense(256, ReLU)` -> `Dropout(0.5)` -> `Dense(7, Softmax)`.
-   **Fine-Tuning:** Nakon što se prilagođena glava obuči, gornji slojevi baznog modela mogu biti odmrznutí i obučeni sa veoma niskom stopom učenja za dalju doradu.
## **7. Plan evaluacije**
**1. Primarne metrike:**
-   **Ukupna tačnost:** Standardna mera ukupno tačnih predviđanja.
-   **Ponderisani F1-score:** **Ključna metrika** zbog neravnoteže klasa. Pruža balans između preciznosti i odziva za svaku klasu, ponderisan podrškom.
-   **Macro F1-score:** Dodatna ključna metrika koja daje jednaku težinu svakoj klasi, neophodna za evaluaciju performansi na manjinskim klasama (npr. "Gađenje").
-   **Preciznost, odziv i F1-score po klasama:** Za dijagnosticiranje performansi na svakoj specifičnoj emociji.
**2. Sveobuhvatna analiza generalizacije:**
Kako bismo detailno procenili robustnost modela, pored standardne evaluacije na celom test skupu, sprovedećemo i **analizu po domenu**:
-   **Ukupne performanse:** Model će biti evaluiran na celokupnom test skupu (FER2013_test + RAF-DB_test) kako bi se dobila opšta slika tačnosti.
-   **Analiza po izvoru podataka:** Zatim će se test skup podeliti na njegove originalne komponente kako bi se analizirale performanse modela na **svakom skupu podataka posebno:**
    -   **Performanse na FER2013 test skupu:** Ovo meri efikasnost modela na laboratorijski kontrolisanim, crno-belim slikama.
    -   **Performanse na RAF-DB test skupu:** Ovo meri sposobnost modela da generalizuje na slikama "iz stvarnog sveta" (in-the-wild) sa prirodnim pozadinama, osvetljenjem i bojom.
-   **Očekivani ishod i implikacije:** Razlika u performansama između ova dva test skupa će direktno ukazati na **jaz u domenu** (domain gap). Visoke performanse na oba skupa ukazuju na robustan model. Značajan pad performansi na RAF-DB skupu ukazao bi na to da model, iako dobar za laboratorijske uslove, ima poteškoća sa generalizacijom u realnijim scenarijima.
**3. Vizuelna analiza:**
-   **Krive učenja:** Crtanje gubitka/tačnosti na skupu za obučavanje vs. validacioni skup za otkrivanje preprilagođavanja/podprilagođavanja.
-   **Normalizovana matrica konfuzije:** Generisaće se dve heatmape: jedna za ceo test skup, a posebno jedna za RAF-DB test skup kako bi se identifikovali specifični obrasci pogrešne klasifikacije karakteristični za realan svet (npr. da li se "Strah" obično predviđa kao "Iznenađenje"?).
-   **Analiza grešaka:** Prikaz pogrešno klasifikovanih slika sa tačnim i predviđenim labelama za kvalitativno razumevanje mana modela (npr. okluzije, ekstremni uglovi, dvosmisleni izrazi). Ova analiza će se posebno fokusirati na greške napravljene na RAF-DB skupu.
**4. Poređenje sa osnovom:**
-   Uporediti performanse modela sa **nasumičnim klasifikatorom (~14.3% tačnosti)** i **klasifikatorom većinske klase (uvek predviđa "Sreća")**.
## **8. Tehnološki stek**
-   **Programski jezik:** Python 3.10+
-   **Okvir za duboko učenje:** TensorFlow
-   **Manipulacija podacima:** NumPy, Pandas
-   **Obrada slike:** OpenCV
-   **Vizuelizacija:** Matplotlib, Seaborn
-   **Alati za mašinsko učenje:** Scikit-learn, Imbalanced-learn (imblearn)
## **9. Relevantna literatura**
1.  Goodfellow, I., et al. (2013). ["Challenges in Representation Learning: A Report on the Machine Learning Contest on Facial Expression Recognition."](https://arxiv.org/abs/1307.0414) *arXiv preprint arXiv:1307.0414*.
2.  Li, S., Deng, W. (2018). ["Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild."](https://ieeexplore.ieee.org/document/8099760) *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
3.  He, K., et al. (2016). ["Deep Residual Learning for Image Recognition."](https://arxiv.org/abs/1512.03385) *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
4.  Simonyan, K., Zisserman, A. (2014). ["Very Deep Convolutional Networks for Large-Scale Image Recognition."](https://arxiv.org/abs/1409.1556) *arXiv preprint arXiv:1409.1556*. 
5.  Chawla, N. V., et al. (2002). ["SMOTE: Synthetic Minority Over-sampling Technique."](https://www.jair.org/index.php/jair/article/view/10302) *Journal of Artificial Intelligence Research*, 16, 321-357.
6.  Kampel, M et al. (2016). ["Facial Expression Recognition using Convolutional Neural Networks: State of the Art."](https://arxiv.org/abs/1612.02903) *arXiv preprint arXiv:1612.02903*.