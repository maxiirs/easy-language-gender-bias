# Forschungsdesign: Multimodale Bias-Analyse (Text-zu-Bild)
**Projekt:** Gender Bias in Leichter Sprache vs. Standardsprache (easy-language-gender-bias)

## 1. Theoretischer Rahmen & Forschungsfrage
Die Untersuchung folgt einem **mehrstufigen Validierungsansatz**. Ziel ist es nachzuweisen, ob geschlechtsspezifische Stereotype (z. B. Karriere/Kompetenz vs. Familie/Wärme) in der Leichten Sprache stärker ausgeprägt sind als in der Standardsprache und ob sich dieser linguistische Bias in der visuellen Modalität (Text-to-Image) manifestiert oder sogar verstärkt.

## 2. Phase 1: Textuelle Bias-Exploration (Abgeschlossen)
Bevor Bilddaten generiert werden, wurde der "Linguistic Ground Truth" isoliert betrachtet. Dies ist essenziell, um die Ursache späterer visueller Verzerrungen eindeutig der Sprache zuordnen zu können.
* **Pipeline A (WEAT/ML-EAT):** Quantifizierung der impliziten Assoziationen in statischen Word Embeddings (`spaCy`).
* **Pipeline B (Local LLM):** Zero-Shot Klassifikation mittels `Llama-3.1-8B`, um kontextuelle Verzerrungen in komplexen Sätzen zu messen.
* **Erkenntnisgewinn:** Diese Phase liefert die Baseline-Werte (Text-Bias) des gesamten Korpus.

## 3. Phase 2: Visueller Transfer-Test (In Vorbereitung)
Um den modalitätsübergreifenden Bias unverfälscht zu messen, wird die Pipeline um eine visuelle Komponente erweitert:
1.  **Prompt-Generierung (Exhaustive Sampling):** Um einen Selektionsbias (Cherry-Picking) zu vermeiden, werden *alle* Sätze aus der LLM-Analyse in Prompts überführt. Die Grundgesamtheit des Textkorpus wird somit 1:1 in die Bildgenerierung übernommen.
2.  **Bildgenerierung:** Massengenerierung (z. B. n=100 pro Prompt) mittels **Stable Diffusion**, wobei die originalen, neutralen Akteurs-Bezeichnungen beibehalten werden.
3.  **Automatisierte Klassifikation (MiVOLO):** Einsatz des State-of-the-Art Modells **MiVOLO** (Multi-input Transformer) auf einer **Nvidia Tesla V100**. 
    * **Vorteil:** MiVOLO kombiniert Gesichts- und Körpermerkmale, was die Analyse von KI-generierten Bildern (die oft Artefakte in Gesichtern aufweisen) hochgradig robust macht.
    * **Output:** Quantitative Daten zur Geschlechterverteilung (M/F) und Confidence-Scores pro Bild.

## 4. Wissenschaftlicher Mehrwert dieses Designs
Das Design geht über eine einfache Bildanalyse hinaus, da es die **Kausalkette** abbildet:
* **Isolierung:** Durch den Vorab-Check der Texte wird sichergestellt, dass der Bias nicht erst im Bildgenerator entsteht, sondern bereits in der Sprache angelegt ist.
* **Verzerrungsfreiheit:** Das Exhaustive Sampling garantiert, dass das volle Spektrum (von extrem stereotypen bis zu völlig neutralen Sätzen) repräsentativ getestet wird.
* **Amplification-Messung:** Es lässt sich quantitativ bestimmen, ob der Bildgenerator den bestehenden Text-Bias lediglich reproduziert, abschwächt oder durch "Winner-takes-all"-Effekte signifikant verstärkt.
