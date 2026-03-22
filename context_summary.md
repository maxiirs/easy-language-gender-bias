# Projekt-Status: Gender Bias in Leichter Sprache (easy-language-gender-bias)

## 1. Ausgangslage & Ziel
Es soll der geschlechtsspezifische Bias in Korpora der "Leichten Sprache" im Vergleich zur "Standardsprache" gemessen werden. Die Ergebnisse dienen als Basis, um später zu testen, ob diese Verzerrungen in der Text-zu-Bild-Generierung (z. B. Stable Diffusion) reproduziert oder verstärkt werden.

## 2. Erreichte Meilensteine & Pipelines
Wir haben zwei parallele, State-of-the-Art (SotA) Bias-Analyse-Pipelines aufgebaut, die auf den extrahierten Akteuren und Prädikationen (`data/processed/actor_features.csv`) arbeiten. Beide Skripte befinden sich im Ordner `src/bias_analysis/`.

### Pipeline A: Statistische Vektoranalyse (WEAT & ML-EAT)
- **Skript:** `weat_analysis.py`
- **Methodik:** Nutzt statische Word Embeddings von `spaCy` (`de_core_news_lg`), um den Word Embedding Association Test (Level 1) und den Differential Bias (ML-EAT Level 2) zu berechnen.
- **SotA-Ground-Truth:** Verwendet offiziell übersetzte, kulturell evaluierte deutsche Wort-Sets von Kurpicz-Briki (2020) für "Karriere vs. Familie" und das Stereotype Content Model für "Kompetenz vs. Wärme".
- **Fallbacks:** Da in der aktuellen Extraktions-Tabelle keine Pronomen aufgelöst wurden (`main_pronoun_class` = unknown), nutzt das Skript eine robuste lexikalische Deduzierung (z.B. *der Kunde* -> maskulin, *die Bürgerin* -> feminin), um die Akteure zuzuordnen und geschlechtsneutrale Begriffe wie *Fachkraft* sauber herauszufiltern.
- **Output:** `weat_results.json` sowie generierte Verteilungs-Graphen (`data/processed/plots/`).

### Pipeline B: Generative AI (Local LLM Zero-Shot)
- **Skript:** `llm_bias_eval.py`
- **Methodik:** Eine parallele Evaluation durch ein lokales großes Sprachmodell (`meta-llama/Llama-3.1-8B-Instruct`) über die HuggingFace `transformers`-Pipeline.
- **Logik:** Die extrahierten Sätze werden direkt in einen Forced-Choice-Prompt eingespeist (z.B. "Wird diese Person eher in einen Karriere- oder Familien-Kontext gesetzt?"). Dies liefert eine direkte quantitative Vergleichs-Metrik (generative Klassifikation vs. Embedding-Distanz).
- **Setup:** Die `.env` wird automatisch ausgelesen (`HFTOKEN=...`), das Modell lädt auf GPUs via `accelerate` und `device_map="auto"`.
- **Output:** `llm_bias_results.json`

## 3. Aktueller technischer Stand
- Das Projekt läuft ressourcenschonend in einem Conda-Environment (`environment.yml`).
- Die Umgebung enthält alle nötigen Deep-Learning-Abhängigkeiten (`torch`, `transformers`, `accelerate`, `tqdm`), womit das Repo nahtlos auf den ressourcenstarken GPU-basierten Jupyter-Hub der Hochschule umgezogen ist.
- Der Ausführungslauf des Milliarden-Parameter-LLMs auf dem Uni-Cluster war erfolgreich, und die generierten Inference-Ergebnisse (`llm_bias_results.json`) wurden per Git Pull zurück auf den lokalen Rechner migriert (siehe aktuellster Git Commit).

## 4. Nächste Schritte (Next Steps)
- Die Ergebnisse auswerten: Was sagt der klassische WEAT-Baseline-Test aus, und inwiefern repliziert/verstärkt das LLaMA-Modell diesen Bias in der Leichten Sprache?
- Herausarbeiten der stärksten Stereotyp-Kontraste (Sätze) aus den Metriken, um daraus Prompts für den Text-to-Image-Transfer (z. B. Stable Diffusion) abzuleiten und den modalitätsübergreifenden Bias zu bewerten.
