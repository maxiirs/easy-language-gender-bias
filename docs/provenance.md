# Provenance

## Datenquellen

Die in diesem Ordner gesammelten Daten stammen aus den folgenden lokalen Repositories:

- `ger-gov-to-easy-lang-internal`
- `ger-gov-easy-lang`

## Uebernommene Dateien

### Hauptkorpus

- `data/raw/parallel/simple-german-articles.csv`
  Quelle: `ger-gov-to-easy-lang-internal/data/prepared/simple-german-articles.csv`

### Zusatzkorpus

- `data/raw/aligned/a_new_aligned_simple_german_corpus.csv`
  Quelle: `ger-gov-to-easy-lang-internal/data/raw/a_new_aligned_simple_german_corpus.csv`

### Crawl-Snapshots

- `data/raw/crawl_snapshots/*_data.json`
  Quelle: `ger-gov-to-easy-lang-internal/data/raw/`

### Bereits erzeugte Analyse-Artefakte

- `data/processed/bias_dataset_summary.json`
- `data/processed/bias_doc_features.csv`
- `data/processed/bias_pair_features.csv`
- `data/processed/actor_features.csv`
- `data/processed/actor_features_summary.json`
  Quelle: spaCy-basierte Vorverarbeitung aus `ger-gov-to-easy-lang-internal`

### Pipeline-Code

- `src/pipeline/build_bias_working_dataset.py`
  Quelle: `ger-gov-to-easy-lang-internal/scripts/build_bias_working_dataset.py`

- `src/pipeline/build_paired_dataset.py`
  Quelle: `ger-gov-easy-lang/scripts/build_paired_dataset.py`

- `src/bias_analysis/extract_actors.py`
  Quelle: neu fuer dieses Projekt erstellt, inhaltlich adaptiert aus der actor-level Logik der Referenzpipeline in `reference_pipeline/code/analysis.py`

## Paper-Repository

Der lokale Ordner `corpus_pipeline` existierte beim Einsammeln zwar als Git-Repository, hatte aber noch keine Commits und keine Dateien im Worktree. Deshalb wurde daraus noch kein Code uebernommen.
