# easy-language-gender-bias

Arbeitsordner fuer ein eigenstaendiges Folgeprojekt zur Untersuchung von Gender-Bias in deutscher Leichter Sprache vs. Standardsprache sowie zum spaeteren Transfer auf Text-to-Image-Prompts.

## Inhalt

- `data/raw/parallel`: Hauptkorpus mit Paralleltexten aus Standardsprache und Leichter Sprache
- `data/raw/aligned`: zusaetzliches breiteres aligned Korpus
- `data/raw/crawl_snapshots`: Roh-Crawl-Ausgaben aus dem internen Projektstand
- `data/processed`: erste spaCy-basierte Arbeitsdaten fuer die Bias-Analyse
- `data/processed/actor_features.csv`: actor-level Tabelle mit Nominations- und Predication-Information
- `src/pipeline`: bisherige Datenaufbereitungs- und Filter-Skripte
- `src/bias_analysis`: neue Analyseschicht fuer actor-level und Gender-Bias-Extraktion
- `src/adapted_from_paper`: Platz fuer adaptierten Code aus dem Paper-Repo
- `docs`: Projekt- und Provenance-Notizen

## Aktuell wichtigste Dateien

- `data/raw/parallel/simple-german-articles.csv`
- `data/processed/bias_doc_features.csv`
- `data/processed/bias_pair_features.csv`
- `data/processed/actor_features.csv`
- `src/pipeline/build_bias_working_dataset.py`
- `src/bias_analysis/extract_actors.py`

## Hinweis zum Paper-Repo

Der lokale Klon von `corpus_pipeline` war beim Einsammeln dieses Arbeitsordners leer und enthielt noch keine Commits. Deshalb wurde daraus noch kein Code uebernommen. Die vorgesehene Zielstelle fuer spaetere Adaptionen ist `src/adapted_from_paper/`.
