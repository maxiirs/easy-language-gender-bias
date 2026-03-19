# Reference Pipeline Adaptation

## Kurzfazit

Die Referenzpipeline aus `reference_pipeline/code` ist fuer das neue Projekt nur teilweise direkt uebernehmbar. Der groesste Mehrwert liegt in der actor-level Logik:

- Actor-Extraktion
- Pronomen-Zuordnung
- Predication-Extraktion
- Knowledgebase-Struktur

Weniger passend oder problematisch sind:

- jahrbasierte Korpuslogik
- sentiment-lastige Auswertung als Hauptsignal
- newspaper-spezifische Aggregation
- mehrere heuristische Gender-Wortlisten

## Dateiweise Bewertung

### `main.py`

Status: `teilweise uebernehmen`

Nutzbar:

- Grundidee einer mehrstufigen Pipeline
- Initialisierung einer spaCy-Pipeline mit `merge_entities` und `coreferee`
- getrennte Verarbeitung von Einzeltexten und spaeterer Aggregation

Nicht passend:

- erwartet ein jahrbasiertes Zeitungskorpus unter `corpus/<year>`
- schreibt pro Artikel Pickle-Knowledgebases in eine temporale Ordnerstruktur
- ist auf Bulk-Verarbeitung ueber viele JSON-Dateien ausgelegt, nicht auf Parallelpaare `simple_text` / `normal_text`

Empfehlung:

- nur die Pipeline-Idee und die spaCy-Initialisierung als Vorlage nutzen
- keine direkte Uebernahme des Orchestrierungs-Codes

### `data_processing.py`

Status: `teilweise uebernehmen`

Nutzbar:

- Laden und Speichern von Zwischenergebnissen
- Idee, pro Dokument eine strukturierte Knowledgebase abzulegen

Nicht passend:

- `load_texts_from_file()` erwartet ein sehr spezielles JSON-Schema mit `text.text`
- `save_knowledgebase()` ist auf Pickle-basierten Artikel-Workflow zugeschnitten
- in `load_knowledgebase()` wird `pd.DataFrame()` verwendet, aber `pandas` nicht importiert

Empfehlung:

- Speichern/Laden nur als Muster verwenden
- fuer unser Projekt lieber CSV/Parquet auf Paar- und Actor-Ebene statt Pickle pro Artikel

### `analysis.py`

Status: `gezielt adaptieren`

Das ist die wertvollste Datei fuer dein Projekt.

Direkt interessant:

- `get_actors()`
- `combine_names()`
- `get_pronouns_main()`
- `build_knowledgebase_nomination()`
- `extract_predications()`

Warum:

- diese Funktionen bilden genau die actor-level Perspektive aus dem Paper ab
- sie passen gut zu einer spaCy-basierten Gender-Bias-Pipeline
- sie helfen uns, von blosem Wortzaehlen zu einer Analyse von Akteuren und ihren Beschreibungen zu kommen

Mit Vorsicht:

- `get_generic_names()` arbeitet nur mit sehr kleinen generischen Listen
- `detect_gender_neutral_language()` und `detect_generic_masculine()` sind sehr grobe Dokumentheuristiken
- `count_gendered_words()` ist fuer deutsche Morphologie zu simpel
- `detect_sentiment()` zieht ein externes Transformer-Modell nach und macht die Pipeline schwerer

Fuer unser Projekt aendern:

- auf Parallelpaare statt Einzelartikel umstellen
- Actor-Knowledgebase pro `pair_id` und `side` speichern
- Pronomenklassen nicht nur `she_her` und `he_him`, sondern zusaetzlich `unknown` fuehren
- Rollen-/Berufsmarker sauber von Pronomen trennen
- generisches Maskulinum und inklusive Formen nicht dokumentweit, sondern markerbasiert extrahieren

### `utils.py`

Status: `eher als Vorlage`

Nutzbar:

- Aggregationsidee ueber viele Einzel-Knowledgebases
- Trennung zwischen individueller und aggregierter Auswertung

Nicht passend:

- starke Ausrichtung auf das Pronomenpaar `she_her` vs. `he_him`
- mehrere Kennzahlen sind newspaper-spezifisch aggregiert
- Visualisierung und Report-Format sind auf deren Ergebnisdarstellung zugeschnitten
- kleinere Inkonsistenzen in Schluesseln und Variablennamen

Empfehlung:

- nur die Strukturidee uebernehmen
- Auswertungslogik fuer unser Projekt neu schreiben: Vergleich `normal` vs. `simple` pro Paar

### `visualisation.py`

Status: `nicht uebernehmen`

Gruende:

- vollstaendig auf jahresbasierte Zeitreihen fuer Zeitungsdaten ausgelegt
- arbeitet direkt auf deren Ergebnisordnern und Pickles
- fuer dein Vorhaben sind Paarvergleiche, Quellenvergleiche und spaetere Prompt-Analysen wichtiger als Jahresplots

Empfehlung:

- neu schreiben

### `setup/requirements.txt`

Status: `teilweise uebernehmen`

Wichtig:

- `spacy==3.5.0`
- `coreferee==1.4.1`
- `transformers`
- `torch`

Nicht zwingend fuer den ersten Schritt:

- komplette Visualisierungs- und Sentiment-Kette
- `spacytextblob`

Empfehlung:

- spaeter eine eigene, schlankere `requirements.txt` fuer das neue Projekt bauen

## Konkrete Uebernahmekandidaten

Diese Bausteine sind die besten Kandidaten fuer eine Adaption in das neue Projekt:

1. Actor-Extraktion ueber `PER`-Entities plus generische Personenbezeichnungen
2. Zusammenfuehren von Nominationsformen ueber `combine_names()`
3. Pronomenzuordnung ueber Coreference-Information
4. Predication-Extraktion ueber Saetze, in denen Actor-Nennungen oder zugeordnete Pronomen vorkommen
5. Knowledgebase-Konzept: ein strukturierter Actor-Datensatz pro Text

## Nicht direkt uebernehmen

1. Jahrordner-Logik in `main.py`
2. Zeitreihenplots in `visualisation.py`
3. kleine Listen fuer maskulin/feminin codierte Woerter als Hauptmethode
4. Dokumentweite Heuristik `majority gender neutral`
5. reine Sentimentanalyse als zentrales Bias-Mass

## Empfohlene Zielarchitektur fuer das neue Projekt

1. `build_pair_dataset.py`
   Laedt `simple-german-articles.csv` und erzeugt den Arbeitsdatensatz.

2. `run_spacy_pipeline.py`
   Annotiert `simple_text` und `normal_text` mit spaCy.

3. `extract_actors.py`
   Adaptiert die guten Teile aus `analysis.py` fuer Actor-Extraktion und Pronomenzuordnung.

4. `extract_gender_markers.py`
   Extrahiert Pronomen, Rollen, Berufsbezeichnungen, inklusive Formen, Paarformen und generisches Maskulinum.

5. `aggregate_pair_bias.py`
   Vergleicht Marker zwischen `simple` und `normal` pro Paar.

## Gesamtbewertung

Die Referenzpipeline ist als Vorlage sehr nuetzlich, aber nicht als Plug-and-Play-Loesung. Fuer dein Projekt sollten wir vor allem die actor-level Kernlogik aus `analysis.py` adaptieren und den Rest neu auf Paralleltexte und Bias-Vergleiche zwischen Leichter Sprache und Standardsprache zuschneiden.
