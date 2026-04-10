# Person Lexicon Resources

`person_lexicon.csv` externalizes lexical assumptions that should not be hidden
inside Python code. Each row must document at least:

- `lemma`
- `class`
- `source`
- `license`
- `notes`

Recommended classes:

- `explicit_female`
- `explicit_male`
- `collective`
- `person_hint`

Current status:

- The current file is a conservative project seed.
- It is intended as a temporary, auditable resource for development.
- For a final scientific analysis, replace or extend it with entries extracted
  from versioned open resources and keep the source and license columns filled.

Recommended upstream resources:

- German Wiktionary dump for lexical entries and feminine/masculine counterpart
  extraction.
- IWNLP for Wiktionary-derived German lemma and morphology resources.
- DEMorphy for German morphological analysis and lexical normalization.
- Wikidata for name metadata and structured person/name information.

Recommended workflow:

1. Freeze one resource version per experiment.
2. Store source URLs, dump dates, and licenses in the CSV rows or a companion
   metadata file.
3. Keep project-specific manual additions separate from imported entries.
4. Report lexicon provenance in the method section.
