# Person Lexicon Resources

`kaikki_de_lookup.jsonl` is a local cache of Kaikki.org entries for the lemmas
that actually occur in the current annotated corpus. The goal is to avoid
manual lemma lists in Python code while keeping the lexical lookup reproducible.

Current status:

- The intended primary source is Kaikki.org machine-readable data extracted from
  the German Wiktionary.
- Collective/group overrides can additionally be derived from Wikidata and
  stored in a separate local cache.
- The local cache should be rebuilt from the current annotated corpus whenever
  the corpus changes substantially.
- Do not maintain a hidden manual fallback list in Python code.

Recommended workflow:

1. Run `src/pipeline/annotate_texts.py`.
2. Run `src/resources/build_kaikki_lookup.py`.
3. Optionally run `src/resources/build_wikidata_collectives.py`.
4. Run `src/analysis/prepare_gender_data.py`.
5. Report the Kaikki source URL and retrieval timestamp from the metadata file
   in the method section.

Kaikki cache build:

- Use [build_kaikki_lookup.py](/home/max/projects/easy-language-gender-bias/src/resources/build_kaikki_lookup.py)
  to collect the lemmas from `annotated_texts.json`, download the current
  German Kaikki raw data, and write a local filtered JSONL cache.
- The script writes:
  - `kaikki_de_lookup.jsonl`
  - `kaikki_de_lookup_metadata.json`

Example:

```bash
python src/resources/build_kaikki_lookup.py
python src/resources/build_wikidata_collectives.py
python src/analysis/prepare_gender_data.py
```
