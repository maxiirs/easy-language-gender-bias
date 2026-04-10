#!/usr/bin/env python3
"""Extract actor-level knowledge from parallel Easy Language pairs.

This script adapts the core ideas of the reference newspaper pipeline to the
paired Easy Language corpus used in this project:

- actor extraction via named entities and generic person nouns
- optional pronoun linking via coreference if available
- predication extraction via sentences mentioning an actor

The output is an actor-level table with one row per actor mention cluster.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import spacy


DEFAULT_INPUT = Path("data/raw/parallel/simple-german-articles.csv")
DEFAULT_OUTPUT = Path("data/processed/actor_features.csv")
DEFAULT_SUMMARY_OUTPUT = Path("data/processed/actor_features_summary.json")

WHITESPACE_RE = re.compile(r"\s+")

FEMININE_PRONOUNS = {"sie", "ihr", "ihre", "ihren", "ihrem", "ihrer", "ihres", "deren", "der", "die", "derjenige", "diejenige"}
MASCULINE_PRONOUNS = {"er", "ihn", "ihm", "sein", "seine", "seinen", "seinem", "seiner", "seines", "dessen", "dem", "den", "der", "derjenige"}

GENERIC_PERSON_NOUNS = {
    # Original
    "person", "personen", "mensch", "menschen", "frau", "frauen", "mann", "männer",
    "bürger", "bürgerin", "bürgerinnen", "bürgern", "mitarbeiter", "mitarbeiterin", 
    "mitarbeitende", "mitarbeitenden", "fachkraft", "fachkräfte", "kraft", "kräfte",
    "lehrkraft", "lehrkräfte", "kundin", "kunde", "kunden", "patient", "patienten",
    "patientin", "patientinnen",
    # Added roles and generic nouns
    "teilnehmer", "teilnehmerin", "teilnehmerinnen", "teilnehmende", "teilnehmenden",
    "nutzer", "nutzerin", "nutzerinnen", "nutzende", "nutzenden",
    "besucher", "besucherin", "besucherinnen", "besuchende", "besuchenden",
    "leser", "leserin", "leserinnen", "lesende", "lesenden",
    "zuschauer", "zuschauerin", "zuschauerinnen", "zuschauende", "zuschauenden",
    "anwender", "anwenderin", "anwenderinnen", "anwendende", "anwendenden",
    "autor", "autorin", "autorinnen", "autoren",
    "sprecher", "sprecherin", "sprecherinnen", "sprechende", "sprechenden",
    "experte", "expertin", "expertinnen", "experten",
    "kollege", "kollegin", "kolleginnen", "kollegen",
    "partner", "partnerin", "partnerinnen", "partnern",
    "chef", "chefin", "chefinnen", "chefs",
    "vertreter", "vertreterin", "vertreterinnen", "vertretern",
    "schüler", "schülerin", "schülerinnen", "schülern",
    "student", "studentin", "studentinnen", "studenten", "studierende", "studierenden",
    "polizist", "polizistin", "polizistinnen", "polizisten",
    "arzt", "ärztin", "ärztinnen", "ärzte", "ärzten",
    "politiker", "politikerin", "politikerinnen", "politikern",
    "arbeitgeber", "arbeitgeberin", "arbeitgeberinnen", "arbeitgebern",
    "arbeitnehmer", "arbeitnehmerin", "arbeitnehmerinnen", "arbeitnehmern",
    "gast", "gäste", "gästen",
    "bewohner", "bewohnerin", "bewohnerinnen", "bewohnern",
    "fahrer", "fahrerin", "fahrerinnen", "fahrern",
    "spieler", "spielerin", "spielerinnen", "spielern",
    "forscher", "forscherin", "forscherinnen", "forschern",
    "wissenschaftler", "wissenschaftlerin", "wissenschaftlerinnen", "wissenschaftlern",
    "täter", "täterin", "täterinnen", "tätern",
    "opfer", "opfern",
    "angestellte", "angestellter", "angestellten",
    "beamte", "beamter", "beamtin", "beamtinnen", "beamten",
    "wähler", "wählerin", "wählerinnen", "wählern",
    "verbraucher", "verbraucherin", "verbraucherinnen", "verbrauchern",
    "kandidat", "kandidatin", "kandidatinnen", "kandidaten"
}



ENTITY_NOISE_PATTERNS = ("©", "stock.adobe", "pixel-shot", "|")


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def load_models(model: str, enable_coref: bool, use_gliner: bool, use_bert: bool, predict_gender: bool):
    nlp = spacy.load(model, disable=["textcat"])
    if "merge_entities" not in nlp.pipe_names:
        nlp.add_pipe("merge_entities")

    coref_available = False
    if enable_coref:
        try:
            import coreferee  # noqa: F401

            if "coreferee" not in nlp.pipe_names:
                nlp.add_pipe("coreferee")
            coref_available = True
        except Exception:
            coref_available = False

    gliner_model = None
    if use_gliner:
        try:
            from gliner import GLiNER
            gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        except ImportError:
            raise ImportError("Please install gliner using `pip install gliner` to use the --use-gliner flag.")

    bert_model = None
    if use_bert:
        try:
            from transformers import pipeline
            import torch
            device = 0 if (torch.backends.mps.is_available() or torch.cuda.is_available()) else -1
            bert_model = pipeline(
                "token-classification",
                model="CarlaSoe/personal-noun-detection-german-bert",
                aggregation_strategy="simple",
                device=device
            )
        except ImportError:
            raise ImportError("Please install transformers and torch using `pip install transformers torch` to use the --use-bert-person-nouns flag.")

    zero_shot_model = None
    if predict_gender:
        try:
            from transformers import pipeline
            import torch
            device = 0 if (torch.backends.mps.is_available() or torch.cuda.is_available()) else -1
            zero_shot_model = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                device=device
            )
        except ImportError:
            raise ImportError("Please install transformers and torch using `pip install transformers torch` to use the --predict-gender flag.")

    return nlp, coref_available, gliner_model, bert_model, zero_shot_model


def get_person_entity_actors(doc, gliner_model) -> dict[str, dict[str, object]]:
    actors: dict[str, dict[str, object]] = {}
    
    if gliner_model is not None:
        # Excluded "Person" to avoid Named Entities (Olaf Scholz). Focused strictly on generic roles/jobs.
        labels = ["Beruf", "Job Title", "Funktion", "Rolle"]
        entities = gliner_model.predict_entities(doc.text, labels, threshold=0.5)
        for ent in entities:
            # Try to map character span back to tokens
            span = doc.char_span(ent["start"], ent["end"], alignment_mode="contract")
            if span is None:
                continue
            
            # Stereotype Bias exclusion: We do not want PROPN (Proper Nouns / Real Individuals)
            if span.root.pos_ == "PROPN":
                continue
                
            label = span.text.strip()
            if not label or "'" in label:
                continue
            if any(pattern in label.lower() for pattern in ENTITY_NOISE_PATTERNS):
                continue
            
            actor = actors.setdefault(
                label,
                {
                    "actor_label": label,
                    "actor_type": "person_entity_gliner",
                    "nomination_tokens": [],
                    "nominations": set(),
                    "pronoun_tokens": [],
                    "sent_indices": set(),
                },
            )
            actor["nomination_tokens"].extend(list(span))
            actor["nominations"].add(label)
            actor["sent_indices"].add(span.sent.start)
    else:
        for ent in doc.ents:
            if ent.label_ != "PER":
                continue
            label = ent.text.strip()
            if not label or "'" in label:
                continue
            if any(pattern in label.lower() for pattern in ENTITY_NOISE_PATTERNS):
                continue
            alpha_tokens = [token for token in ent if token.is_alpha]
            if not alpha_tokens:
                continue
            if not any(token.pos_ == "PROPN" for token in alpha_tokens):
                continue
            actor = actors.setdefault(
                label,
                {
                    "actor_label": label,
                    "actor_type": "person_entity",
                    "nomination_tokens": [],
                    "nominations": set(),
                    "pronoun_tokens": [],
                    "sent_indices": set(),
                },
            )
            actor["nomination_tokens"].extend(list(ent))
            actor["nominations"].add(label)
            actor["sent_indices"].add(ent.sent.start)
    return actors


def get_generic_person_actors(doc, actors: dict[str, dict[str, object]], exclude_plurals: bool, bert_model) -> dict[str, dict[str, object]]:
    if bert_model is not None:
        try:
            results = bert_model(doc.text)
            for res in results:
                if res["entity_group"] == "PERS_N":
                    span = doc.char_span(res["start"], res["end"], alignment_mode="contract")
                    if span is None:
                        continue
                    if exclude_plurals and span.root.morph.get("Number") == ["Plur"]:
                        continue
                    label = span.text.strip().lower()
                    if not label or "'" in label:
                        continue
                    
                    actor = actors.setdefault(
                        label,
                        {
                            "actor_label": span.text.strip(),
                            "actor_type": "bert_person_noun",
                            "nomination_tokens": [],
                            "nominations": set(),
                            "pronoun_tokens": [],
                            "sent_indices": set(),
                        },
                    )
                    actor["nomination_tokens"].extend(list(span))
                    actor["nominations"].add(span.text.strip())
                    actor["sent_indices"].add(span.sent.start)
        except Exception as e:
            pass

    for token in doc:
        if not token.is_alpha:
            continue
        lemma = token.lemma_.lower().strip()
        if lemma not in GENERIC_PERSON_NOUNS:
            continue
            
        if exclude_plurals and token.morph.get("Number") == ["Plur"]:
            continue
            
        label = lemma
        actor = actors.setdefault(
            label,
            {
                "actor_label": label,
                "actor_type": "generic_person_noun",
                "nomination_tokens": [],
                "nominations": set(),
                "pronoun_tokens": [],
                "sent_indices": set(),
            },
        )
        actor["nomination_tokens"].append(token)
        actor["nominations"].add(token.text)
        actor["sent_indices"].add(token.sent.start)
    return actors


def combine_overlapping_actors(actors: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    combined = {key: value.copy() for key, value in actors.items()}
    for key in list(combined.keys()):
        for other_key in list(combined.keys()):
            if key == other_key:
                continue
            if key.lower() in other_key.lower() and key.lower() != other_key.lower():
                combined[other_key]["nomination_tokens"].extend(combined[key]["nomination_tokens"])
                combined[other_key]["nominations"].update(combined[key]["nominations"])
                combined[other_key]["sent_indices"].update(combined[key]["sent_indices"])
                combined.pop(key, None)
                break
    return combined


def assign_pronouns_with_coref(doc, actors: dict[str, dict[str, object]]) -> None:
    if not hasattr(doc._, "coref_chains"):
        return

    by_nomination = {}
    for actor_key, actor in actors.items():
        for token in actor["nomination_tokens"]:
            by_nomination[token.text] = actor_key

    for token in doc:
        if token.pos_ != "PRON":
            continue
        try:
            resolved = doc._.coref_chains.resolve(token)
        except Exception:
            resolved = None
        if not resolved or len(resolved) != 1:
            continue
        resolved_text = resolved[0].text
        actor_key = by_nomination.get(resolved_text)
        if not actor_key:
            continue
        actors[actor_key]["pronoun_tokens"].append(token)
        actors[actor_key]["sent_indices"].add(token.sent.start)


def infer_main_pronoun(pronoun_tokens: list[object]) -> str:
    if not pronoun_tokens:
        return "unknown"

    counts = Counter()
    for token in pronoun_tokens:
        lower = token.text.lower()
        if lower in FEMININE_PRONOUNS:
            counts["feminine"] += 1
        elif lower in MASCULINE_PRONOUNS:
            counts["masculine"] += 1
        else:
            counts["other"] += 1

    total = sum(counts.values())
    if not total:
        return "unknown"

    label, label_count = counts.most_common(1)[0]
    if label_count / total < 0.7:
        return "mixed_or_unclear"
    return label


def extract_predications(doc, actor: dict[str, object]) -> list[str]:
    sentences = []
    seen = set()
    for sent in doc.sents:
        if sent.start not in actor["sent_indices"]:
            continue
        sent_text = sent.text.strip()
        if sent_text and sent_text not in seen:
            seen.add(sent_text)
            sentences.append(sent_text)
    return sentences


def build_actor_rows(pair_id: str, side: str, row: dict[str, str], doc, coref_available: bool, gliner_model, exclude_plurals: bool, bert_model, zero_shot_model) -> list[dict[str, object]]:
    actors = get_person_entity_actors(doc, gliner_model)
    actors = get_generic_person_actors(doc, actors, exclude_plurals, bert_model)
    actors = combine_overlapping_actors(actors)
    if coref_available:
        assign_pronouns_with_coref(doc, actors)

    actor_rows = []
    sorted_actors = sorted(actors.values(), key=lambda item: item["actor_label"].lower())
    for index, actor in enumerate(sorted_actors, start=1):
        nominations = sorted(actor["nominations"])
        predications = extract_predications(doc, actor)
        main_pronoun = infer_main_pronoun(actor["pronoun_tokens"])

        predicted_gender = "not_predicted"
        prediction_score = 0.0
        
        if zero_shot_model is not None and predications:
            context_sentence = predications[0] 
            candidate_labels = ["männlich", "weiblich", "generisch"]
            hypothesis_template = f"Das Geschlecht der genannten Entität ('{actor['actor_label']}') ist {{}}."
            try:
                res = zero_shot_model(context_sentence, candidate_labels, hypothesis_template=hypothesis_template)
                base_gender = res["labels"][0]
                prediction_score = float(res["scores"][0])
                
                # Check spacy tokens for Plural tag
                is_plural = False
                for token in actor["nomination_tokens"]:
                    if token.morph.get("Number") == ["Plur"]:
                        is_plural = True
                        break
                        
                # Cross-reference AI meaning with precise grammatical count
                if is_plural:
                    if base_gender == "weiblich":
                        predicted_gender = "Frauengruppe"
                    elif base_gender == "männlich":
                        predicted_gender = "Männergruppe"
                    else:
                        predicted_gender = "gemischte/generische Gruppe"
                else:
                    if base_gender == "weiblich":
                        predicted_gender = "weiblich (Singular)"
                    elif base_gender == "männlich":
                        predicted_gender = "maskulin (Singular)"
                    else:
                        predicted_gender = "generisch maskulin (Singular)"
            except Exception as e:
                pass

        # Build a grammatically safe lemma (Nominative) to construct prompts
        actor_lemma = str(actor["actor_label"])
        if actor["nomination_tokens"]:
            head_token = actor["nomination_tokens"][-1]
            if head_token.lemma_:
                lemma_val = head_token.lemma_
                if len(lemma_val) > 0:
                    lemma_val = lemma_val[0].upper() + lemma_val[1:]
                words = actor_lemma.split(" ")
                words[-1] = lemma_val
                actor_lemma = " ".join(words)

        actor_rows.append(
            {
                "pair_id": pair_id,
                "side": side,
                "source": row["source"],
                "topic": row.get("topic", ""),
                "url": row.get(f"{side}_url", ""),
                "actor_id": f"{pair_id}_{side}_{index:03d}",
                "actor_label": actor["actor_label"],
                "actor_lemma": actor_lemma,
                "actor_type": actor["actor_type"],
                "nomination_count": len(actor["nomination_tokens"]),
                "nominations": " | ".join(nominations),
                "pronoun_count": len(actor["pronoun_tokens"]),
                "pronouns": " | ".join(sorted({token.text for token in actor["pronoun_tokens"]})),
                "main_pronoun_class": main_pronoun,
                "predicted_gender": predicted_gender,
                "predicted_gender_score": round(prediction_score, 4),
                "predication_count": len(predications),
                "predications": " || ".join(predications),
            }
        )
    return actor_rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_summary(rows: list[dict[str, object]], coref_available: bool, model: str) -> dict[str, object]:
    source_counts = Counter(row["source"] for row in rows)
    side_counts = Counter(row["side"] for row in rows)
    actor_type_counts = Counter(row["actor_type"] for row in rows)
    pronoun_class_counts = Counter(row["main_pronoun_class"] for row in rows)
    return {
        "actor_row_count": len(rows),
        "sources": dict(sorted(source_counts.items())),
        "sides": dict(sorted(side_counts.items())),
        "actor_types": dict(sorted(actor_type_counts.items())),
        "main_pronoun_classes": dict(sorted(pronoun_class_counts.items())),
        "spacy_model": model,
        "coreference_enabled": coref_available,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--model", default="de_core_news_sm")
    parser.add_argument("--enable-coref", action="store_true")
    parser.add_argument("--use-gliner", action="store_true", help="Use GLiNER to extract Person and Job Titles.")
    parser.add_argument("--use-bert-person-nouns", action="store_true", help="Use fine-tuned BERT model for detecting personal nouns.")
    parser.add_argument("--exclude-plurals", action="store_true", help="Filter out plural generic nouns to isolate individual entities.")
    parser.add_argument("--predict-gender", action="store_true", help="Use mDeBERTa zero-shot classification to predict the gender context of extracted actors.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]

    nlp, coref_available, gliner_model, bert_model, zero_shot_model = load_models(args.model, args.enable_coref, args.use_gliner, args.use_bert_person_nouns, args.predict_gender)

    doc_inputs: list[tuple[str, str, dict[str, str], str]] = []
    for index, row in enumerate(rows, start=1):
        pair_id = f"pair_{index:04d}"
        for side in ("simple", "normal"):
            text = row.get(f"{side}_text", "")
            if text:
                doc_inputs.append((pair_id, side, row, normalize_text(text)))

    actor_rows: list[dict[str, object]] = []
    for meta, doc in zip(doc_inputs, nlp.pipe((item[3] for item in doc_inputs), batch_size=16), strict=True):
        pair_id, side, row, _text = meta
        actor_rows.extend(build_actor_rows(pair_id, side, row, doc, coref_available, gliner_model, args.exclude_plurals, bert_model, zero_shot_model))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, actor_rows)

    summary = build_summary(actor_rows, coref_available, args.model)
    with args.summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
