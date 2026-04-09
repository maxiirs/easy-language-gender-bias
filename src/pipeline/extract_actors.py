import sys
import os
import pandas as pd
import json
import torch
import spacy
import llpro
from tqdm import tqdm

# ... Rest des Skripts ...
def extract_actors(text, nlp):
    """
    Jagt den Text durch die Pipeline und extrahiert Akteure und deren Geschlecht.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return json.dumps([])
    
    try:
        doc = nlp(text)
        actors_data = []
        
        # LLpro speichert die erkannten Figuren in doc._.characters
        if hasattr(doc._, 'characters'):
            for character in doc._.characters:
                actors_data.append({
                    "name": character.main_name,
                    "gender": character.gender,
                    "mentions": len(character.mentions)
                })
                
        return json.dumps(actors_data)
    
    except Exception as e:
        # Falls ein Text wegen Überlänge oder unerwarteter Zeichen crasht,
        # fangen wir den Fehler ab, damit nicht das ganze Skript abbricht.
        print(f"\nFehler bei einem Text: {e}")
        return json.dumps([{"error": "Verarbeitungsfehler"}])

def main():
    input_file = 'data/processed/paired_dataset_accepted.csv'
    output_file = 'data/processed/paired_dataset_mit_akteuren.csv'
    
    # 1. GPU Check
    print("--- System-Check ---")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[OK] GPU gefunden: {gpu_name}. Aktiviere CUDA.")
        spacy.require_gpu()
    else:
        print("[WARNUNG] Keine GPU gefunden. Skript läuft auf CPU (wird sehr lange dauern!).")

    # 2. Daten laden
    print(f"\n--- Lade Daten: {input_file} ---")
    df = pd.read_csv(input_file)
    
    # WICHTIG: Testmodus!
    # Wir testen hier erst mit den ersten 5 Zeilen. 
    # Wenn alles klappt, ändern Sie diese Zeile zu: df_process = df.copy()
    print("[HINWEIS] Testmodus aktiv: Verarbeite nur die ersten 5 Zeilen.")
    df_process = df.head(5).copy()

    # 3. Pipeline laden
    print("\n--- Lade LLpro Modelle ---")
    print("(Beim ersten Start werden jetzt die großen Transformer-Modelle heruntergeladen...)")
    
    # Wir laden das deutsche Basis-Modell für die grundlegende Grammatik und Satz-Trennung
    nlp = spacy.load("de_core_news_sm")
    
    # Jetzt stecken wir die spezifischen LLpro-Werkzeuge für unsere Aufgabe dazu:
    # 1. 'ner_flair': Sucht nach Namen und Entitäten im Text
    nlp.add_pipe("ner_flair")
    
    # 2. 'coref_uhhlt': Löst Pronomen auf (verknüpft "Angela Merkel" mit "sie")
    nlp.add_pipe("coref_uhhlt")
    
    # 3. 'character_recognizer': Bündelt diese Infos zu Figuren und schätzt das Gender (m/w/x)
    nlp.add_pipe("character_recognizer")
    
    # 4. Verarbeitung
    print("\n--- Starte Extraktion: Standardtexte ---")
    df_process['normal_actors'] = df_process['normal_text'].progress_apply(lambda t: extract_actors(t, nlp))
    
    print("\n--- Starte Extraktion: Leichte Sprache ---")
    df_process['simple_actors'] = df_process['simple_text'].progress_apply(lambda t: extract_actors(t, nlp))
    
    # 5. Speichern
    df_process.to_csv(output_file, index=False)
    print(f"\n[ERFOLG] Verarbeitung abgeschlossen. Ergebnisse gespeichert in: {output_file}")

if __name__ == "__main__":
    main()