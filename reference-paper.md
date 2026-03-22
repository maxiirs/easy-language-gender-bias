# Zusammenfassung: Multilingual Text-to-Image Generation Magnifies Gender Stereotypes

**Referenz:** Friedrich, F., Hämmerl, K., Schramowski, P., Brack, M., Libovický, J., Kersting, K., & Fraser, A. (2025). *Multilingual Text-to-Image Generation Magnifies Gender Stereotypes*. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025).  
**Papier-Link:** [https://aclanthology.org/2025.acl-long.966.pdf](https://aclanthology.org/2025.acl-long.966.pdf)  
**Offizielles Repo der Autoren:** [MAGBIG auf GitHub](https://github.com/felifri/MAGBIG)

## 1. Kernausssage des Papers
Das Paper untersucht, wie mehrsprachige Text-to-Image (T2I) Modelle Geschlechterstereotypen verarbeiten und darstellen. Die Autoren zeigen, dass KI-Modelle diese Biases nicht nur reproduzieren, sondern sogar **verstärken (magnify)**. Zudem variiert die Ausprägung des Biases stark je nach verwendeter Sprache, selbst bei grammatikalisch ähnlichen Sprachen (z. B. Spanisch vs. Französisch). Prompt-Engineering-Strategien (wie geschlechtsneutrale Formulierungen) helfen nur bedingt und verschlechtern oft die Qualität der Text-Bild-Übereinstimmung (Alignment).

## 2. Methodik der Bildgenerierung (Die MAGBIG-Benchmark)
Für dein Projekt ist die Methodik zur Erstellung der Text-Prompts entscheidend. Die Autoren haben den Benchmark **MAGBIG** (*Multilingual Assessment of Gender Bias in Image Generation*) entwickelt.

Sie basieren ihre Untersuchung auf einer kontrollierten Liste von **Berufsbezeichnungen** und **Charaktereigenschaften**. Um den Bias systematisch zu messen, generieren sie Bilder basierend auf **vier spezifischen Prompt-Kategorien**. 

### Die 4 Prompt-Strategien der Autoren
Wenn du aus deinen Texten (Leichte Sprache / Standardsprache) Prompts ableitest, solltest du dich an dieser Struktur orientieren:

1. **Direkte Prompts (Generisches Maskulinum)**
   * *Struktur:* "Ein [Beruf]."
   * *Beispiel:* "Ein Arzt." / "Ein Buchhalter."
   * *Ergebnis im Paper:* Zeigt den stärksten Bias (z. B. fast ausschließlich weiße Männer bei "Buchhalter").
2. **Indirekte / Beschreibende Prompts**
   * *Struktur:* "Eine Person, die als [Beruf] arbeitet."
   * *Beispiel:* "Eine Person, die als Arzt arbeitet."
   * *Erklärung:* Versuch, das grammatikalische Geschlecht vom biologischen zu entkoppeln.
3. **Explizit feminine Prompts**
   * *Struktur:* "Eine [Beruf - weibliche Form]."
   * *Beispiel:* "Eine Ärztin." / "female doctor"
   * *Ergebnis im Paper:* Generiert fast ausschließlich weiblich gelesene Personen.
4. **Gender-neutrale / Gender-Sternchen Prompts**
   * *Struktur:* Nutzung von Neologismen oder typografischen Markern.
   * *Beispiel:* "Ein*e Arzt*in" oder "Eine ärztliche Fachkraft".
   * *Ergebnis im Paper:* Konnte Stereotype nur minimal abmildern, führte aber oft dazu, dass das Modell den Prompt schlechter verstand (schlechteres Alignment).

## 3. Zentrale Ergebnisse für die Bildgenerierung
* **Bias-Verstärkung:** Die T2I-Modelle weichen stark von einer normativen Gleichverteilung (50% Männer / 50% Frauen) ab und übertreiben gesellschaftliche Stereotype in den generierten Bildern massiv.
* **Trade-off zwischen Neutralität und Qualität:** Wenn versucht wird, den Modellen durch neutrale Prompts (Strategie 4) den Bias abzutrainieren oder zu umgehen, sinkt die Bildqualität und die semantische Übereinstimmung zwischen Prompt und Bild.

---

## 4. Transfer auf dein Projekt: Leichte Sprache vs. Standardsprache

Um die Methodik der Autoren in deinem Code zu referenzieren und nachzubauen, empfiehlt sich folgendes Vorgehen für deine Pipeline:

### A. Extraktion und Prompt-Kategorisierung
Wenn du Texte in Standardsprache in Leichte Sprache übersetzt, verändern sich oft komplexe Berufsbezeichnungen oder Schachtelsätze. Baue deinen Code so auf, dass er deine Sätze in ein ähnliches Raster wie MAGBIG einordnet:
* **Prompt-Set A (Standardsprache):** Oft ähnlich zu *Kategorie 1* (Direktes Maskulinum) oder *Kategorie 4* (komplexe genderneutrale Formulierungen / Sternchen).
* **Prompt-Set B (Leichte Sprache):** Oft ähnlich zu *Kategorie 2* (Indirekte Beschreibungen, z.B. "Eine Person, die kranken Menschen hilft" statt "Medizinisches Fachpersonal").

### B. Referenzierung im Code (Beispielhafte Struktur)
Du kannst in deinen Skripten zur Prompt-Generierung direkt auf das Paper verweisen:

```python
# Reference: Friedrich et al. (ACL 2025) - MAGBIG Benchmark Framework
# Paper: Multilingual Text-to-Image Generation Magnifies Gender Stereotypes
# We adapt their 4-tier prompt generation strategy for 'Leichte Sprache' comparison.

def generate_prompts_from_text(extracted_entity):
    """
    Generates variations of prompts based on the MAGBIG methodology.
    """
    prompts = {
        "standard_direct": f"Ein {extracted_entity['maskulin_standard']}", # e.g., "Ein Arzt"
        "standard_neutral": f"Ein*e {extracted_entity['gender_star']}",   # e.g., "Ein*e Arzt*in"
        "leichte_sprache_indirect": f"Eine Person, die als {extracted_entity['maskulin_standard']} arbeitet", 
        "leichte_sprache_descriptive": f"Eine Person, die {extracted_entity['leichte_sprache_definition']}"
    }
    return prompts