# easy-language-gender-bias

Pipeline for analyzing gender representation in German Easy Language and Standard German, and for evaluating how actor-level text categories are translated into text-to-image outputs.

Tested on Ubuntu 24.04.1 LTS.

## Environments

The pipeline uses three Conda environments:

- `easy-language-gender-bias`: main environment for corpus filtering, lexicon lookup, gender-data preparation, prompt generation, FLUX image generation, and result analysis
- `llpro`: LLpro/spaCy annotation environment
- `mivolo`: MiVOLO image-evaluation environment

Create the environments from the repository root:

```bash
conda env create -n easy-language-gender-bias -f easy-language-gender-bias.yml
conda env create -n llpro -f llpro.yml
conda env create -n mivolo -f mivolo.yml
```

## Final Pipeline

Run all commands from the repository root.

### 1. Corpus Filtering

```bash
conda activate easy-language-gender-bias
python src/corpus/filter_paired_dataset.py
```

Input:

- `data/simple-german-articles.csv`

Outputs:

- `data/processed/paired_dataset_accepted.csv`
- `data/processed/paired_dataset_scored.csv`
- `data/processed/paired_dataset_summary.json`

### 2. Linguistic Annotation

LLpro is expected as a local checkout in the repository root:

```bash
git clone https://github.com/uhh-lt/LLpro.git
```

```bash
conda activate llpro
python src/annotation/annotate_texts.py
```

Output:

- `data/processed/annotated_texts.json`

### 3. Lexicon Lookup And Text-Side Actor Preparation

```bash
conda activate easy-language-gender-bias
python src/lexicon/build_kaikki_lookup.py
python src/text_analysis/prepare_gender_data.py
python src/text_analysis/merge_gender_reviews.py
```

Outputs:

- `data/resources/kaikki_de_lookup.jsonl`
- `data/resources/kaikki_de_lookup_metadata.json`
- `data/processed/gender_analysis_auto.csv`
- `data/processed/gender_analysis_final.csv`
- `data/processed/gender_analysis_final_summary.json`

### 4. Prompt Construction And Image Generation

```bash
conda activate easy-language-gender-bias
python src/t2i/build_actor_prompts.py
python src/t2i/generate_t2i_images.py
```

Outputs:

- `data/processed/actor_prompts.csv`
- `data/processed/actor_prompts_paired.csv`
- `data/processed/t2i_metadata.csv`
- `data/processed/t2i_images/`

### 5. MiVOLO Image Evaluation

MiVOLO is expected as a local checkout in the repository root:

```bash
git clone https://github.com/WildChlamydia/MiVOLO.git
```

The MiVOLO detector and age/gender weights must be available locally. By default, the script expects:

```text
yolov8x_person_face.pt
model_imdb_cross_person_4.22_99.46.pth.tar
```

Run evaluation:

```bash
conda activate mivolo
python src/image_analysis/evaluate_mivolo_images.py \
  --yolo-weights yolov8x_person_face.pt \
  --mivolo-weights model_imdb_cross_person_4.22_99.46.pth.tar
```

For a small smoke test:

```bash
python src/image_analysis/evaluate_mivolo_images.py --limit 10
```

Output:

- `data/processed/t2i_mivolo_results.csv`

### 6. Final Result Analysis

```bash
conda activate easy-language-gender-bias
python src/results/analyze_text_bias.py
python src/results/analyze_image_bias.py
python src/results/analyze_text_image_bias.py
```

Outputs:

- `data/processed/results/text_bias/`
- `data/processed/results/image_bias/`
- `data/processed/results/text_image_bias/`

Each result directory contains a `summary.json`, CSV tables, and a `plots/` subdirectory.

## Source Layout

- `src/corpus/`: corpus filtering and pair selection
- `src/annotation/`: LLpro/spaCy annotation
- `src/lexicon/`: Kaikki lookup construction
- `src/text_analysis/`: text-side actor and gender-category preparation
- `src/t2i/`: actor prompt construction and image generation
- `src/image_analysis/`: MiVOLO image evaluation
- `src/results/`: final text, image, and text-image analyses

## Main Inputs

- `data/simple-german-articles.csv`: source corpus with Easy Language and Standard German article pairs
- `data/annotations/gender_review.csv`: manual review overrides for gender/category assignment
- `data/resources/kaikki_de_lookup.jsonl`: local Kaikki cache for corpus lemmas

## Main Outputs

- `data/processed/paired_dataset_accepted.csv`: filtered article pairs
- `data/processed/annotated_texts.json`: LLpro/spaCy annotations
- `data/processed/gender_analysis_final.csv`: final mention-level actor/gender labels
- `data/processed/actor_prompts.csv`: actor-level prompts
- `data/processed/actor_prompts_paired.csv`: aligned actor prompts across both language varieties
- `data/processed/t2i_metadata.csv`: generated image metadata
- `data/processed/t2i_mivolo_results.csv`: MiVOLO image evaluation output
- `data/processed/results/text_bias/`: structured text-side bias results
- `data/processed/results/image_bias/`: structured image-side bias results
- `data/processed/results/text_image_bias/`: structured text-to-image alignment results

## Notes

Generated T2I images are stored under `data/processed/t2i_images/` and tracked with Git LFS. Use `git lfs pull` after cloning if image files are needed locally.
