# easy-language-gender-bias

Pipeline for analyzing gender representation in German Easy Language and Standard German, and for evaluating how actor-level text categories are translated into text-to-image outputs.

## Final Pipeline

Run the steps from the repository root:

```bash
python src/corpus/filter_paired_dataset.py
python src/annotation/annotate_texts.py
python src/lexicon/build_kaikki_lookup.py
python src/text_analysis/prepare_gender_data.py
python src/text_analysis/merge_gender_reviews.py
python src/t2i/build_actor_prompts.py
python src/t2i/generate_t2i_images.py
python src/image_analysis/evaluate_mivolo_images.py
python src/results/analyze_text_bias.py
python src/results/analyze_image_bias.py
python src/results/analyze_text_image_bias.py
```

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

Each result directory contains a `summary.json`, CSV tables, and a `plots/` subdirectory.

## Notes

Generated T2I images are stored under `data/processed/t2i_images/` and tracked with Git LFS. Use `git lfs pull` after cloning if image files are needed locally.
