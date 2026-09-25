# Local dataset files

Place downloaded data and preprocessing artifacts in the corresponding folder:
`ml-1m/`, `lastfm-1k/`, `coco/`, or `goodreads/`. These files are ignored by Git.
Dataset download links are in the repository README.

Store experiment hyperparameters, metrics, and model checkpoints under
`results/<dataset>/<seed>/` instead. COCO additionally uses `all/` or `subset/`
between its dataset name and seed.
