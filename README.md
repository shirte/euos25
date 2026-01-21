# Solution for the EUOS25 challenge

## Implementation

* 

## Generate submission file

```bash
# clone repository
git clone https://github.com/shirte/euos25.git

# install repository
cd euos25
poetry install

# generate normalized data files
poetry run python -m euos25.data.process_data

# train models (optional; for reproducing results)
poetry run python -m euos25.ensemble.model_1
poetry run python -m euos25.ensemble.model_2
poetry run python -m euos25.ensemble.model_3
poetry run python -m euos25.ensemble.model_4
poetry run python -m euos25.ensemble.model_5

# generate submission file from model checkpoints
poetry run python -m euos25.ensemble.build_final_predictions

# find submission file submission.csv in project root
```