# Solution for the EUOS25 challenge

## Implementation

* the raw competition data sets can be found in data/challenge
* we standardized the data sets and saved them to data/derived (ready for ML)
* we implemented a torch_geometric data set able to return the data in different variations:
  * `mode` ("binary", "continuous"): affects the label format
  * `use_absorbance` (bool): whether continuous transmittance values should be transformed using 
    the formula $-\log(\mathrm{transmittance}/100)$
  * `use_plate_indices` (bool): adds plate indices if true (note: can only be used on 
    training and leaderboard data)
  * `normalize_by_plate` (bool): normalized labels by subtracting plate mean label and dividing by 
    standard deviation (note: can only be used on training and leaderboard data)
  * `normalize_labels` (bool): subtract overall label mean and divide by standard deviation (note:
    can only be used on training and leaderboard data)
* multiple ML models were trained according to training protocols in euos25/training
* the final model is an ensemble containing
  * a matrix factorization model (model_1)
  * multiple NeuMF models (model_2, model_3, model_4, model_5)
  * multiple single-task models (see models/single/$task for prediction files)
* parameters for the matrix factorization models are specified in euos25/ensemble/model_$i.py
* parameters for the single-task models can be found in https://github.com/KdDiedrich/euos25_challenge

Note: plate information is only used for training and leaderboard compounds. Prediction on new
molecules (including test set) uses structural information only. 

## Generate submission file

```bash
# clone repository
git clone https://github.com/shirte/euos25.git

# provide Python=3.12 (e.g. via conda or similar)

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