# SurrockPred
strength prediction 
# Tunnel Surrounding Rock Strength Prediction
## Files
- `data.csv`: simulated strength labels, construction groups and 32 F/T features
- `mwd_features.csv`: 32 simulated MWD descriptors linked by `sample_id`
- `model.py`: dual-branch model, directly takes MWD feature samples as input
- `run.py`: one stratified group-held-out train/validation/test run
- `requirements.txt`: runtime environment

## Run
```bash
pip install -r requirements.txt
python run.py
```
The `data.csv` and `mwd_features.csv` are already included.

After running, the results are saved in `results/`:
- `summary.json`: model and evaluation metrics
- `predictions.csv`: train/validation/test predictions 
