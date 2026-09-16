# GitHub repository workflow

Repository: https://github.com/aylinunel/champions-league-predictor
Default branch: `main`.

## Clone and work locally

With Git installed:

```bash
git clone https://github.com/aylinunel/champions-league-predictor.git
cd champions-league-predictor
git switch -c feature/your-change
```

Make the intended changes and run the verification commands below.

## Push an update

From your checkout, with GitHub authentication available:

```bash
git add <changed-files>
git commit -m "Describe the change"
git push -u origin feature/your-change
```

Then open a pull request on GitHub. Keep generated forecasts and live download
directories out of source control; the included `.gitignore` excludes them.

## Local verification

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_model.py
python ucl_model.py --history football_stats_data.json --backtest
```

The Actions workflow runs the offline tests on pushes and pull requests.
Its action usage follows the official [checkout](https://github.com/actions/checkout)
and [setup-python](https://github.com/actions/setup-python) documentation.
Live result fetching is not part of CI.
