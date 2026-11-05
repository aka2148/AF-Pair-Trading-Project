# Saved backtest outputs

`full_fast/` contains the saved full-universe experiment. It covers 57 stocks, forms up to 50 pairs per year, trains on 2017-2020, selects on 2021-2023 validation, and reports held-out performance for 2024-2025. The FAST configuration uses one PPO seed, 12,000 PPO training steps, and three LSTM epochs; it is intended for a quicker research iteration, not a final robustness claim.

| File | Contents |
| --- | --- |
| `full_fast/run_manifest.json` | Input hash, profile, date coverage, train/validation/test years, costs, exposure cap, and training budget. |
| `full_fast/results_table.csv` | Test metrics for baseline and model variants, including Sharpe confidence intervals. |
| `full_fast/test_daily_returns.csv` | Daily held-out returns for every reported strategy. |
| `full_fast/selection_counts.csv` | Annual pair selection and screening counts. |
| `full_fast/selected_pairs.csv` | Pair ledger with formation year, method, and selection tier. |
| `full_fast/selection_comparison.csv` | Comparison of the all-pairs and clustered selectors. |
| `full_fast/lstm_diagnostics.csv` | Saved LSTM forecast diagnostics. |
| `full_fast/test_results.png` | Plot of the held-out strategy results. |
| `full_fast/final_ppo_model.zip` | Stable-Baselines3 PPO checkpoint for the variant selected using validation Sharpe. In this run it is `PPO + LSTM + Markov`, seed 0. |

The PPO ZIP is a research checkpoint, not a standalone deployment package. The pipeline still needs the matching feature-generation logic and licensed input data. LSTM weights and Markov transition matrices are currently trained or estimated during a run and are not saved as separate model files.

The `compact_fast/` folder is kept local and ignored by Git because its input CSV has unverified provenance. The run outputs are not published without a verifiable data source.
