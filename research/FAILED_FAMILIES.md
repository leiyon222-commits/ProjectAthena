# Failed Strategy Families

## Simple EMA rules

Existing backtests and optimizer outputs did not establish a robust walk-forward edge. Do not retry by small EMA/RSI threshold changes.

## USDJPY M5 expected-R regression

Expected-R prediction correlations were weak and validation-selected conditions generally produced no eligible rule. The evaluated test lost money.

## Expected-R top-percentile selection

The selected validation condition reversed out of sample: the evaluated fold returned about -11.53R with PF about 0.77. Do not retry percentile tweaks.

## Long-term and currency-strength feature addition

Feature-ablation variants selected different sets across folds and all evaluated tests were negative. This indicates selection instability rather than a durable feature edge.

## Repeated entry during long-direction regimes

Direction-only tests produced -431.85R, PF 0.52 and 0/4 positive folds, with severe direction and regime concentration.

## Long direction plus M5 timing classification

Only one fold found a validation-eligible AI filter; its test returned -10.70R and PF 0.66. AUC above 0.5 did not translate into trading edge.

## M5 EMA20 pullback/recovery cross

The fixed event experiment produced 332 trades, -130.32R, PF 0.47 and 0/4 positive folds. Restricting repeated entries to M5 recovery events did not repair the edge.

M5-centered families are closed to incremental threshold tuning. New historical work should prioritize structurally different H1/H4 hypotheses.

## Confirmed-H1 Donchian breakout with H4 trend

The 20-H1 primary produced 327 trades, -56.45R, PF 0.74, 1/4 positive folds and 74.88R drawdown after the preregistered 0.05R cost. The 15- and 30-H1 neighbors also had PF below 0.80 and only 1/4 positive folds. Close this family to lookback tuning.

## H1 Asian-session range breakout

The primary range filter produced only 48 trades, -10.40R, PF 0.69 and 0/4 positive folds. One narrow robustness neighbor looked better but was not preregistered as the primary and cannot be promoted after observing test data. Close this family to range-bound tuning.

## H1 distributional mean reversion

The two-sigma primary produced 482 trades, -137.06R, PF 0.59 and 0/4 positive folds. Both 1.8- and 2.2-sigma neighbors also lost in every fold. Close this family to z-score tuning.

## H1 volatility expansion

The primary produced 431 trades, -106.29R, PF 0.64 and 0/4 positive folds. Both preregistered neighbors failed in every Fold.

## Final preregistered H4, H1 candle, daily and session structures

Thirteen conditions were preregistered together before their results were observed. All failed. PF ranged from 0.40 to 0.83; no condition reached three positive folds; all confidence lower bounds were negative. Detailed Fold and monthly results are preserved in `final_batch_results.csv` and `final_batch_monthly.csv`. These families must not be threshold-tuned on the current historical data.

## Cross-asset USD/JPY pressure impulse

This was the first experiment under the corrected SHA-256-backed ledger. The primary 75% agreement event produced 1,159 trades, -303.57R, PF 0.62 and 0/4 positive folds. The preregistered 2/3 and 5/6 agreement neighbors also lost in all four folds. This closes the remaining distinct cross-asset event family.

## Cross-asset dispersion and horizon-state experiments

Six individually hash-preregistered experiments completed the active 24-experiment protocol. Dispersion expansion (PF 0.63), dispersion convergence (0.64), H1/H4/D1 alignment (0.60), H1/D1 resolution (0.60), agreement shock (0.65), and weekly/monthly hierarchy conflict (0.61) all produced 0/4 positive Folds. Their preregistered robustness variants did not provide a stable edge and were not substituted for the primaries.
