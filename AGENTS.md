# ProjectAthena operating rules

- Python execution must use only `E:\ProjectAthena\.python-runtime\python.exe`.
- Work only inside `E:\ProjectAthena`.
- Never place MT5 live orders and never add or call `MetaTrader5.order_send`.
- Never use future bars, incomplete higher-timeframe bars, or future-derived features.
- Never tune conditions after viewing test-period results.
- Preserve rejected experiments and their results.
- Fix bugs by overwriting the same script; use a new descriptive filename for a genuinely new hypothesis.
- Obtain approval before dangerous operations, package installation, network access, large deletion, commit, or push.
- Historical research stops at 24 protocol-period new experiments total or 5 protocol-period experiments per strategy family without an edge, with state `STOP_NO_EDGE`. Imported pre-protocol experiments do not count toward either limit.
- Stop and report at `PRE_LIVE_READY`, `STOP_NO_EDGE`, `BLOCKED_APPROVAL`, or when authority outside these rules is required.
- Historical acceptance requires all of: preregistered minimum trades (normally at least 200; never below 100), at least 3/4 positive folds, cost-adjusted PF >= 1.20, mean-R 95% lower bound > 0, max drawdown <= 15R, no fold contributing over 40% of total profit, no single-month concentration, no unjustified direction dependence, nearby-parameter stability, and a passed leakage audit.
