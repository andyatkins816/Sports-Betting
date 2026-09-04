# Data, model, backtest, and risk governance

## Data architecture

```text
Licensed odds / results / roster providers
        │ immutable raw payload + provider timestamp + checksum
        ▼
Provider adapter → canonical event/market mapping → append-only Postgres ledger
        │                                      │
        │                                      └─ data-quality incidents
        ▼
Point-in-time feature store → versioned model artifact → calibrated probability
        │                                                │
        └───────────── decision timestamp ──────────────┘
                                      ▼
                     price-aware risk policy → analytic decision
                                      ▼
               provider settlement → evaluation / monitoring reports
```

Store provider IDs, received timestamps, provider-captured timestamps, raw
payload checksum/object location, event mapping version, and transformation
version with every datum. Retain original provider responses according to the
signed contract. A corrected feed is a new record; it must not silently replace
the original record used to make an old decision.

Before adding a source, record these contract fields: permitted sports/markets,
pregame versus in-play rights, update latency, rate/credit calculation, historic
depth, retention, display/attribution, derived-model rights, redistribution,
geography, service-level support, and termination/deletion requirements.

## Ingestion and quality controls

One provider adapter must own its schema translation. The included The Odds API
v4 adapter covers featured pregame markets only, preserves the provider event/
book/market/selection structure, filters in-play events, and returns quota
headers. It is intentionally not scheduled until its database repository and
data-quality incident writer are connected. Every adapter should:

1. Pull with timeouts, retry only idempotent requests using exponential backoff,
   honor `Retry-After`, and emit remaining quota from response headers.
2. Reconcile each provider event to a canonical event ID using date, league,
   participants, and provider ID—not display names alone.
3. Preserve every quote with its provider timestamp; reject future timestamps,
   non-finite odds, live/completed events in the pregame path, duplicate quote
   fingerprints, mismatched selection pairs, and abnormal currency/line changes.
4. Treat stale feeds, event starts moving, missing books, invalid lines, mapping
   conflicts, quota exhaustion, and provider drift as data-quality incidents.
5. Never fall back to synthetic odds, team stats, injuries, sentiment, or results.

Start with moneyline/spread/total pregame markets. Player props, in-play, and
same-game parlays add different settlement, correlation, and latency problems;
they are separate products, not a field added to the same model.

## Modeling standard

Build a strong and auditable baseline before a complex ensemble:

- target: defined market and settlement rule (including overtime, push, void,
  postponement, and scratched-player behavior);
- features: only information available before the recorded `as_of`; roster,
  injury, travel, schedule, and team statistics need source timestamps;
- splits: expanding or rolling time windows by event start/decision time; never
  random train/test splits, and purge overlapping feature windows where needed;
- calibration: fit Platt or isotonic calibration on a dedicated prior/OOF set,
  then lock it before the final holdout evaluation;
- artifacts: hash feature contract, training data snapshot, code revision,
  hyperparameters, calibration set, and metrics; a human approves promotion;
- baselines: compare against no-vig consensus/closing price (only when licensed),
  home advantage, and simple rating models.

Report Brier score, log loss, calibration bins, sharpness, and per-segment
coverage. Accuracy is not sufficient for a probability model. Report realized
returns only after realistic historical prices, settlement rules, limits, timing,
and non-selection-cherry-picking are represented; it remains noisy evidence, not
a promise.

## Backtesting protocol

For each historical decision, freeze:

- decision timestamp and provider quote timestamp;
- feature availability timestamp and feature values/checksum;
- exact market, book, line, and price that could have been accepted then;
- model + calibration + policy version; and
- authoritative final settlement plus void/push rules.

The included run_backtest rejects out-of-order rows, post-start decisions,
pre-release model use, missing immutable quote/prediction IDs, and any feature
or quote whose timestamp arrives after the decision. It requires a
registry-derived release identity with an artifact digest. A production job must
also account for response latency and stale prices, and log all
candidates—including rejected ones—to avoid selection bias.

Evaluate walk-forward periods, seasons, leagues, favorites/underdogs, price
buckets, start-time horizons, and injury-news states. Perform a final untouched
holdout after strategy/risk parameters are frozen. Do not optimize thresholds on
the reported holdout, and control for multiple strategy/model comparisons.

## Risk and bankroll controls

`BankrollPolicy` applies quarter-Kelly by default and caps a single decision at
1% of bankroll, one event at 2%, and daily exposure at 5%. Those are conservative
defaults to test, not universally correct parameters. A live deployment should
also implement:

- user/account-level authorization and jurisdiction eligibility;
- hard maximum loss, drawdown stop, cooldown, and manual kill switch;
- book, market, sport, team/player, start-time, and correlated-event exposure;
- a covariance-aware portfolio optimizer only after correlation estimates are
  validated; no independent-leg parlay EV assumption;
- price revalidation immediately before any external action, slippage/limit
  handling, and immutable decision and response logs; and
- responsible-gambling features and human review.

There is intentionally no betting-execution client in this repository. Any
future integration with a regulated operator requires its explicit API approval,
jurisdictional legal review, consent, and separate security review.

## Monitoring and model operations

Alert on feed freshness, quote count by provider/market, mapping failures, API
quota remaining, worker retry/dead-letter count, scheduler lag, database errors,
model artifact mismatch, null/missing feature rate, prediction distribution,
calibration drift, realized-vs-expected segments, CLV where allowed, and risk
limit hits. Keep a dashboard for SLOs and a separate model report; never mix
demo, backtest, shadow, and live measurements.

Promote only after a reviewable report and signed approval. Roll back to the last
approved model when data quality alarms, calibration deterioration, or serving
errors trip the predetermined guardrails. Keep the last known good artifact and
perform restore drills for both database and object storage.
