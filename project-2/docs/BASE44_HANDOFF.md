# Base44 and Python handoff

Base44 is now SAM's evidence-and-governance experience, not the source of
predictions. Its assistant can explain a verified model release, fresh or stale
feed, calibration report, risk gate, or incident in plain language. It cannot
invent an edge, estimate missing odds, create a pick, or bypass a Python-side
publication gate.

## Safe connection one: Base44 reads Python status

The Base44 server function sam-backend-status calls the Python status endpoint.
Set these as Base44 server secrets, never in React code, browser storage, or an
AI prompt:

- SAM_BACKEND_URL set to https://api.sam.vegas
- SAM_BACKEND_HOST set to api.sam.vegas
- SAM_BACKEND_STATUS_API_KEY set to the separate SAM_STATUS_API_KEY value

The status function pins the hostname, requires HTTPS, rejects redirects and
non-JSON/oversized responses, and projects a fixed nested schema before any
value reaches the UI or evidence assistant. The status key cannot call the
private research evaluator.

## Safe connection two: Python publishes evidence to Base44

The Base44 python-evidence-ingest function accepts a small authenticated
evidence record. Generate a separate random secret of at least 32 characters.
Do not reuse the Python API key or a provider key.

Set the following Base44 server secret:

- SAM_EVIDENCE_WEBHOOK_TOKEN

After Base44 supplies the production function endpoint, set these Python host
secrets:

- BASE44_EVIDENCE_WEBHOOK_URL, the HTTPS endpoint without a query string;
- BASE44_EVIDENCE_WEBHOOK_HOST, the exact hostname parsed from that endpoint;
- BASE44_EVIDENCE_WEBHOOK_TOKEN, the same separate webhook secret.

The Python publisher validates the evidence, pins the destination host, refuses
redirects, and sends the token only in an Authorization header. It also signs
the exact body with HMAC-SHA256 and a short-lived timestamp. The Base44 function
rejects stale/replayed requests, wrong content types, oversized bodies, and an
older evidence record attempting to replace a newer provenance record.

Evidence covers data freshness, model releases, backtests, calibration, risk
gates, provider notices, and incidents. It never includes raw odds payloads,
feature vectors, customer information, or provider credentials.

## Operations copilot role

The Evidence Copilot reads only two sanitized Base44 server functions:
sam-evidence-feed and sam-backend-status. The evidence ledger remains available
only to Base44 operations administrators. The assistant can explain why a
model is withheld or what a calibration report means, but it cannot produce a
betting prediction or turn a probability into a recommendation.

## Rollout order

1. Deploy the Python API, private worker, Postgres, Redis or Valkey, and object
   storage. Run numbered migrations through the included migration runner.
2. Rotate the Odds API credential that was shared in chat. Store its replacement
   only in the worker's host secret manager.
3. Configure the status secrets and confirm Base44 displays blocked until real
   operational evidence exists.
4. Configure the webhook secret on both sides and publish one harmless
   provider_notice or risk_gate record from a private worker.
5. Add licensed feed storage, a feature contract, frozen backtest, calibration
   report, artifact checksum, and approval decision.
6. Enable a model release only after every gate passes. Withhold publication
   whenever a gate fails.
