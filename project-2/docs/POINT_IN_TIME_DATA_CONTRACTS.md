# Point-in-time data contracts

This contract is the minimum evidence standard for any SAM training row,
backtest observation, or served pregame prediction. It does not authorize a
model to infer, fill, scrape, or simulate missing data. A missing exact record
is a failed data dependency, not an invitation to make up a value.

The executable definitions live in `sam_analytics/data_contracts.py` and are
dependency-free so ingestion, workers, model training, and audit jobs can use
the same rules.

## Immutable provider evidence

Every raw response must be written unchanged to durable, access-controlled
object storage before it is eligible for features. Store a separate
`RawDataProvenance` record with:

- provider name and provider record ID;
- source type and provider schema version;
- SHA-256 of the exact raw payload;
- durable object URI (never a temporary signed URL or credential); and
- timezone-aware provider capture and local receipt timestamps.

The payload digest is the integrity check; the provenance digest additionally
binds its provider identity, timestamps, object reference, schema, and license
scope. Both are content-addressed identifiers. They are frozen Python values,
and a database implementation must be append-only. If a provider corrects a
record, store a new raw payload and a new provenance record. Do not update the
one that supported an earlier decision.

Local receipt time is the conservative knowledge boundary. A provider may say
that a fact was published earlier, but it was not available to SAM until SAM
received and retained it. The validator permits only five minutes of provider
clock skew between capture and receipt; larger disagreement is an error that
requires investigation.

## Feature contract and leakage rule

Each model declares an ordered `FeatureContract`. A feature definition locks:

- lowercase feature name;
- transformation version;
- whether its observation is required;
- whether a missing value is permitted and, if so, how it is labelled;
- optional numeric range; and
- optional maximum staleness in seconds.

Each `FeatureObservation` carries its raw provenance, transformation time, and
`available_at`. A feature is valid only when:

1. its raw payload was received before it was computed;
2. it was made available after computation and after local source receipt;
3. its transformation version matches the model's feature contract;
4. its numeric value is finite and within its declared bounds (or its missing
   policy explicitly permits a labelled missing value); and
5. `available_at <= as_of`, where `as_of` is the actual historical decision
   time.

`PointInTimeFeatureVector` additionally requires an exact contract match,
unique observations, pregame `as_of < event_starts_at`, and any configured
freshness limit. A vector containing data that became available one second
after its claimed decision time fails validation; it may not be silently
shifted, imputed, or used in a backtest.

Use the same availability rule for injuries, lineup news, weather, schedule
changes, odds, team statistics, and any derived feature. An event timestamp or
provider's retroactive historical timestamp is not proof that SAM knew the
information at the time.

## Labels and training manifests

`LabeledTrainingExample` joins a frozen feature vector to an authoritative
binary settlement and a separate result payload. It rejects a result that
predates the event and rejects use of that exact result payload as a pregame
feature source.

Build a `TrainingDatasetManifest` only after labels are settled. The manifest
contains the exact feature-contract digest, target definition, chronological
row digests, deduplicated raw-source digests, training cutoff, code revision,
and split strategy. Its own SHA-256 is the training-data identity recorded in a
model registry or experiment tracker.

`training_cutoff` is not merely a calendar filter. Every feature decision,
settlement, and result receipt must be on or before it. This keeps a historical
experiment from claiming it could have trained with labels SAM had not yet
received. The builder also rejects empty datasets, duplicate row IDs, duplicate
events, nonchronological rows, contract mismatch, and future manifest cutoffs.

Before training or reproducing a result, call
`manifest.verify_examples(examples)`. If it returns `False`, stop and create a
new reviewed manifest rather than substituting corrected provider data or a
different row ordering under an old model version.

## Repository requirements

`PointInTimeDataRepository` is intentionally a protocol, not a fake local
database. A production implementation must:

1. validate an object before appending it;
2. preserve it by digest without update/delete semantics;
3. record object-storage access separately from public API responses;
4. return an exact feature vector only for the requested event, decision time,
   and feature-contract digest; and
5. return `None` if that exact persisted record does not exist.

It must never select the nearest quote, rebuild a vector with newer data,
default missing values, or generate synthetic source records. Those operations
would make the point-in-time claim false and must instead create a data-quality
incident or a new explicitly-versioned dataset.

## Operational response to validation errors

Treat these errors as fail-closed for model training, backtesting, and served
recommendations:

- `FEATURE_AVAILABLE_AFTER_AS_OF`, `POST_START_FEATURE_VECTOR`, and
  `FEATURE_COMPUTED_BEFORE_SOURCE_RECEIPT` — potential look-ahead leakage;
- `STALE_FEATURE`, `MISSING_REQUIRED_FEATURE`, nonfinite values, and range
  violations — unsafe feature input;
- invalid or future provenance timestamps/digests — unverifiable source data;
- `LABEL_AVAILABLE_AFTER_CUTOFF` and `SETTLEMENT_AFTER_CUTOFF` — invalid
  historical training claim; and
- feature-contract, row, event, or manifest digest mismatch — reproducibility
  failure.

Persist the report as a `data_quality_incident`, preserve the rejected raw
evidence where licensing permits, stop the affected market/model, and require
human review before promotion. Do not turn an error into a warning in a model
pipeline without a documented policy change and a new feature-contract version.
