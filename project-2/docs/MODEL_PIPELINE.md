# SAM probability-model pipeline

SAM now has a real model-evaluation foundation. It does not claim that any
candidate is profitable, and it does not permit a browser, an LLM, or a caller
supplied model label to promote a model.

## What is implemented

The sam_analytics.modeling package accepts only complete, numeric, versioned,
point-in-time features with immutable source-snapshot identifiers. Every
training row has a scheduled event start, decision, feature-availability, and
label-availability timestamp. Inputs recorded after the decision or event
start, labels reused as features, missing lineage, and non-finite values are
rejected.

The default candidate set is intentionally diverse but bounded:

- regularized logistic regression for a stable probability baseline;
- histogram gradient boosting for nonlinear interactions; and
- a neural multilayer perceptron evaluated under the same chronology and
  governance rules.

Candidates are evaluated with expanding chronological folds. The default
splitter scores every eligible historical validation era, so its promotion
sample and coverage gates can actually be met rather than only looking strong
in a configuration file. A training fold can use a result only after its label
would have been available at that historical cutoff.

Out-of-fold scores are calibrated forward with isotonic regression and reported
with Brier score, log loss, calibration bins, and expected calibration error.
Model inference cannot score a decision dated before the model release time.
Final fitting recomputes the promotion decision from the evaluation and policy;
it does not accept a caller-created approval flag.

## Required operating sequence

1. Capture licensed raw provider payloads in immutable object storage and
   record digest, source time, receipt time, and licence scope.
2. Build point-in-time feature vectors using only those source snapshots and
   persist the exact feature contract.
3. Freeze a training-data manifest with chronological split plan and code
   revision.
4. Train all approved candidates, preserve out-of-fold predictions, calibrate
   forward, and write a model evaluation report.
5. Verify the model artifact checksum, write an append-only governance
   decision, then release the version at an explicit effective time.
6. Monitor freshness, calibration, drift, and provider quality. Suspend
   publication whenever a required signal is stale, invalid, or unavailable.

The numbered migrations create an append-only evidence, manifest, evaluation,
approval, and operational-signal ledger. Migration 003 adds database-level
cross-table checks for event times, source receipt, vector/result matching,
training cutoffs, prediction/vector links, and governance/report identity.

## Important limits

This is a professional-style research framework, not a guaranteed edge. A
neural network only becomes useful after enough correctly licensed,
time-aligned, feature-complete data exists. Early leagues or sparse markets may
justify retaining the logistic baseline or withholding output entirely. The
current web API is research-only and never submits wagers.
