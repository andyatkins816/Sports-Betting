"""Immutable, point-in-time contracts for sports-analytics data.

This module deliberately models *evidence*, not forecasts.  It gives an
ingestion or feature-store implementation a small, dependency-free contract
that makes it possible to answer three critical questions for every training
row or decision:

* What immutable provider payload did this value come from?
* When could SAM actually have known that value?
* Does the exact dataset used by a model still match its recorded manifest?

There is no fallback data, nearest-record lookup, or synthetic-data generator
here.  A repository implementation must return ``None`` when an exact,
persisted point-in-time record does not exist.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Protocol, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FEATURE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_MAX_PROVIDER_CLOCK_SKEW = timedelta(minutes=5)


@dataclass(frozen=True)
class DataQualityIssue:
    """A machine-readable validation finding.

    ``severity`` is either ``"error"`` (the record is unsafe to use) or
    ``"warning"`` (it is retained for review but may be disallowed by a
    caller's stricter policy).
    """

    code: str
    message: str
    path: str = ""
    severity: str = "error"


@dataclass(frozen=True)
class DataQualityReport:
    """Immutable result returned by all contract validators."""

    issues: tuple[DataQualityIssue, ...] = ()

    @property
    def errors(self) -> tuple[DataQualityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[DataQualityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def require_valid(self, context: str = "data contract") -> None:
        if self.errors:
            findings = "; ".join(
                f"{issue.code}{f' ({issue.path})' if issue.path else ''}: {issue.message}"
                for issue in self.errors
            )
            raise DataContractViolation(f"{context} validation failed: {findings}")


class DataContractViolation(ValueError):
    """Raised only when a caller asks a validation report to fail closed."""


def _issue(code: str, message: str, *, path: str = "", severity: str = "error") -> DataQualityIssue:
    return DataQualityIssue(code=code, message=message, path=path, severity=severity)


def _is_aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _utc(value: datetime) -> str:
    """Return a stable UTC representation for hashes and manifests."""

    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class RawDataProvenance:
    """Content-addressed evidence for one unmodified provider payload.

    ``payload_uri`` must identify the durable, access-controlled raw object;
    it is a reference only, not an API request URL.  ``received_at`` is the
    earliest time SAM possessed the payload and is therefore the conservative
    lower bound for any feature derived from it.
    """

    provider: str
    provider_record_id: str
    source_type: str
    payload_sha256: str
    payload_uri: str
    captured_at: datetime
    received_at: datetime
    schema_version: str
    license_scope: str | None = None

    @property
    def digest(self) -> str:
        return _canonical_digest(self.to_canonical_dict())

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "provider_record_id": self.provider_record_id,
            "source_type": self.source_type,
            "payload_sha256": self.payload_sha256,
            "payload_uri": self.payload_uri,
            "captured_at": _utc(self.captured_at),
            "received_at": _utc(self.received_at),
            "schema_version": self.schema_version,
            "license_scope": self.license_scope,
        }


def validate_raw_data_provenance(
    record: RawDataProvenance, *, now: datetime | None = None
) -> DataQualityReport:
    """Validate an immutable raw-data receipt without mutating it."""

    issues: list[DataQualityIssue] = []
    for field in ("provider", "provider_record_id", "source_type", "payload_uri", "schema_version"):
        if not _nonempty_text(getattr(record, field, None)):
            issues.append(_issue("REQUIRED_FIELD", "a non-empty value is required", path=field))
    if isinstance(record.payload_uri, str) and ("\n" in record.payload_uri or "\r" in record.payload_uri):
        issues.append(_issue("INVALID_PAYLOAD_URI", "payload_uri cannot contain a line break", path="payload_uri"))
    if isinstance(record.payload_uri, str) and ("?" in record.payload_uri or "#" in record.payload_uri):
        issues.append(
            _issue(
                "UNSTABLE_PAYLOAD_URI",
                "payload_uri must be a durable object reference, not a signed or fragment URL",
                path="payload_uri",
            )
        )
    if record.license_scope is not None and not _nonempty_text(record.license_scope):
        issues.append(_issue("INVALID_LICENSE_SCOPE", "license_scope must be a non-empty string or None", path="license_scope"))
    if not isinstance(record.payload_sha256, str) or not _SHA256_RE.fullmatch(record.payload_sha256):
        issues.append(
            _issue(
                "INVALID_PAYLOAD_SHA256",
                "payload_sha256 must be a lowercase, 64-character SHA-256 hexadecimal digest",
                path="payload_sha256",
            )
        )
    if not _is_aware(record.captured_at):
        issues.append(_issue("NAIVE_TIMESTAMP", "captured_at must be timezone-aware", path="captured_at"))
    if not _is_aware(record.received_at):
        issues.append(_issue("NAIVE_TIMESTAMP", "received_at must be timezone-aware", path="received_at"))
    if _is_aware(record.captured_at) and _is_aware(record.received_at):
        if record.captured_at > record.received_at + _MAX_PROVIDER_CLOCK_SKEW:
            issues.append(
                _issue(
                    "CAPTURED_AFTER_RECEIPT",
                    "captured_at is more than five minutes after local receipt; verify provider clock or payload",
                    path="captured_at",
                )
            )
    if now is not None:
        if not _is_aware(now):
            issues.append(_issue("NAIVE_TIMESTAMP", "validator now must be timezone-aware", path="now"))
        else:
            for field in ("captured_at", "received_at"):
                value = getattr(record, field)
                if _is_aware(value) and value > now:
                    issues.append(
                        _issue(
                            "FUTURE_PROVENANCE_TIMESTAMP",
                            f"{field} cannot be after validation time",
                            path=field,
                        )
                    )
    return DataQualityReport(tuple(issues))


@dataclass(frozen=True)
class FeatureDefinition:
    """One numeric feature in a versioned model input contract."""

    name: str
    transformation_version: str
    required: bool = True
    allow_missing: bool = False
    minimum: float | None = None
    maximum: float | None = None
    max_age_seconds: int | None = None

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "transformation_version": self.transformation_version,
            "required": self.required,
            "allow_missing": self.allow_missing,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "max_age_seconds": self.max_age_seconds,
        }


def validate_feature_definition(definition: FeatureDefinition) -> DataQualityReport:
    issues: list[DataQualityIssue] = []
    if not isinstance(definition.name, str) or not _FEATURE_NAME_RE.fullmatch(definition.name):
        issues.append(
            _issue(
                "INVALID_FEATURE_NAME",
                "feature names must be lowercase snake_case and begin with a letter",
                path="name",
            )
        )
    if not _nonempty_text(definition.transformation_version):
        issues.append(_issue("REQUIRED_FIELD", "a non-empty value is required", path="transformation_version"))
    if not isinstance(definition.required, bool) or not isinstance(definition.allow_missing, bool):
        issues.append(_issue("INVALID_FEATURE_POLICY", "required and allow_missing must be booleans"))
    for field in ("minimum", "maximum"):
        value = getattr(definition, field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            issues.append(_issue("INVALID_FEATURE_BOUND", f"{field} must be a finite number", path=field))
    if (
        isinstance(definition.minimum, (int, float))
        and not isinstance(definition.minimum, bool)
        and isinstance(definition.maximum, (int, float))
        and not isinstance(definition.maximum, bool)
        and definition.minimum > definition.maximum
    ):
        issues.append(_issue("INVALID_FEATURE_BOUND", "minimum cannot exceed maximum"))
    if definition.max_age_seconds is not None and (
        isinstance(definition.max_age_seconds, bool)
        or not isinstance(definition.max_age_seconds, int)
        or definition.max_age_seconds <= 0
    ):
        issues.append(_issue("INVALID_FEATURE_AGE", "max_age_seconds must be a positive integer", path="max_age_seconds"))
    return DataQualityReport(tuple(issues))


@dataclass(frozen=True)
class FeatureContract:
    """The ordered and hashable numeric schema a model is allowed to consume."""

    name: str
    version: str
    target_definition: str
    features: tuple[FeatureDefinition, ...]

    @property
    def digest(self) -> str:
        return _canonical_digest(self.to_canonical_dict())

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "target_definition": self.target_definition,
            "features": [feature.to_canonical_dict() for feature in self.features],
        }

    def definition_for(self, name: str) -> FeatureDefinition | None:
        return next(
            (
                definition
                for definition in self.features
                if isinstance(definition, FeatureDefinition) and definition.name == name
            ),
            None,
        )


def validate_feature_contract(contract: FeatureContract) -> DataQualityReport:
    issues: list[DataQualityIssue] = []
    for field in ("name", "version", "target_definition"):
        if not _nonempty_text(getattr(contract, field, None)):
            issues.append(_issue("REQUIRED_FIELD", "a non-empty value is required", path=field))
    if not isinstance(contract.features, tuple):
        issues.append(_issue("MUTABLE_CONTRACT_COLLECTION", "features must be a tuple to preserve immutability", path="features"))
        return DataQualityReport(tuple(issues))
    if not contract.features:
        issues.append(_issue("EMPTY_FEATURE_CONTRACT", "at least one feature definition is required", path="features"))
    seen: set[str] = set()
    for index, definition in enumerate(contract.features):
        if not isinstance(definition, FeatureDefinition):
            issues.append(_issue("INVALID_FEATURE_DEFINITION", "features must contain FeatureDefinition objects", path=f"features[{index}]"))
            continue
        if definition.name in seen:
            issues.append(
                _issue(
                    "DUPLICATE_FEATURE_DEFINITION",
                    "feature names must be unique within a contract",
                    path=f"features[{index}].name",
                )
            )
        seen.add(definition.name)
        for issue in validate_feature_definition(definition).issues:
            issues.append(
                _issue(issue.code, issue.message, path=f"features[{index}].{issue.path}".rstrip("."), severity=issue.severity)
            )
    return DataQualityReport(tuple(issues))


@dataclass(frozen=True)
class FeatureObservation:
    """One feature value with a conservative local availability timestamp."""

    name: str
    value: float | int | None
    source: RawDataProvenance
    computed_at: datetime
    available_at: datetime
    transformation_version: str
    missing_reason: str | None = None

    def to_canonical_dict(self) -> dict[str, object]:
        value: float | None
        if self.value is None:
            value = None
        else:
            value = float(self.value)
        return {
            "name": self.name,
            "value": value,
            "source_digest": self.source.digest,
            "computed_at": _utc(self.computed_at),
            "available_at": _utc(self.available_at),
            "transformation_version": self.transformation_version,
            "missing_reason": self.missing_reason,
        }


def validate_feature_observation(
    observation: FeatureObservation,
    definition: FeatureDefinition,
    *,
    now: datetime | None = None,
) -> DataQualityReport:
    """Validate value, provenance, and availability for one feature."""

    issues: list[DataQualityIssue] = []
    for issue in validate_raw_data_provenance(observation.source, now=now).issues:
        issues.append(_issue(issue.code, issue.message, path=f"source.{issue.path}".rstrip("."), severity=issue.severity))
    if observation.name != definition.name:
        issues.append(_issue("FEATURE_NAME_MISMATCH", "observation does not match its contract definition", path="name"))
    if observation.transformation_version != definition.transformation_version:
        issues.append(
            _issue(
                "TRANSFORMATION_VERSION_MISMATCH",
                "observation transformation version does not match the feature contract",
                path="transformation_version",
            )
        )
    if not _is_aware(observation.computed_at):
        issues.append(_issue("NAIVE_TIMESTAMP", "computed_at must be timezone-aware", path="computed_at"))
    if not _is_aware(observation.available_at):
        issues.append(_issue("NAIVE_TIMESTAMP", "available_at must be timezone-aware", path="available_at"))
    if _is_aware(observation.computed_at) and _is_aware(observation.source.received_at):
        if observation.computed_at < observation.source.received_at:
            issues.append(
                _issue(
                    "FEATURE_COMPUTED_BEFORE_SOURCE_RECEIPT",
                    "computed_at cannot precede the local receipt of its source payload",
                    path="computed_at",
                )
            )
    if _is_aware(observation.available_at) and _is_aware(observation.computed_at):
        if observation.available_at < observation.computed_at:
            issues.append(
                _issue(
                    "FEATURE_AVAILABLE_BEFORE_COMPUTED",
                    "available_at cannot precede computed_at",
                    path="available_at",
                )
            )
    if _is_aware(observation.available_at) and _is_aware(observation.source.received_at):
        if observation.available_at < observation.source.received_at:
            issues.append(
                _issue(
                    "FEATURE_AVAILABLE_BEFORE_SOURCE_RECEIPT",
                    "available_at cannot precede the local receipt of its source payload",
                    path="available_at",
                )
            )
    if now is not None and _is_aware(now) and _is_aware(observation.available_at) and observation.available_at > now:
        issues.append(
            _issue(
                "FUTURE_FEATURE_AVAILABILITY",
                "available_at cannot be after validation time",
                path="available_at",
            )
        )
    value = observation.value
    if value is None:
        if not definition.allow_missing:
            issues.append(_issue("MISSING_FEATURE_VALUE", "the feature contract does not permit a missing value", path="value"))
        if not _nonempty_text(observation.missing_reason):
            issues.append(
                _issue(
                    "MISSING_REASON_REQUIRED",
                    "a missing value requires a non-empty missing_reason",
                    path="missing_reason",
                )
            )
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            issues.append(_issue("NONFINITE_OR_NONNUMERIC_FEATURE", "feature values must be finite numbers", path="value"))
        else:
            numeric_value = float(value)
            if definition.minimum is not None and numeric_value < definition.minimum:
                issues.append(_issue("FEATURE_VALUE_BELOW_MINIMUM", "feature value is below its contract minimum", path="value"))
            if definition.maximum is not None and numeric_value > definition.maximum:
                issues.append(_issue("FEATURE_VALUE_ABOVE_MAXIMUM", "feature value is above its contract maximum", path="value"))
        if observation.missing_reason is not None:
            issues.append(_issue("UNEXPECTED_MISSING_REASON", "missing_reason is only allowed for a missing value", path="missing_reason"))
    return DataQualityReport(tuple(issues))


@dataclass(frozen=True)
class PointInTimeFeatureVector:
    """A pregame feature vector frozen at one historical decision time."""

    event_id: str
    event_starts_at: datetime
    as_of: datetime
    contract: FeatureContract
    observations: tuple[FeatureObservation, ...]

    @property
    def digest(self) -> str:
        return _canonical_digest(self.to_canonical_dict())

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_starts_at": _utc(self.event_starts_at),
            "as_of": _utc(self.as_of),
            "feature_contract_sha256": self.contract.digest,
            "observations": [observation.to_canonical_dict() for observation in self.observations],
        }


def validate_point_in_time_feature_vector(
    vector: PointInTimeFeatureVector, *, now: datetime | None = None
) -> DataQualityReport:
    """Reject vectors that contain post-decision, post-start, or stale data."""

    issues: list[DataQualityIssue] = []
    if not _nonempty_text(vector.event_id):
        issues.append(_issue("REQUIRED_FIELD", "a non-empty value is required", path="event_id"))
    for field in ("event_starts_at", "as_of"):
        if not _is_aware(getattr(vector, field)):
            issues.append(_issue("NAIVE_TIMESTAMP", f"{field} must be timezone-aware", path=field))
    if _is_aware(vector.as_of) and _is_aware(vector.event_starts_at) and vector.as_of >= vector.event_starts_at:
        issues.append(
            _issue(
                "POST_START_FEATURE_VECTOR",
                "pregame feature vectors must be frozen before the scheduled event start",
                path="as_of",
            )
        )
    if now is not None:
        if not _is_aware(now):
            issues.append(_issue("NAIVE_TIMESTAMP", "validator now must be timezone-aware", path="now"))
        elif _is_aware(vector.as_of) and vector.as_of > now:
            issues.append(_issue("FUTURE_DECISION_TIME", "as_of cannot be after validation time", path="as_of"))
    contract_report = validate_feature_contract(vector.contract)
    for issue in contract_report.issues:
        issues.append(_issue(issue.code, issue.message, path=f"contract.{issue.path}".rstrip("."), severity=issue.severity))
    if not isinstance(vector.observations, tuple):
        issues.append(
            _issue(
                "MUTABLE_OBSERVATION_COLLECTION",
                "observations must be a tuple to preserve immutable feature evidence",
                path="observations",
            )
        )
        return DataQualityReport(tuple(issues))

    seen: set[str] = set()
    observed_names: set[str] = set()
    for index, observation in enumerate(vector.observations):
        if not isinstance(observation, FeatureObservation):
            issues.append(
                _issue("INVALID_FEATURE_OBSERVATION", "observations must contain FeatureObservation objects", path=f"observations[{index}]"))
            continue
        observed_names.add(observation.name)
        if observation.name in seen:
            issues.append(
                _issue(
                    "DUPLICATE_FEATURE_OBSERVATION",
                    "a feature may appear only once in a vector",
                    path=f"observations[{index}].name",
                )
            )
            continue
        seen.add(observation.name)
        definition = vector.contract.definition_for(observation.name)
        if definition is None:
            issues.append(
                _issue(
                    "UNDECLARED_FEATURE",
                    "feature is not declared in this feature contract",
                    path=f"observations[{index}].name",
                )
            )
            continue
        for issue in validate_feature_observation(observation, definition, now=now).issues:
            issues.append(
                _issue(
                    issue.code,
                    issue.message,
                    path=f"observations[{index}].{issue.path}".rstrip("."),
                    severity=issue.severity,
                )
            )
        if _is_aware(observation.available_at) and _is_aware(vector.as_of):
            if observation.available_at > vector.as_of:
                issues.append(
                    _issue(
                        "FEATURE_AVAILABLE_AFTER_AS_OF",
                        "feature became available after the claimed decision time",
                        path=f"observations[{index}].available_at",
                    )
                )
            if definition.max_age_seconds is not None:
                age_seconds = (vector.as_of - observation.available_at).total_seconds()
                if age_seconds > definition.max_age_seconds:
                    issues.append(
                        _issue(
                            "STALE_FEATURE",
                            f"feature age ({age_seconds:.0f}s) exceeds its {definition.max_age_seconds}s limit",
                            path=f"observations[{index}].available_at",
                        )
                    )
    for definition in (vector.contract.features if isinstance(vector.contract.features, tuple) else ()):
        if isinstance(definition, FeatureDefinition) and definition.required and definition.name not in observed_names:
            issues.append(
                _issue(
                    "MISSING_REQUIRED_FEATURE",
                    "a required feature observation is absent from the vector",
                    path=f"observations.{definition.name}",
                )
            )
    return DataQualityReport(tuple(issues))


@dataclass(frozen=True)
class LabeledTrainingExample:
    """A feature vector paired with an authoritative, immutable settlement."""

    row_id: str
    vector: PointInTimeFeatureVector
    target: int
    settled_at: datetime
    result_source: RawDataProvenance

    @property
    def digest(self) -> str:
        return _canonical_digest(self.to_canonical_dict())

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "vector_digest": self.vector.digest,
            "target": self.target,
            "settled_at": _utc(self.settled_at),
            "result_source_digest": self.result_source.digest,
        }


def validate_labeled_training_example(
    example: LabeledTrainingExample, *, now: datetime | None = None
) -> DataQualityReport:
    """Validate an outcome label while preserving the pregame feature guard."""

    issues: list[DataQualityIssue] = []
    if not _nonempty_text(example.row_id):
        issues.append(_issue("REQUIRED_FIELD", "a non-empty value is required", path="row_id"))
    for issue in validate_point_in_time_feature_vector(example.vector, now=now).issues:
        issues.append(_issue(issue.code, issue.message, path=f"vector.{issue.path}".rstrip("."), severity=issue.severity))
    for issue in validate_raw_data_provenance(example.result_source, now=now).issues:
        issues.append(_issue(issue.code, issue.message, path=f"result_source.{issue.path}".rstrip("."), severity=issue.severity))
    if isinstance(example.target, bool) or example.target not in (0, 1):
        issues.append(_issue("INVALID_BINARY_TARGET", "target must be the integer 0 or 1", path="target"))
    if not _is_aware(example.settled_at):
        issues.append(_issue("NAIVE_TIMESTAMP", "settled_at must be timezone-aware", path="settled_at"))
    elif _is_aware(example.vector.event_starts_at) and example.settled_at < example.vector.event_starts_at:
        issues.append(
            _issue(
                "RESULT_BEFORE_EVENT_START",
                "a final settlement cannot precede the scheduled event start",
                path="settled_at",
            )
        )
    if _is_aware(example.settled_at) and _is_aware(example.result_source.received_at):
        if example.result_source.received_at < example.settled_at:
            issues.append(
                _issue(
                    "RESULT_RECEIVED_BEFORE_SETTLEMENT",
                    "result source receipt cannot precede the final settlement timestamp",
                    path="result_source.received_at",
                )
            )
    feature_source_digests = {
        observation.source.digest for observation in example.vector.observations if isinstance(observation, FeatureObservation)
    }
    if example.result_source.digest in feature_source_digests:
        issues.append(
            _issue(
                "LABEL_SOURCE_USED_AS_FEATURE",
                "the exact result payload cannot also be a pregame feature source",
                path="result_source",
            )
        )
    return DataQualityReport(tuple(issues))


@dataclass(frozen=True)
class TrainingDatasetManifest:
    """A content-addressed record of exactly what a training run consumed."""

    dataset_name: str
    dataset_version: str
    feature_contract_sha256: str
    target_definition: str
    training_cutoff: datetime
    created_at: datetime
    code_revision: str
    split_strategy: str
    row_digests: tuple[str, ...]
    source_digests: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return len(self.row_digests)

    @property
    def digest(self) -> str:
        return _canonical_digest(self.to_canonical_dict())

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "feature_contract_sha256": self.feature_contract_sha256,
            "target_definition": self.target_definition,
            "training_cutoff": _utc(self.training_cutoff),
            "created_at": _utc(self.created_at),
            "code_revision": self.code_revision,
            "split_strategy": self.split_strategy,
            "row_digests": list(self.row_digests),
            "source_digests": list(self.source_digests),
        }

    def verify_examples(self, examples: Iterable[LabeledTrainingExample]) -> bool:
        """Return whether these exact rows still reproduce this manifest.

        A false result is intentional: callers must rebuild/review the dataset
        rather than silently substituting corrected, newer, or reordered rows.
        """

        rows = tuple(examples)
        if tuple(example.digest for example in rows) != self.row_digests:
            return False
        source_digests = _source_digests(rows)
        return source_digests == self.source_digests


def validate_training_dataset_manifest(manifest: TrainingDatasetManifest) -> DataQualityReport:
    """Validate a manifest loaded from storage before trusting its digest.

    This is intentionally separate from ``verify_examples``: the first check
    protects the manifest record itself, while the second compares an actual
    materialized row set to the manifest's immutable content references.
    """

    issues: list[DataQualityIssue] = []
    for field in ("dataset_name", "dataset_version", "target_definition", "code_revision", "split_strategy"):
        if not _nonempty_text(getattr(manifest, field, None)):
            issues.append(_issue("REQUIRED_FIELD", "a non-empty value is required", path=field))
    if not isinstance(manifest.feature_contract_sha256, str) or not _SHA256_RE.fullmatch(manifest.feature_contract_sha256):
        issues.append(
            _issue(
                "INVALID_FEATURE_CONTRACT_SHA256",
                "feature_contract_sha256 must be a lowercase, 64-character SHA-256 hexadecimal digest",
                path="feature_contract_sha256",
            )
        )
    for field in ("training_cutoff", "created_at"):
        if not _is_aware(getattr(manifest, field)):
            issues.append(_issue("NAIVE_TIMESTAMP", f"{field} must be timezone-aware", path=field))
    if _is_aware(manifest.training_cutoff) and _is_aware(manifest.created_at):
        if manifest.training_cutoff > manifest.created_at:
            issues.append(_issue("CUTOFF_AFTER_CREATION", "training_cutoff cannot be after created_at", path="training_cutoff"))
    for field in ("row_digests", "source_digests"):
        values = getattr(manifest, field)
        if not isinstance(values, tuple):
            issues.append(_issue("MUTABLE_MANIFEST_COLLECTION", f"{field} must be a tuple", path=field))
            continue
        if field == "row_digests" and not values:
            issues.append(_issue("EMPTY_TRAINING_DATASET", "row_digests cannot be empty", path=field))
        digest_values_are_text = all(isinstance(digest, str) for digest in values)
        if digest_values_are_text and len(values) != len(set(values)):
            issues.append(_issue("DUPLICATE_MANIFEST_DIGEST", f"{field} must not contain duplicates", path=field))
        if digest_values_are_text and tuple(sorted(values)) != values and field == "source_digests":
            issues.append(_issue("UNSORTED_SOURCE_DIGESTS", "source_digests must be sorted for deterministic manifests", path=field))
        for index, digest in enumerate(values):
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                issues.append(
                    _issue(
                        "INVALID_MANIFEST_DIGEST",
                        "row and source digests must be lowercase, 64-character SHA-256 hexadecimal digests",
                        path=f"{field}[{index}]",
                    )
                )
    return DataQualityReport(tuple(issues))


def _source_digests(examples: Sequence[LabeledTrainingExample]) -> tuple[str, ...]:
    digests: set[str] = set()
    for example in examples:
        digests.add(example.result_source.digest)
        for observation in example.vector.observations:
            if isinstance(observation, FeatureObservation):
                digests.add(observation.source.digest)
    return tuple(sorted(digests))


def build_training_dataset_manifest(
    *,
    dataset_name: str,
    dataset_version: str,
    contract: FeatureContract,
    examples: Iterable[LabeledTrainingExample],
    training_cutoff: datetime,
    created_at: datetime,
    code_revision: str,
    split_strategy: str,
) -> TrainingDatasetManifest:
    """Build a fail-closed manifest for chronological, fully-settled rows.

    The ``training_cutoff`` applies to both feature and result availability.
    That prevents an experiment from claiming it was trainable before its
    labels or source records actually existed in SAM's own archive.
    """

    issues: list[DataQualityIssue] = []
    for field, value in (
        ("dataset_name", dataset_name),
        ("dataset_version", dataset_version),
        ("code_revision", code_revision),
        ("split_strategy", split_strategy),
    ):
        if not _nonempty_text(value):
            issues.append(_issue("REQUIRED_FIELD", "a non-empty value is required", path=field))
    for field, value in (("training_cutoff", training_cutoff), ("created_at", created_at)):
        if not _is_aware(value):
            issues.append(_issue("NAIVE_TIMESTAMP", f"{field} must be timezone-aware", path=field))
    if _is_aware(training_cutoff) and _is_aware(created_at) and training_cutoff > created_at:
        issues.append(
            _issue(
                "CUTOFF_AFTER_CREATION",
                "training_cutoff cannot be after manifest creation time",
                path="training_cutoff",
            )
        )
    for issue in validate_feature_contract(contract).issues:
        issues.append(_issue(issue.code, issue.message, path=f"contract.{issue.path}".rstrip("."), severity=issue.severity))
    rows = tuple(examples)
    if not rows:
        issues.append(_issue("EMPTY_TRAINING_DATASET", "at least one settled training example is required", path="examples"))
    previous_as_of: datetime | None = None
    row_ids: set[str] = set()
    event_ids: set[str] = set()
    for index, example in enumerate(rows):
        for issue in validate_labeled_training_example(example, now=created_at if _is_aware(created_at) else None).issues:
            issues.append(
                _issue(issue.code, issue.message, path=f"examples[{index}].{issue.path}".rstrip("."), severity=issue.severity)
            )
        if example.vector.contract.digest != contract.digest:
            issues.append(
                _issue(
                    "FEATURE_CONTRACT_MISMATCH",
                    "all examples must use the manifest's exact feature contract",
                    path=f"examples[{index}].vector.contract",
                )
            )
        if example.row_id in row_ids:
            issues.append(_issue("DUPLICATE_TRAINING_ROW", "row_id must be unique", path=f"examples[{index}].row_id"))
        row_ids.add(example.row_id)
        if example.vector.event_id in event_ids:
            issues.append(
                _issue(
                    "DUPLICATE_TRAINING_EVENT",
                    "one training manifest may contain at most one decision row per event",
                    path=f"examples[{index}].vector.event_id",
                )
            )
        event_ids.add(example.vector.event_id)
        if _is_aware(example.vector.as_of):
            if previous_as_of is not None and example.vector.as_of < previous_as_of:
                issues.append(
                    _issue(
                        "NONCHRONOLOGICAL_TRAINING_ROWS",
                        "examples must be supplied in ascending decision-time order",
                        path=f"examples[{index}].vector.as_of",
                    )
                )
            previous_as_of = example.vector.as_of
            if _is_aware(training_cutoff) and example.vector.as_of > training_cutoff:
                issues.append(
                    _issue(
                        "FEATURE_DECISION_AFTER_CUTOFF",
                        "feature vector decision time is after training_cutoff",
                        path=f"examples[{index}].vector.as_of",
                    )
                )
        if _is_aware(example.settled_at) and _is_aware(training_cutoff) and example.settled_at > training_cutoff:
            issues.append(
                _issue(
                    "SETTLEMENT_AFTER_CUTOFF",
                    "target settlement is after training_cutoff",
                    path=f"examples[{index}].settled_at",
                )
            )
        if _is_aware(example.result_source.received_at) and _is_aware(training_cutoff):
            if example.result_source.received_at > training_cutoff:
                issues.append(
                    _issue(
                        "LABEL_AVAILABLE_AFTER_CUTOFF",
                        "result source arrived after training_cutoff",
                        path=f"examples[{index}].result_source.received_at",
                    )
                )
    DataQualityReport(tuple(issues)).require_valid("training dataset")
    manifest = TrainingDatasetManifest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        feature_contract_sha256=contract.digest,
        target_definition=contract.target_definition,
        training_cutoff=training_cutoff,
        created_at=created_at,
        code_revision=code_revision,
        split_strategy=split_strategy,
        row_digests=tuple(example.digest for example in rows),
        source_digests=_source_digests(rows),
    )
    validate_training_dataset_manifest(manifest).require_valid("training dataset manifest")
    return manifest


class PointInTimeDataRepository(Protocol):
    """Persistence boundary for append-only, reproducible model evidence.

    Implementations must validate before append, preserve the original object
    under its digest, and never overwrite a correcting provider payload.  An
    exact lookup must not return a nearest timestamp or recompute from newer
    data; it returns ``None`` when the requested immutable vector is absent.
    """

    def append_raw_provenance(self, record: RawDataProvenance) -> str:
        """Persist one immutable receipt and return its digest."""

    def append_feature_vector(self, vector: PointInTimeFeatureVector) -> str:
        """Persist one validated, immutable feature vector and return its digest."""

    def append_training_manifest(self, manifest: TrainingDatasetManifest) -> str:
        """Persist one immutable training manifest and return its digest."""

    def get_feature_vector_exact(
        self, *, event_id: str, as_of: datetime, feature_contract_sha256: str
    ) -> PointInTimeFeatureVector | None:
        """Return only an exact persisted as-of vector, otherwise ``None``."""
