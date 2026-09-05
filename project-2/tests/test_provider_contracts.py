import unittest
from datetime import datetime, timezone

from sam_analytics.provider_contracts import (
    ApprovedProviderContract,
    ProviderContractRegistry,
    ProviderContractViolation,
    ProviderUse,
    validate_private_payload_metadata_for_use,
    validate_provider_use,
)
from sam_analytics.raw_payload_store import RawPayloadMetadata


class ProviderContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = ApprovedProviderContract(
            provider="the_odds_api",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            permitted_source_types=frozenset({"odds", "historical_odds"}),
        )
        self.registry = ProviderContractRegistry([self.contract])

    def test_exact_approved_contract_authorizes_private_ingestion_and_derived_output(self):
        ingestion = self.registry.authorize_ingestion(
            provider="the_odds_api",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            source_type="odds",
        )
        derived = self.registry.authorize_derived_exposure(
            provider="the_odds_api",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            source_type="historical_odds",
        )

        self.assertEqual(ingestion.exposure, "private_raw")
        self.assertEqual(derived.exposure, "derived")
        self.assertEqual(derived.license_version, "terms-2026-08-31")
        self.assertEqual(self.registry.contracts, (self.contract,))

    def test_rejects_unapproved_scope_version_provider_and_source_type(self):
        for overrides in (
            {"provider": "unapproved_provider"},
            {"license_scope": "external_redistribution"},
            {"license_version": "terms-older"},
            {"source_type": "results"},
        ):
            request = {
                "provider": "the_odds_api",
                "license_scope": "internal_analytics_only",
                "license_version": "terms-2026-08-31",
                "source_type": "odds",
            }
            request.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaises(ProviderContractViolation):
                    self.registry.authorize_ingestion(**request)

    def test_contract_is_derived_only_and_public_raw_use_is_not_a_valid_mode(self):
        with self.assertRaises(ProviderContractViolation):
            ApprovedProviderContract(
                provider="the_odds_api",
                license_scope="internal_analytics_only",
                license_version="terms-2026-08-31",
                permitted_source_types=frozenset({"odds"}),
                derived_only=False,
            )

    def test_private_evidence_metadata_must_match_the_exact_ingestion_authorization(self):
        use = self.registry.authorize_ingestion(
            provider="the_odds_api",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            source_type="odds",
        )
        timestamp = datetime(2026, 9, 4, 18, tzinfo=timezone.utc)
        metadata = RawPayloadMetadata(
            provider="the_odds_api",
            provider_record_id="request-1",
            source_type="odds",
            captured_at=timestamp,
            received_at=timestamp,
            schema_version="v4",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
        )
        validate_private_payload_metadata_for_use(metadata, use)
        mismatched = RawPayloadMetadata(
            provider="the_odds_api",
            provider_record_id="request-1",
            source_type="odds",
            captured_at=timestamp,
            received_at=timestamp,
            schema_version="v4",
            license_scope="internal_analytics_only",
            license_version="terms-newer",
        )
        with self.assertRaises(ProviderContractViolation):
            validate_private_payload_metadata_for_use(mismatched, use)
        derived = self.registry.authorize_derived_exposure(
            provider="the_odds_api",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            source_type="odds",
        )
        with self.assertRaises(ProviderContractViolation):
            validate_private_payload_metadata_for_use(metadata, derived)
        with self.assertRaises(ProviderContractViolation):
            validate_provider_use(
                ProviderUse(
                    provider="the_odds_api",
                    license_scope="internal_analytics_only",
                    license_version="terms-2026-08-31",
                    source_type="odds",
                    exposure="public_raw",  # type: ignore[arg-type]
                )
            )

    def test_source_type_collection_is_frozen_and_duplicate_policy_is_rejected(self):
        source_types = {"odds"}
        contract = ApprovedProviderContract(
            provider="another_provider",
            license_scope="research",
            license_version="v1",
            permitted_source_types=source_types,  # type: ignore[arg-type]
        )
        source_types.add("results")
        self.assertEqual(contract.permitted_source_types, frozenset({"odds"}))
        with self.assertRaises(ProviderContractViolation):
            ProviderContractRegistry([self.contract, self.contract])

    def test_contract_rejects_unsafe_policy_tokens_and_empty_registry(self):
        with self.assertRaises(ProviderContractViolation):
            ApprovedProviderContract(
                provider="The Odds API",
                license_scope="internal_analytics_only",
                license_version="v1",
                permitted_source_types=frozenset({"odds"}),
            )
        with self.assertRaises(ProviderContractViolation):
            ProviderContractRegistry([])


if __name__ == "__main__":
    unittest.main()
