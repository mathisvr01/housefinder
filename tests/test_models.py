from __future__ import annotations

from housefinder.models import ListingProfile


def test_structured_output_schema_requires_all_profile_fields() -> None:
    schema = ListingProfile.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False


def test_unknown_profile_never_asserts_absence() -> None:
    profile = ListingProfile.unknown()
    assert profile.absence_of_outbuildings_conclusive is False
    assert profile.outbuildings_confidence == 0
