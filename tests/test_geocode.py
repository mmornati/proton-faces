import pytest

import geocode


class _FakeRG:
    """Minimal fake for the reverse_geocoder module."""

    def __init__(self, results):
        self._results = results

    def search(self, points):
        if isinstance(points, tuple):
            r = self._results.get(points)
            return (r if r else None,)
        return [self._results.get(p, (None,)) for p in points]


@pytest.fixture
def fake_rg(monkeypatch):
    rg = _FakeRG(
        {
            (48.8584, 2.2945): {"name": "Paris", "admin1": "Île-de-France", "cc": "FR"},
            (40.7128, -74.006): {"name": "New York", "admin1": "New York", "cc": "US"},
        }
    )
    monkeypatch.setattr(geocode, "_rg", rg)
    return rg


def test_reverse_geocode_known_point(fake_rg):
    assert geocode.reverse_geocode(48.8584, 2.2945) == "Paris, Île-de-France"


def test_reverse_geocode_unknown_point(fake_rg):
    assert geocode.reverse_geocode(0.0, 0.0) is None


def test_reverse_geocode_no_rg(monkeypatch):
    monkeypatch.setattr(geocode, "_rg", False)
    assert geocode.reverse_geocode(48.8584, 2.2945) is None


def test_reverse_geocode_many_batches_and_rounds(fake_rg, monkeypatch):
    calls = []

    def spy_search(points):
        calls.append(points)
        return [fake_rg._results.get(p) for p in points]

    monkeypatch.setattr(fake_rg, "search", spy_search)
    out = geocode.reverse_geocode_many([(48.85844, 2.29454), (40.7128, -74.0060), (1.5, 2.5)])
    assert out[(48.85844, 2.29454)] == "Paris, Île-de-France"
    assert out[(40.7128, -74.0060)] == "New York, New York"
    assert out[(1.5, 2.5)] is None
    assert len(calls) == 1
    assert (48.8584, 2.2945) in calls[0]


def test_reverse_geocode_many_no_rg(monkeypatch):
    monkeypatch.setattr(geocode, "_rg", False)
    out = geocode.reverse_geocode_many([(48.8584, 2.2945)])
    assert out == {(48.8584, 2.2945): None}


def test_reverse_geocode_none_coords(fake_rg):
    assert geocode.reverse_geocode(None, 2.2945) is None
    assert geocode.reverse_geocode(48.8584, None) is None
