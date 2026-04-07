"""Tests for cuddlytoddly.core.id_generator."""
import json

from cuddlytoddly.core.id_generator import StableIDGenerator, base62_encode

# ── base62_encode ──────────────────────────────────────────────────────────────

class TestBase62Encode:
    def test_zero(self):
        assert base62_encode(0) == "000000"

    def test_length_padded(self):
        result = base62_encode(1, length=6)
        assert len(result) == 6

    def test_deterministic(self):
        assert base62_encode(12345) == base62_encode(12345)

    def test_large_number(self):
        result = base62_encode(999_999_999)
        assert len(result) == 6

    def test_different_inputs_differ(self):
        assert base62_encode(1) != base62_encode(2)

    def test_custom_length(self):
        assert len(base62_encode(1, length=4)) == 4


# ── StableIDGenerator ─────────────────────────────────────────────────────────

class TestStableIDGenerator:
    def test_deterministic_id(self, tmp_path):
        gen = StableIDGenerator(mapping_file=tmp_path / "ids.json")
        id1 = gen.get_id("hello")
        id2 = gen.get_id("hello")
        assert id1 == id2

    def test_different_keys_different_ids(self, tmp_path):
        gen = StableIDGenerator(mapping_file=tmp_path / "ids.json")
        assert gen.get_id("a") != gen.get_id("b")

    def test_domain_isolation(self, tmp_path):
        gen = StableIDGenerator(mapping_file=tmp_path / "ids.json")
        id_default = gen.get_id("key", domain="default")
        id_other = gen.get_id("key", domain="other")
        # Same key, different domains — IDs may collide by hash but are tracked separately
        assert gen.mapping["default"]["key"] == id_default
        assert gen.mapping["other"]["key"] == id_other

    def test_persists_to_file(self, tmp_path):
        path = tmp_path / "ids.json"
        gen = StableIDGenerator(mapping_file=path)
        gen.get_id("persistent_key")
        assert path.exists()
        data = json.loads(path.read_text())
        assert "default" in data
        assert "persistent_key" in data["default"]

    def test_loads_from_existing_file(self, tmp_path):
        path = tmp_path / "ids.json"
        gen1 = StableIDGenerator(mapping_file=path)
        id1 = gen1.get_id("key")

        gen2 = StableIDGenerator(mapping_file=path)
        id2 = gen2.get_id("key")
        assert id1 == id2

    def test_in_memory_mode_no_file_written(self, tmp_path):
        gen = StableIDGenerator(mapping_file=None)
        gen.get_id("x")
        # Nothing should be written anywhere
        assert list(tmp_path.iterdir()) == []

    def test_in_memory_mode_still_deterministic_within_session(self):
        gen = StableIDGenerator(mapping_file=None)
        id1 = gen.get_id("test")
        id2 = gen.get_id("test")
        assert id1 == id2

    def test_collision_handled(self, tmp_path):
        """When two keys hash to the same short ID, one gets incremented."""
        gen = StableIDGenerator(mapping_file=tmp_path / "ids.json", id_length=1)
        # With id_length=1 there are only 62 possible IDs — force a domain fill
        ids = set()
        for i in range(10):
            ids.add(gen.get_id(f"key_{i}"))
        assert len(ids) == 10  # all unique within domain

    def test_corrupted_file_resets_gracefully(self, tmp_path):
        path = tmp_path / "ids.json"
        path.write_text("not valid json")
        gen = StableIDGenerator(mapping_file=path)
        result = gen.get_id("test")
        assert isinstance(result, str)
        assert len(result) > 0


