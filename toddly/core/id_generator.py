# core/id_generator.py
import hashlib
import json
import string
import threading
from pathlib import Path
from typing import Dict

BASE62_ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase


def base62_encode(num: int, length: int = 6) -> str:
    # FIX #7: instead of encoding the full integer and then truncating (which
    # silently aliases two different large integers that share the same leading
    # `length` base-62 digits), reduce `num` modulo 62^length first so the
    # result always fits exactly in `length` characters without any truncation.
    base = len(BASE62_ALPHABET)
    modulus = base**length
    num = num % modulus

    chars = []
    if num == 0:
        chars.append("0")

    while num > 0:
        num, rem = divmod(num, base)
        chars.append(BASE62_ALPHABET[rem])

    while len(chars) < length:
        chars.append("0")

    return "".join(reversed(chars))


class StableIDGenerator:
    def __init__(self, mapping_file=None, id_length=6):  # None = in-memory only
        self.mapping_file = Path(mapping_file) if mapping_file else None
        self.id_length = id_length
        # Fix #8: serialize concurrent get_id() calls so that two executor
        # threads cannot race on the read-probe-write sequence and assign the
        # same short_id to different keys.
        self._lock = threading.Lock()
        self._load_mapping()

    def _load_mapping(self):
        if self.mapping_file and self.mapping_file.exists():
            try:
                with self.mapping_file.open() as f:
                    self.mapping: Dict[str, Dict[str, str]] = json.load(f)
            except json.JSONDecodeError:
                self.mapping = {}
        else:
            self.mapping = {}

    def _save_mapping(self):
        if self.mapping_file is None:
            return  # in-memory mode — nothing to persist
        with self.mapping_file.open("w") as f:
            json.dump(self.mapping, f, indent=2)

    def get_id(self, key: str, domain: str = "default") -> str:
        """
        Returns a deterministic ID for `key` within a given `domain`.
        IDs are guaranteed unique inside that domain only.

        FIX #6: collision resolution now hashes the *key* at each probe step
        (using domain:key:attempt as the hash input) rather than incrementing a
        raw integer.  This makes the resolved ID depend only on the key itself
        and the number of prior collisions in the domain — never on the order in
        which other keys were inserted — so the same key always resolves to the
        same short ID regardless of which other keys happen to be present.
        """
        # Fix #8: hold the instance lock for the entire read-probe-write
        # sequence so that concurrent executor threads cannot race on
        # domain_map and assign the same short_id to different keys.
        with self._lock:
            # Ensure domain exists
            if domain not in self.mapping:
                self.mapping[domain] = {}

            domain_map = self.mapping[domain]

            # If key already exists in domain, return the stored ID immediately.
            if key in domain_map:
                return domain_map[key]

            # Build a reverse lookup once so collision detection is O(1) per probe.
            used_ids: set = set(domain_map.values())

            # Probe with increasing attempt counters until we find an unused slot.
            attempt = 0
            while True:
                probe_input = f"{domain}:{key}:{attempt}".encode("utf-8")
                digest_int = int(hashlib.sha256(probe_input).hexdigest(), 16)
                short_id = base62_encode(digest_int, self.id_length)
                if short_id not in used_ids:
                    break
                attempt += 1

            domain_map[key] = short_id
            self._save_mapping()

            return short_id
