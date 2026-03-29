# core/id_generator.py
import hashlib
import string
import json
from pathlib import Path
from typing import Dict

BASE62_ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase

def base62_encode(num: int, length: int = 6) -> str:
    chars = []
    base = len(BASE62_ALPHABET)

    if num == 0:
        chars.append('0')

    while num > 0:
        num, rem = divmod(num, base)
        chars.append(BASE62_ALPHABET[rem])

    while len(chars) < length:
        chars.append('0')

    return ''.join(reversed(chars))[:length]


class StableIDGenerator:
    def __init__(self, mapping_file=None, id_length=6):  # None = in-memory only
        self.mapping_file = Path(mapping_file) if mapping_file else None
        self.id_length = id_length
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
            return          # in-memory mode — nothing to persist
        with self.mapping_file.open("w") as f:
            json.dump(self.mapping, f, indent=2)

    def get_id(self, key: str, domain: str = "default") -> str:
        """
        Returns a deterministic ID for `key` within a given `domain`.
        IDs are guaranteed unique inside that domain only.
        """

        # Ensure domain exists
        if domain not in self.mapping:
            self.mapping[domain] = {}

        domain_map = self.mapping[domain]

        # If key already exists in domain
        if key in domain_map:
            return domain_map[key]

        # Generate deterministic hash
        digest = hashlib.sha256(f"{domain}:{key}".encode("utf-8")).hexdigest()
        digest_int = int(digest, 16)
        short_id = base62_encode(digest_int, self.id_length)

        # Handle collisions ONLY inside this domain
        while short_id in domain_map.values():
            digest_int += 1
            short_id = base62_encode(digest_int, self.id_length)

        domain_map[key] = short_id
        self._save_mapping()

        return short_id
