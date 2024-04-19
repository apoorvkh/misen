from unittest.mock import patch


class deterministic_hashing:
    def __enter__(self):
        self.patch_hash = patch("builtins.hash", det_hash)
        self.patch_hash.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patch_hash.__exit__(exc_type, exc_val, exc_tb)


def det_hash(obj):
    return 10
