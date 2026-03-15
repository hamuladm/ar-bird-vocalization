import csv
import os
import logging
from pathlib import Path
from typing import Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from config import TAXONOMY_API_URL, TAXONOMY_CACHE

logger = logging.getLogger(__name__)


class TaxonomyMapper:
    def __init__(self, cache_path: str | Path | None = None):
        self.cache_path = Path(cache_path or TAXONOMY_CACHE)
        self._code_to_family: Dict[str, str] = {}
        self._code_to_order: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            self._fetch_and_cache()
        self._parse_csv()

    def _fetch_and_cache(self) -> None:
        api_key = os.environ.get("XC_API_KEY")
        logger.info("Fetching eBird taxonomy from API...")
        req = Request(TAXONOMY_API_URL, headers={"X-eBirdApiToken": api_key})
        with urlopen(req) as resp:
            data = resp.read().decode("utf-8")

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(data, encoding="utf-8")
        logger.info(f"Cached taxonomy to {self.cache_path}")

    def _parse_csv(self) -> None:
        with open(self.cache_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("SPECIES_CODE") or row.get("species_code")
                family = (
                    row.get("FAMILY_SCI_NAME")
                    or row.get("family_sci_name")
                    or row.get("FAMILY")
                    or row.get("family")
                )
                order = row.get("ORDER1") or row.get("order") or row.get("ORDER")
                if code:
                    if family:
                        self._code_to_family[code] = family
                    if order:
                        self._code_to_order[code] = order

    def get_family(self, ebird_code: str) -> Optional[str]:
        return self._code_to_family.get(ebird_code)

    def get_order(self, ebird_code: str) -> Optional[str]:
        return self._code_to_order.get(ebird_code)

    def same_family(self, code_a: str, code_b: str) -> bool:
        fam_a = self._code_to_family.get(code_a)
        fam_b = self._code_to_family.get(code_b)
        if fam_a is None or fam_b is None:
            return False
        return fam_a == fam_b
