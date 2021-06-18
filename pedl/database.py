import gzip
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import indra
from indra import statements
from indra.sources.biopax.api import process_owl_str
from tqdm import tqdm

from pedl.utils import cache_root, cached_path, root, get_gene_mapping


class PathwayCommonsDB:
    def __init__(self, name, gene_universe):
        url = f"https://www.pathwaycommons.org/archives/PC2/v12/PathwayCommons12.{name}.BIOPAX.owl.gz"
        self.name = name
        self.gene_universe = set(gene_universe)
        self.path = cached_path(url, cache_dir=cache_root / "pc")
        self._relations_by_pair = self._get_statements_by_pair()

    def get_statements(self, p1: str, p2: str) -> List[indra.statements.Statement]:
        return self._relations_by_pair.get((p1, p2), [])

    def _get_statements_by_pair(self) -> Dict[Tuple[str, str], List[statements.Statement]]:
        statements_by_pair = defaultdict(list)

        with gzip.open(self.path) as f:
            processor = process_owl_str(f.read().decode())
            for stmt in tqdm(processor.statements, desc=f"Indexing {self.name}"):
                if isinstance(stmt, statements.Modification):
                    if "EGID" in stmt.enz.db_refs and "EGID" in stmt.sub.db_refs:
                        enz = stmt.enz.db_refs["EGID"]
                        sub = stmt.sub.db_refs["EGID"]
                        if sub in self.gene_universe:
                            statements_by_pair[(enz, sub)].append(stmt)

        return statements_by_pair


if __name__ == '__main__':
    PathwayCommonsDB("pid", gene_universe={"5970", "8517"})

