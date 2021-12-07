import gzip
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Union, Optional, List, Set, Tuple, Dict
from urllib.parse import urlparse

import requests
import bioc
import numpy as np
from transformers.file_utils import default_cache_path
from segtok.segmenter import split_multi


cache_root = Path(
    os.getenv("PEDL_CACHE", Path(default_cache_path).parent.parent / "pedl")
)
if not cache_root.exists():
    os.makedirs(cache_root, exist_ok=True)

root = Path(__file__).parent


class Sentence:
    def __init__(
        self,
        text: str,
        start_pos: int,
        pmid: Optional[str] = None,
        text_blinded: Optional[str] = None,
    ):
        self.text = text
        self.pmid = pmid
        self.start_pos = start_pos
        self.end_pos = start_pos + len(text)
        self.text_blinded = text_blinded

    def __str__(self):
        return self.text

    def __repr__(self):
        return str(self)

    def get_unmarked_text(self):
        return (
            self.text.replace("<e1>", "")
            .replace("</e1>", "")
            .replace("<e2>", "")
            .replace("</e2>", "")
        )


class SegtokSentenceSplitter:
    """
    For further details see: https://github.com/fnl/segtok
    """

    def split(self, text: str) -> List[Sentence]:
        sentences = []
        offset = 0

        plain_sentences = split_multi(text)
        for sentence in plain_sentences:
            sentence_offset = text.find(sentence, offset)

            if sentence_offset == -1:
                raise AssertionError(
                    f"Can't find offset for sentences {plain_sentences} "
                    f"starting from {offset}"
                )

            sentences += [
                Sentence(
                    text=sentence,
                    start_pos=sentence_offset,
                )
            ]

            offset += len(sentence)

        return sentences


from tqdm import tqdm as _tqdm, tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_pmid(document: bioc.BioCDocument) -> Tuple[str, int]:
    infons = document.passages[0].infons
    if "article-id_pmid" in infons:
        pmid = document.passages[0].infons["article-id_pmid"]
        is_fulltext = int(document.id == infons.get("article-id_pmc", None))
    else:
        pmid = document.id
        is_fulltext = 0

    return pmid, is_fulltext




class Tqdm:
    # These defaults are the same as the argument defaults in tqdm.
    default_mininterval: float = 0.1

    @staticmethod
    def set_default_mininterval(value: float) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) -> None:
        """
        If ``use_slower_interval`` is ``True``, we will dramatically slow down ``tqdm's`` default
        output rate.  ``tqdm's`` default output rate is great for interactively watching progress,
        but it is not great for log files.  You might want to set this if you are primarily going
        to be looking at output through log files, not the terminal.
        """
        if use_slower_interval:
            Tqdm.default_mininterval = 10.0
        else:
            Tqdm.default_mininterval = 0.1

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {"mininterval": Tqdm.default_mininterval, **kwargs}

        return _tqdm(*args, **new_kwargs)


def get_from_cache(url: str, cache_dir: Path = None) -> Path:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    # get cache path to put the file
    cache_path = cache_dir / filename
    if cache_path.exists():
        return cache_path

    # make HEAD request to check ETag
    response = requests.head(url, headers={"User-Agent": "Flair"}, allow_redirects=True)
    if response.status_code != 200:
        raise IOError(
            f"HEAD request failed for url {url} with status code {response.status_code}."
        )

    # add ETag to filename if it exists
    # etag = response.headers.get("ETag")

    if not cache_path.exists():
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        fd, temp_filename = tempfile.mkstemp()

        # GET file object
        req = requests.get(url, stream=True, headers={"User-Agent": "PEDL"})
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = Tqdm.tqdm(unit="B", total=total)
        with open(temp_filename, "wb") as temp_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)

        progress.close()

        shutil.copyfile(temp_filename, str(cache_path))
        os.close(fd)
        os.remove(temp_filename)

    return cache_path


def cached_path(url_or_filename: str, cache_dir: Union[str, Path]) -> Path:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if type(cache_dir) is str:
        cache_dir = Path(cache_dir)
    dataset_cache = Path(cache_root) / cache_dir

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, dataset_cache)
    elif parsed.scheme == "" and Path(url_or_filename).exists():
        # File, and it exists.
        return Path(url_or_filename)
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename)
        )


def unpack_file(file: Path, unpack_to: Path, mode: str = None, keep: bool = True):
    """
    Unpacks a file to the given location.

    :param file Archive file to unpack
    :param unpack_to Destination where to store the output
    :param mode Type of the archive (zip, tar, gz, targz, rar)
    :param keep Indicates whether to keep the archive after extraction or delete it
    """
    if mode == "zip" or (mode is None and str(file).endswith("zip")):
        from zipfile import ZipFile

        with ZipFile(file, "r") as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(unpack_to)

    elif mode == "targz" or (
        mode is None and str(file).endswith("tar.gz") or str(file).endswith("tgz")
    ):
        import tarfile

        with tarfile.open(file, "r:gz") as tarObj:
            tarObj.extractall(unpack_to)

    elif mode == "tar" or (mode is None and str(file).endswith("tar")):
        import tarfile

        with tarfile.open(file, "r") as tarObj:
            tarObj.extractall(unpack_to)

    elif mode == "gz" or (mode is None and str(file).endswith("gz")):
        import gzip

        with gzip.open(str(file), "rb") as f_in:
            with open(str(unpack_to), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    elif mode == "rar" or (mode is None and str(file).endswith("rar")):
        import patoolib

        patoolib.extract_archive(str(file), outdir=unpack_to, interactive=False)

    else:
        if mode is None:
            raise AssertionError(f"Can't infer archive type from {file}")
        else:
            raise AssertionError(f"Unsupported mode {mode}")

    if not keep:
        os.remove(str(file))


def replace_consistently(offset, length, replacement, text, offsets):
    delta_len = len(replacement) - length
    new_text = text[:offset] + replacement + text[offset + length :]
    new_offsets = offsets.copy()
    new_offsets[offsets >= offset + length] += delta_len

    return new_text, new_offsets

def insert_consistently(offset, insertion, text, starts, ends):
    new_text = text[:offset] + insertion + text[offset:]
    new_starts = starts.copy()
    new_ends = ends.copy()
    new_starts[starts >= offset] += len(insertion)
    new_ends[ends >= offset] += len(insertion)

    return new_text, new_starts, new_ends


def delete_consistently(from_idx, to_idx , text, starts, ends):
    assert to_idx >= from_idx
    new_text = text[:from_idx] + text[to_idx:]
    new_starts = starts.copy()
    new_ends = ends.copy()
    new_starts[(from_idx <= starts) & (starts <= to_idx)] = from_idx
    new_ends[(from_idx <= ends) & (ends <= to_idx)] = from_idx
    new_starts[starts > to_idx] -= (to_idx - from_idx)
    new_ends[ends > to_idx] -= (to_idx - from_idx)

    return new_text, new_starts, new_ends


def replace_consistently_dict(text, span_to_replacement):
    offsets = list(span_to_replacement)
    replacements = [span_to_replacement[i] for i in offsets]

    starts = np.array([i[0] for i in offsets])
    ends = np.array([i[1] for i in offsets])

    for i, replacement in enumerate(replacements):
        text, starts, ends = delete_consistently(from_idx=starts[i],
                                                 to_idx=ends[i], starts=starts,
                                                 ends=ends, text=text)
        text, starts, ends = insert_consistently(offset=starts[i], insertion=replacement,
                                                 starts=starts, ends=ends, text=text)

    return text


def get_homologue_mapping(
    expand_species_names: List[str], protein_universe: Set[str]
) -> Dict[str, Set[str]]:
    """
    Get mapping from genes in `expand_species` that are not in `protein_universe` to all
    homologous gene_id in `protein_universe`
    """
    species_to_tax_id = {
        "human": "9606",
        "mouse": "10090",
        "rat": "10116",
        "zebrafish": "7955",
    }
    expand_species = [species_to_tax_id[i] for i in expand_species_names]
    homologue_mapping = defaultdict(set)
    gene_id_to_cluster_id = {}
    cluster_id_to_gene_ids = defaultdict(set)

    with open(root / "data" / "HOM_AllOrganism.rpt") as f:
        next(f)
        for line in f:
            fields = line.strip().split("\t")
            if not fields:
                continue
            cluster_id = fields[0]
            taxon_id = fields[2]
            gene_id = fields[4]
            if taxon_id in expand_species or gene_id in protein_universe:
                cluster_id_to_gene_ids[cluster_id].add(gene_id)
            if gene_id in protein_universe:
                gene_id_to_cluster_id[gene_id] = cluster_id

        for gene_id in protein_universe:
            if gene_id in gene_id_to_cluster_id:
                cluster_id = gene_id_to_cluster_id[gene_id]
                for mapped_gene_id in cluster_id_to_gene_ids[cluster_id]:
                    if mapped_gene_id not in protein_universe:
                        homologue_mapping[mapped_gene_id].add(gene_id)

    return dict(homologue_mapping)


@dataclass(unsafe_hash=True)
class Entity:
    cuid: str
    type: str

    def to_json(self):
        return [self.cuid, self.type]

    def __str__(self):
        return f"{self.cuid}|{self.type}"

    def __lt__(self, other):
        if not isinstance(other, Entity):
            raise TypeError

        return str(self) < str(other)


def get_geneid_to_name():
    with open(root / "data" / "geneid_to_name.json") as f:
        return json.load(f)


def get_gene_mapping(from_db: str, to_db: str):
    final_mapping = defaultdict(set)
    uniprot_to_from_db = defaultdict(set)
    uniprot_to_to_db = defaultdict(set)
    uniprot_mapping_file = cached_path(
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz",
        cache_dir=cache_root,
    )
    with gzip.open(uniprot_mapping_file) as f:
        for line in f:
            uniprot_id, other_db, other_db_id = line.decode().strip().split("\t")
            if other_db == from_db:
                uniprot_to_from_db[uniprot_id].add(other_db_id)
            if other_db == to_db:
                uniprot_to_to_db[uniprot_id].add(other_db_id)

        for k in uniprot_to_from_db:
            from_ids = uniprot_to_from_db[k]
            to_ids = uniprot_to_to_db[k]

            for from_id in from_ids:
                for to_id in to_ids:
                    final_mapping[from_id].add(to_id)

    return dict(final_mapping)


def build_summary_table(
    raw_dir: Path, score_cutoff: float = 0.0, no_association_type: bool = False
) -> List[Tuple[str, float]]:
    table = []

    rel_to_score_sum = defaultdict(float)
    rel_to_score_max = defaultdict(float)

    hgnc_symbols = set(get_hgnc_symbol_to_gene_id().keys())

    files = list(raw_dir.glob("*.txt"))
    for file in tqdm(files):
        with file.open() as f:
            p1 = file.name.replace(".txt", "").split("_")[0]

            if p1 in hgnc_symbols:
                # P1_P2 or P1_P_2
                p2 = "_".join(file.name.replace(".txt", "").split("_")[1:])
            else:
                # P_1_P2 or P_1_P_2
                p1 = "_".join(file.name.replace(".txt", "").split("_")[:2])
                p2 = "_".join(file.name.replace(".txt", "").split("_")[2:])
            for line in f:
                fields = line.strip().split()
                if fields:
                    if no_association_type:
                        p1_unified, p2_unified = sorted([p1, p2])
                        rel = (p1_unified, "association", p2_unified)
                    else:
                        rel = (p1, fields[0], p2)
                    if float(fields[1]) >= score_cutoff:
                        rel_to_score_sum[rel] += float(fields[1])
                        rel_to_score_max[rel] = max(
                            float(fields[1]), rel_to_score_max[rel]
                        )

    for rel, score_sum in rel_to_score_sum.items():
        score_max = rel_to_score_max[rel]
        row = rel + (score_sum, score_max)
        table.append(row)

    return sorted(table, key=itemgetter(3), reverse=True)


def get_hgnc_symbol_to_gene_id():
    hgnc_symbol_to_gene_id = {}
    url = "http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt"
    with open(cached_path(url, cache_dir=cache_root), encoding="utf8") as f:
        next(f)
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) > 18:
                symbol = fields[1]
                gene_id = fields[18]
                hgnc_symbol_to_gene_id[symbol] = gene_id

    return hgnc_symbol_to_gene_id
