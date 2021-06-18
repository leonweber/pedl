import gzip
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import shutil
import sys
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List, Set, Tuple, Dict
from urllib.parse import urlparse

import requests
from lxml import etree
import yaml
import bioc
import numpy as np
from flair.tokenization import SegtokSentenceSplitter
from transformers.file_utils import default_cache_path



cache_root = Path(os.getenv("PEDL_CACHE", Path(default_cache_path).parent.parent / "pedl"))
if not cache_root.exists():
    os.makedirs(cache_root, exist_ok=True)

root = Path(__file__).parent

from tqdm import tqdm as _tqdm, tqdm

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_pmid(document: bioc.BioCDocument) -> Tuple[str, int]:
    infons = document.passages[0].infons
    if "article-id_pmid" in infons:
        pmid = document.passages[0].infons["article-id_pmid"]
        is_fulltext = int(document.id == infons.get("article-id_pmc", None))
    else:
        pmid = document.id
        is_fulltext = 0

    return pmid, is_fulltext

def _process_pubtator_files(files: List[Path], q: mp.Queue,
                            pickle_path: Path):
    for file in files:
        partial_index = {}
        with file.open() as f:
            collection = bioc.load(f)
            for i, document in enumerate(collection.documents):
                pmid, is_fulltext = get_pmid(document)
                partial_index[pmid] = (file.name, i, is_fulltext)
        q.put(partial_index)

        if pickle_path is not None:
            with open(pickle_path / file.with_suffix(".pkl").name, "wb") as f:
                pickle.dump(collection, f)


def build_index_and_document_pickles(pubtator_path, n_processes, pickle_path=None):
    index_path = cache_root / "pubtator.index"
    n_processes = n_processes or mp.cpu_count()

    if pickle_path:
        os.makedirs(pickle_path, exist_ok=True)

    index = {}
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    files = list(pubtator_path.glob("*bioc.xml"))
    processes = []
    for file_chunk in chunks(files, len(files) // n_processes-1):
        p = ctx.Process(target=_process_pubtator_files,
                        args=(file_chunk, q, pickle_path))
        p.start()
        processes.append(p)

    if pickle_path:
        desc = "Building PubTator Index and Pickling Documents"
    else:
        desc = "Building PubTator Index"
    pbar = tqdm(desc=desc, total=len(files))
    n_files_processed = 0
    while n_files_processed < len(files):
        partial_index = q.get()
        for k, v in partial_index.items():
            if k not in index or v[2]: # Either a new ID or the full text id that replaces the abstract id
                index[k] = v
        n_files_processed += 1
        pbar.update()

    for p in processes:
        p.join()

    with index_path.open("w") as f:
        for k, v in index.items():
            f.write(k + "\t" + v[0] + "\t" + str(v[1]) + "\t" + str(v[2]) + "\n")

    return index



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
        req = requests.get(url, stream=True, headers={"User-Agent": "Flair"})
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
    new_text = text[:offset] + replacement + text[offset+length:]
    new_offsets = offsets.copy()
    new_offsets[offsets >= offset+length] += delta_len

    return new_text, new_offsets

@dataclass
class Sentence:
    pmid: str
    text: str
    text_blinded: str

    def get_unmarked_text(self):
        return self.text.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")


class LocalPubtatorManager:

    def __init__(self, pubtator_path: Path, n_processes: Optional[int] = None):
        self.path = pubtator_path
        self.logger = logging.getLogger(LocalPubtatorManager.__name__)
        if n_processes is not None:
            assert n_processes > 1, "We need at least two processes for building the pubtator index"
            self.n_processes = n_processes
        else:
            self.n_processes = mp.cpu_count() + 1

        self.index = self.maybe_build_index()

    def maybe_build_index(self):
        index_path = cache_root / "pubtator.index"
        if not index_path.exists():
            index = build_index_and_document_pickles(pickle_path=None,
                                                     pubtator_path=self.path,
                                                     n_processes=self.n_processes)
        else:
            index = {}
            with index_path.open() as f:
                for line in tqdm(f, total=32280915,
                                 desc="Loading cached PubTator index"):
                    line = line.strip()
                    if line:
                        fields = line.split("\t")
                        index[fields[0]] = (fields[1], int(fields[2]), int(fields[3]))

        return index

    def _get_docs_from_files(self, files_to_open, pmids, q):
        docs = []
        for file in files_to_open:
            with open(self.path / file) as f:
                print("Opening " + str(file))
                collection = bioc.load(f)
                for document in collection.documents:
                    pmid = get_pmid(document)[0]
                    if pmid in pmids:
                        docs.append(document)
            print(f"Done: {file}")
            q.put(docs)

    def get_documents(self, pmids: List[str]):
        documents = []

        available_pmids = [i for i in pmids if i in self.index]
        missing_pmids = [i for i in pmids if i not in available_pmids]

        pmids = set(pmids)
        self.logger.info(f"Getting {len(available_pmids)} documents from local PubTator")
        for pmid in available_pmids:
            file = self.index[pmid][0]
            doc_idx = self.index[pmid][1]
            with open(self.path / file) as f:
                lines = f.readlines()
                document = bioc.loads(lines[0] + lines[doc_idx+1] + lines[-1]).documents[0]
                pmid = get_pmid(document)[0]
                assert pmid in pmids
                documents.append(document)

        return documents, missing_pmids


def get_homolog_mapping(expand_species, protein_universe) -> Dict[str, Set[str]]:
    """
    Get mapping from genes in `expand_species` that are not in `protein_universe` to all
    homologous gene_id in `protein_universe`
    """
    homolog_mapping = defaultdict(set)
    gene_id_to_homologene_id = {}
    homologene_id_to_cluster = defaultdict(set)

    with open(root / "data" / "homologene.data") as f:
        for line in tqdm(f, desc="Obtaining homologs", total=275237):
            fields = line.strip().split("\t")
            if not fields:
                continue
            homologene_id, taxon_id, gene_id = fields[:3]
            if gene_id in protein_universe:
                if taxon_id in expand_species or gene_id in protein_universe:
                    homologene_id_to_cluster[homologene_id].add(gene_id)
                if gene_id in protein_universe:
                    gene_id_to_homologene_id[gene_id] = homologene_id

        for gene_id in protein_universe:
            if gene_id in homologene_id_to_cluster:
                homologene_id = gene_id_to_homologene_id[gene_id]
                for mapped_gene_id in homologene_id_to_cluster[homologene_id]:
                    if mapped_gene_id not in protein_universe:
                        homolog_mapping[mapped_gene_id].add(gene_id)

    return homolog_mapping


class DataGetter:

    CHUNK_SIZE = 100
    def __init__(self, protein_universe: Set[str],
                 local_pubtator: Optional[Path] = None,
                 n_processes: Optional[int] = None,
                 cache_documents_for_pairs: Optional[List[Tuple[str, str]]] = None,
                 cache_size: Optional[int] = 10000,
                 api_fallback: Optional[bool] = False,
                 expand_species: Optional[List[str]] = None
                 ):
        self.protein_universe = protein_universe
        self.expand_species = expand_species or []
        if self.expand_species:
            self.homolog_mapping = get_homolog_mapping(self.expand_species,
                                                       self.protein_universe)
        else:
            self.homolog_mapping = {}
        self.gene2pmid = self.get_gene2pmid()
        self._document_cache = {}
        self.api_fallback = api_fallback

        self.cache_pmids = set()
        if cache_documents_for_pairs:
            for p1, p2 in cache_documents_for_pairs:
                shared_pmids = self.gene2pmid[p1] & self.gene2pmid[p2]
                self.cache_pmids.update(shared_pmids)

        self.sentence_splitter = SegtokSentenceSplitter()
        self.cache_size = cache_size
        if local_pubtator:
            self.local_pubtator = LocalPubtatorManager(local_pubtator,
                                                       n_processes=n_processes)
        else:
            self.local_pubtator = None

    def get_gene2pmid(self):
        gene2pmid = defaultdict(set)

        final_path = cache_root/"data"/"gene2pubtatorcentral"
        if not final_path.exists():
            print("Downloading gene2pubtatorcentral...")
            path = cached_path("https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/gene2pubtatorcentral.gz",
                               "data")
            unpack_file(path, final_path)

        with final_path.open() as f:
            for line in tqdm(f, total=53880670, desc="Loading gene2pubtatorcentral"):
                line = line.strip()
                fields = line.split("\t")
                gene_id = fields[2]
                pmid = fields[0]
                normalizers = fields[4]
                if "GNormPlus" not in normalizers:
                    continue

                if gene_id in self.protein_universe:
                    gene2pmid[gene_id].add(pmid)
                elif gene_id in self.homolog_mapping:
                    for mapped_gene_id in self.homolog_mapping[gene_id]: # mapped_gene_id is from self.protein_universe
                        gene2pmid[mapped_gene_id].add(pmid)

        return dict(gene2pmid)

    def maybe_map_to_pmcid(self, pmids):
        pmid_to_pmcid = {}

        service_root = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        ids = ",".join(pmids)
        response = requests.get(service_root, params={"ids": ids})
        for record in etree.fromstring(response.content).xpath("//record"):
            if "pmcid" in record.attrib:
                pmid_to_pmcid[record.attrib["pmid"]] = record.attrib["pmcid"]

        return pmid_to_pmcid

    def get_ids_from_annotation(self, annotation: bioc.BioCAnnotation) -> Set[str]:
        """
        Extract cuids from `annotation` and expand them with homologs from
        `self.expand_species`.
        """
        if "identifier" in annotation.infons:
            identifiers = annotation.infons["identifier"]
        elif "Identifier" in annotation.infons:
            identifiers = annotation.infons["Identifier"]
        else:
            identifiers = ""

        identifiers = set(identifiers.split(";"))

        expanded_identifiers = identifiers.copy()
        for cuid in identifiers:
            expanded_identifiers.update(self.homolog_mapping.get(cuid, {}))

        return expanded_identifiers

    def get_sentences_from_documents(self, protein1, protein2, documents):
        sentences = []
        for document in documents:
            for passage in document.passages:
                protein1_locations = []
                protein1_lengths = []
                protein2_locations = []
                protein2_lengths = []
                for annotation in passage.annotations:
                    if annotation.infons["type"] != "Gene":
                        continue

                    ids = self.get_ids_from_annotation(annotation)

                    if protein1 in ids:
                        for loc in annotation.locations:
                            protein1_locations.append(loc.offset - passage.offset)
                            protein1_lengths.append(loc.length)
                    if protein2 in ids:
                        for loc in annotation.locations:
                            protein2_locations.append(loc.offset - passage.offset)
                            protein2_lengths.append(loc.length)

                protein1_locations_arr = np.array(protein1_locations).reshape(-1, 1)
                protein2_locations_arr = np.array(protein2_locations).reshape(1, -1)
                dists = abs(protein1_locations_arr - protein2_locations_arr)
                for i, j in zip(*np.where(dists <= 300)):
                    loc_prot1 = protein1_locations_arr[i, 0]
                    loc_prot2 = protein2_locations_arr[0, j]
                    len_prot1 = protein1_lengths[i]
                    len_prot2 = protein2_lengths[j]
                    sentence = self.get_sentence(passage=passage,
                                                       offset_prot1=loc_prot1,
                                                       offset_prot2=loc_prot2,
                                                       len_prot1=len_prot1,
                                                       len_prot2=len_prot2,
                                                       pmid=document.id
                                                       )
                    if sentence:
                        sentences.append(sentence)

        return sentences

    def get_sentences(self, protein1, protein2):
        if protein1 not in self.gene2pmid or protein2 not in self.gene2pmid:
            return []
        pmids = self.gene2pmid[protein1] & self.gene2pmid[protein2]
        if not pmids:
            return []

        if self.local_pubtator is not None:
            documents = self.get_documents_from_local(pmids)
        else:
            documents = self.get_documents_from_api(pmids)

        return self.get_sentences_from_documents(protein1=protein1, protein2=protein2,
                                                 documents=documents)

    def maybe_get_from_cache(self, pmids: List[str]) -> Tuple[List[bioc.BioCDocument], List[str]]:
        cached_documents = [self._document_cache[i] for i in pmids if i in self._document_cache]
        uncached_pmids = [i for i in pmids if i not in self._document_cache]

        return cached_documents, uncached_pmids

    def cache_documents(self, documents: List[bioc.BioCDocument]) -> None:
        if self.cache_size is not None:
            n_docs_too_many = len(documents) + len(self._document_cache) - self.cache_size
            if n_docs_too_many > 0:
                for k in list(self._document_cache.keys())[:n_docs_too_many]:
                    del self._document_cache[k]

        for document in documents:
            pmid = get_pmid(document)
            if pmid not in self._document_cache and pmid in self.cache_pmids:
                self._document_cache[pmid] = document

    def get_documents_from_api(self, pmids):
        service_root = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml"
        documents = []

        if len(pmids) > self.CHUNK_SIZE:
            it = tqdm(list(chunks(pmids, self.CHUNK_SIZE)), desc="Receiving sentences")
        else:
            it = pmids
        for pmid_chunk in it:
            cached_documents, uncached_pmids = self.maybe_get_from_cache(pmid_chunk)
            pmid_to_pmcid = self.maybe_map_to_pmcid(pmid_chunk)
            pmids = [i for i in uncached_pmids if i not in pmid_to_pmcid]
            pmcids = [pmid_to_pmcid[i] for i in uncached_pmids if i in pmid_to_pmcid]

            result = requests.get(service_root, params={"pmids": ",".join(pmids),
                                               "pmcids": ",".join(pmcids),
                                                        "concepts": "gene"})
            collection = bioc.loads(result.content.decode())
            self.cache_documents(collection.documents)
            documents += collection.documents

        return documents

    def get_documents_from_local(self, pmids):
        assert isinstance(self.local_pubtator, LocalPubtatorManager)
        documents, missing_pmids = self.local_pubtator.get_documents(pmids)
        if self.api_fallback:
            documents += self.get_documents_from_api(missing_pmids)
        else:
            if len(missing_pmids):
                warnings.warn(f"{len(missing_pmids)}/{len(pmids)} documents could not be found in local PubTator."
                              f" Use api_fallback to retrieve missing documents from API.")

        return documents

    def get_sentence(self, passage, offset_prot1, offset_prot2, len_prot1, len_prot2, pmid):

        if offset_prot1 < offset_prot2:
            left_start = offset_prot1
            right_start = offset_prot2
        else:
            left_start = offset_prot2
            right_start = offset_prot1
        sents = self.sentence_splitter.split(passage.text.strip())
        snippet_start = None
        snippet_end = None
        for sent in sents:
            if sent.end_pos >= left_start >= sent.start_pos:
                snippet_start = sent.start_pos

            if sent.end_pos >= right_start >= sent.start_pos: # is sentence after right entity
                snippet_end = sent.end_pos

        if snippet_start is not None and snippet_end is None:
            snippet_end = sent.end_pos
        if snippet_start is None:
            return None


        offsets = []
        lengths = []
        id_to_offset_idx = defaultdict(list)
        offset_idx_p1 = None
        offset_idx_p2 = None
        for ann in passage.annotations:
            for loc in ann.locations:
                if snippet_end >= loc.offset - passage.offset >= snippet_start:
                    offset_idx = len(offsets)
                    if ann.infons["type"] == "Gene":
                        ids = self.get_ids_from_annotation(ann)
                        for cuid in ids:
                            id_to_offset_idx[cuid].append(len(offsets))

                        offsets.append(loc.offset - passage.offset)
                        lengths.append(loc.length)

                        if loc.offset-passage.offset == offset_prot1:
                            offset_idx_p1 = offset_idx
                        if loc.offset-passage.offset == offset_prot2:
                            offset_idx_p2 = offset_idx

        if offset_idx_p1 is None or offset_idx_p2 is None:
            # Weird encoding error
            return None

        offsets = np.array(offsets)
        text = passage.text[snippet_start:snippet_end]
        offsets -= snippet_start

        text_prot1 = passage.text[offset_prot1:offset_prot1 + len_prot1]
        text, offsets = replace_consistently(offset=offsets[offset_idx_p1], length=lengths[offset_idx_p1],
                                             replacement=f"<e1>{text_prot1}</e1>",
                                             text=text, offsets=offsets)
        offsets[offset_idx_p1] += 4

        text_prot2 = passage.text[offset_prot2:offset_prot2 + len_prot2]
        text, offsets = replace_consistently(offset=offsets[offset_idx_p2], length=lengths[offset_idx_p2],
                                             replacement=f"<e2>{text_prot2}</e2>",
                                             text=text, offsets=offsets)
        offsets[offset_idx_p2] += 4

        blinded_text = text
        for i, idcs in enumerate(id_to_offset_idx.values()):
            for idx in idcs:
                blinded_text, offsets = replace_consistently(offset=offsets[idx],
                                                     length=lengths[idx],
                                                     replacement=f"<protein{i}/>",
                                                     text=blinded_text, offsets=offsets)

        return Sentence(pmid=pmid, text=text, text_blinded=blinded_text)


def get_geneid_to_name():
    with open(root / "data" / "geneid_to_name.json") as f:
        return json.load(f)


def get_gene_mapping(from_db: str, to_db: str):
    final_mapping = defaultdict(set)
    uniprot_to_from_db = defaultdict(set)
    uniprot_to_to_db = defaultdict(set)
    uniprot_mapping_file = cached_path("https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz",
                                       cache_dir=cache_root)
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
