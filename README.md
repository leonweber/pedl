# PEDL: Protein-Drug and Protein-Protein Association Extraction

PEDL is a powerful framework designed to extract protein-protein and drug-protein associations from biomedical literature. By searching over 30 million abstracts of biomedical publications and more than 4 million full texts using [PubTatorCentral](https://www.ncbi.nlm.nih.gov/research/pubtator/), PEDL leverages state-of-the-art machine reading models to predict association types between proteins, supported by literature evidence.

PEDL is capable of detecting the following association types:

- **Protein-Protein**: controls-phosphorylation-of, controls-state-change-of, controls-transport-of, controls-expression-of, in-complex-with, interacts-with, and catalysis-precedes 
- **Drug-Protein**: antagonist, agonist, agonist-inihibitor, direct-regulator, activator, inhibitor, indirect-downregulator, indirect-upregulator, part-of, product-of, substrate, and substrate\_product-of

For example usage with expected results, see [this notebook](https://github.com/leonweber/pedl/blob/master/example_usage.ipynb).

## Installation

Install PEDL via pip:

```
pip install pedl
```

## Usage

PEDL provides two main commands: `pedl-extract` and `pedl-summarize`. The default workflow consists of using `pedl-extract` to obtain associations for one or more protein or drug pairs of interest, storing the results for each pair in separate files. These files can then be aggregated into a single spreadsheet using `pedl-summarize`.

Proteins should be identified using HGNC symbols (for human genes) or Entrez gene IDs, searchable via [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene). Drugs should be identified by their MeSH ID, searchable via [MeSH NLM](https://meshb.nlm.nih.gov/).

### Building the PubTator Index (Optional)

For searches involving more than 100 pairs with PEDL, it is recommended to use an ElasticSearch index. If you do not build the index, PEDL will automatically download the required data using the PubTatorCentral API and run the search on the fly. However, this will be slower and won't work for more than 100 pairs.

PEDL supports storing a preprocessed version of all PTC texts in an ElasticSearch index. Install and run [Elasticsearch 7.17](https://www.elastic.co/guide/en/elasticsearch/reference/master/install-elasticsearch.html) and download all [PubTatorCentral files](https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/PubTatorCentral_BioCXML/) on your server. Note that this requires around 800GB of disk space. You can either build a docker container or follow the instructions on the website to run a default Elasticsearch host. Rebuilding the PubTator index can take considerable time (around a day on 70 threads).

#### pedl-rebuild_pubtator_index

```bash
pedl-rebuild_pubtator_index elastic.server=<server_address> elastic.password=<password> elastic.ca_certs=<path_to_certs> pubtator_file=<path_to_pubtator_file> n_processes=<number of processes>
```

Example call:

```bash
pedl-rebuild_pubtator_index elastic.server=https://localhost:9200 elastic.password=Naifojair141+123cIlPo7fu elastic.ca_certs=/home/pedl/http_ca.crt pubtator_file=/home/pedl/output/BioCXML n_processes=10
```

### PEDL Extract

#### Single Protein or Drug Interactions

To extract interactions, you can use either the Entrez gene ID or name for proteins, and the MeSH ID or name for drugs. If using Entrez and/or MeSH IDs, set `use_ids=True`. Please note that mixing names and IDs in the same query is not possible.

Set `type=protein_protein` for protein-protein interactions or `type=drug_protein` for drug-protein interactions.


- Example (Protein-Protein Interaction):

  ```bash
  pedl-extract e1=CMTM6 e2=CD274 type=protein_protein out=PEDL_extractions
  ```

- Example (Drug-Protein Interaction):

  ```bash
  pedl-extract e1=MeSH:D063325 e2=1813 type=drug_protein out=PEDL_extractions use_ids=True

  ```

#### Pairwise Interactions Between Multiple Proteins or Drugs

- Protein-Protein Interaction:

  ```bash
  pedl-extract e1=[CMTM6,PDCD1LG2] e2=CD274 type=protein_protein out=PEDL_extractions
  ```

- Drug-Protein Interaction:

  ```bash
  pedl-extract e1=[MeSH:D000661,D008694] e2=4129 type=drug_protein out=PEDL_extractions use_ids=true
  ```

#### Pairwise Interactions Between All Proteins or Drugs

- Protein-Protein Interaction:


  ```bash
  pedl-extract e1=all e2=CD274 type=protein_protein out=PEDL_extractions 
  ```

- Drug-Protein Interaction:

  ```bash
  pedl-extract e1=all e2=4129 type=drug_protein out=PEDL_extractions use_ids=true
  ```
- With a local PubTator index:
    
  ```bash
  pedl-extract e1=all e2=4129 type=drug_protein out=PEDL_extractions use_ids=true elastic.server=<server_address> elastic.password=<password> elastic.ca_certs=<path_to_certs> 
  ```

#### Read Protein or Drug Lists from Files

- Protein-Protein Interaction:

  ```bash
  pedl-extract e1=proteins.txt e2=[54918,920] type=protein_protein out=PEDL_extractions use_ids=true
  ```

- Drug-Protein Interaction:

  ```bash
  pedl-extract e1=drugs.txt e2=4129 type=drug_protein out=PEDL_extractions use_ids=true
  ```

### PEDL Summarize

Create a summary spreadsheet for all results in a directory:

```bash
pedl-summarize input=PEDL_predictions output=summary
```

To filter results by specific MeSH terms, use the `mesh_terms` parameter. Escape special characters with a backslash `\`:

```bash
pedl-summarize input=PEDL_extractions output=summary mesh_terms=["Apoptosis","Lymphoma\, B-Cell"]
```

To only use high-confidence extractions, use the `threshold` parameter:

```bash
pedl-summarize input=PEDL_extractions output=summary threshold=0.9
```
