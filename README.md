# PEDL

PEDL is a framework for extracting protein-protein and drug-protein assocations from biomedical literature.
It searches more than 30 million abstracts of biomedical publications and over 4 million
full texts with the help of [PubTatorCentral](https://www.ncbi.nlm.nih.gov/research/pubtator/).
A state-of-the-art machine reading model then predicts which types of association between the proteins
are supported by the literature.

PEDL can detect the following types of associations:
- **Protein-Protein**: controls-phosphorylation-of, controls-state-change-of, controls-transport-of, controls-expression-of, in-complex-with, interacts-with, and catalysis-precedes 
- **Drug-Protein**: antagonist, agonist, agonist-inihibitor, direct-regulator, activator, inhibitor, indirect-downregulator, indirect-upregulator, part-of, product-of, substrate, and substrate\_product-of


## Installation

```
pip install pedl
```

## Usage
PEDL supports two main commands `pedl-extract` and `pedl-summarize`. The default workflow is to 
either first build a pubtator index with `pedl-rebuild_pubtator_index` and then `pedl-extract` associations for one or more protein or drug pairs of interest, which will store the results
for each pair in a separate file or simply run `pedl-extract` without building the pubtator index.
The contents of these files can then be aggregated into a single spreadsheet with `pedl-summarize`.

PEDL requires proteins to be identified either via HGNC symbols (for human genes)
or Entrez gene IDs, searchable via [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene). Drugs should be identified by their MESH id, searchable via [MESH NLM](https://meshb.nlm.nih.gov/).

### Building the PubTator Index
It is recommended to use an ElasticSearch index when searching for more than 100 pairs with PEDL. **This step is optional. If you do not want to build the index, PEDL will automatically download the required data from PubTatorCentral and run the search on the fly. However, this will be much slower and will not work if you search for more than 100 pairs.**

PEDL supports storing a preprocessed version of all PTC texts in an ElasticSearch index.
Please install and run [Elsaticsearch 7.17](https://www.elastic.co/guide/en/elasticsearch/reference/master/install-elasticsearch.html )
and download all [PubTatorCentral files](https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/PubTatorCentral_BioCXML/) on your server. Note, that this requires around 800GB of disk space.
You can either build a docker container or follow the instructions on the website to
run a default elasticsearch host.
You can then run rebuild_pubtator_index. Building a pubtator index can take some time (around a day on 70 threads).

* #### pedl-rebuild_pubtator_index
    ```bash
    pedl-rebuild_pubtator_index pubtator=<PATH_TO_PUBTATOR>
    ```

### pedl-extract

* #### Interactions between single proteins or drugs

  For proteins, you can either type the gene_id (e.g. 54918) or the name (e.g. CD274) of the protein. For drugs, you can 
either type the MESH-ID (e.g. MESH:D000661 or D000661) or the name (e.g. Amphetamine) of the drug. The type
can either be set to "protein_protein" if you want to run protein-protein interactions
or to "drug_protein" if you want to extract drug-protein interactions.

  * ```bash
      pedl-extract e1=CD274 e2=CMTM6 type=protein_protein out=PEDL_extractions
      ```

    Results:
    ```bash
    $ ls PEDL_extractions/
    CD274-CMTM6.txt  CMTM6-CD274.txt
  
    $ head -n1 PEDL_extractions/CD274-CMTM6.txt
    in-complex-with	0.98	6978769	A PD-L1 antibody, H1A, was developed to destabilize PD-L1 by disrupting the <e1>PD-L1</e1> stabilizer <e2>CMTM6</e2>.	PEDL
    ```

  or

  * ```bash
      pedl-extract e1=Amphetamine e2=MAOB type=drug_protein out=PEDL_extractions
      ```



* #### Pairwise interactions between multiple proteins or drugs
  
  * Protein-Protein
    ```bash
    pedl-extract e1=[CMTM6,PDCD1LG2] e2=CD274 type=protein_protein out=PEDL_extractions
    ```
    searches for interactions between CD274 and CMTM6, and for interactions between CD274 and PDCD1LG2

  * Drug-Protein

    ```bash
    pedl-extract e1=[MESH:D000661,D008694] e2=4129 type=drug_protein out=PEDL_extractions
    ```

* #### Pairwise interactions between all proteins or drugs
  
  * Protein-Protein
    ```bash
    pedl-extract e1=all e2=CD274 type=protein_protein out=PEDL_extractions
    ```
    searches for interactions between CD274 and all proteins in the current hgnc data base.

  * Drug-Protein

    ```bash
    pedl-extract e1=all e2=4129 type=drug_protein out=PEDL_extractions
    ```
    searches for interactions between 4129 and all drugs in the current MESH data base.

* #### Read protein or drug lists from files
  
  * Protein-Protein
    ```bash
    pedl-extract e1=proteins.txt e2=[54918,920] type=protein_protein out=PEDL_extractions
    ```
    searches for interactions between the proteins in `proteins.txt` and 54918, as well as interactions between proteins in `proteins.txt` and 920

  * Drug-Protein

    ```bash
    pedl-extract e1=drug.txt e2=4129 type=drug_protein out=PEDL_extractions
    ```
    searches for interactions between the drugs in `drug.txt` and 4129


* #### Allow multiple sentences
  By default, PEDL will only search for interactions described in a single sentence.
  If you want PEDL to read text snippets that span multiple sentences, use
  `--multi_sentence`. Note, that this may slow down reading by a lot if you are not using a GPU.
  ```bash
    pedl-extract e1=CD274 e2=CMTM6 out=PEDL_extractions multi_sentence=True
  ```
  
    ```bash
    pedl-extract e1=D008694 e2=4129 out=PEDL_extractions multi_sentence=True
  ```
  
# 
* #### Search for multiple species at once
  If the provided gene ids are from human, mouse, rat or zebrafish, PEDL can automatically
  search for interactions in the other model species (currently human, mouse, rat and zebrafish)
  via homology classes defined by the [Alliance of Genome Resources](http://www.informatics.jax.org/homology.shtml):
  
    ```bash
    pedl-extract p1=29126 p2=54918 out=PEDL_extractions expand_species=[mouse,zebrafish]
    ```
    would also include interactions in mouse and zebrafish.
  
  
* #### Large gene lists
  If you need to test for more than 100 interactions at once, you have to use a local copy 
  of PubTatorCentral, which can be downloaded [here](https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/PubTatorCentral_BioCXML/).
  Unpack the PubTatorCentral files and point PEDL towards the files:
    ```bash
    pedl-extract e1=large_protein_list1.txt e2=large_protein_list2 out=PEDL_extractions pubtator=<PATH_TO_PUBTATOR>
    ```

  In this case, it is also strongly advised to use a CUDA-compatible GPU to speed up the machine reading:
    ```bash
    pedl-extract e1=large_protein_list1.txt e2=large_protein_list2 out=PEDL_extractions
      pubtator=<PATH_TO_PUBTATOR> device=cuda
    ```
  
### summarize
Use `summarize` to create a summary file describing all results in a directory.
By default, PEDL will create the summary CSV next to the results directory.
```bash
pedl-summarize PEDL_extractions
```
Results:
  ```bash
  $ head path_to_files=PEDL_extractions.tsv
  p1      association type        p2      score (sum)     score (max)
  CMTM6   controls-state-change-of        CD274   4.17    0.90
  CMTM6   in-complex-with CD274   2.48    0.97
  CD274   in-complex-with CMTM6   2.40    0.98
  ````

Results can also be aggregate ignoring the association type and the direction of the association: 
```bash
  $ pedl-summarize PEDL_extractions no_association_type=True
  
  $ cat PEDL_extractions.tsv
  p1      association type        p2      score (sum)     score (max)
  CD274   association     CMTM6   11.52   1.00
  ````




## References
Code and instructions to reproduce the results of our [paper](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i490/5870497), can be found [here](https://github.com/leonweber/pedl_ismb20).

If you use PEDL in your work, please cite us
```
@article{weber2020pedl,
  title={PEDL: extracting protein--protein associations using deep language models and distant supervision},
  author={Weber, Leon and Thobe, Kirsten and Migueles Lozano, Oscar Arturo and Wolf, Jana and Leser, Ulf},
  journal={Bioinformatics},
  volume={36},
  number={Supplement\_1},
  pages={i490--i498},
  year={2020},
  publisher={Oxford University Press}
}
```


