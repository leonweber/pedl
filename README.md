# PEDL

PEDL is a tool for predicting protein-protein assocations from the biomedical literature.
It searches more than 30 million abstracts of biomedical publications and over 4 million
full texts with the help of [PubTatorCentral](https://www.ncbi.nlm.nih.gov/research/pubtator/).
A state-of-the-art machine reading model then predicts which types of association between the proteins
are supported by the literature. Among others, PEDL can detect posttranslational modifications, 
transcription factor-target interactions, complex formations and controlled transports.

## Installation

```
pip install pedl
```

## Usage
PEDL supports two commands `pedl predict` and `pedl summarize`. The default workflow is to first
`predict` associations for one or more protein pairs of interest, which will store the results
for each pair in a separate file.
The contents of these files can then be aggregated into a single csv-file with `summarize`.

PEDL expects proteins to be identified via entrez gene ids. 
These can be looked up via standard webinterfaces like
[NCBI Gene](https://www.ncbi.nlm.nih.gov/gene).


### predict

* #### Interactions between single proteins
    ```bash
    pedl predict --p1 29126 --p2 54918 --out PEDL_predictions
    ```
  Results:
  ```bash
  $ ls PEDL_predictions/
  CD274-CMTM6.txt  CMTM6-CD274.txt
  
  $ head -n1 PEDL_predictions/CD274-CMTM6.txt
  in-complex-with	0.98	6978769	A PD-L1 antibody, H1A, was developed to destabilize PD-L1 by disrupting the <e1>PD-L1</e1> stabilizer <e2>CMTM6</e2>.	PEDL
  ```



* #### Pairwise interactions between multiple proteins
  ```bash
  pedl predict --p1 29126 --p2  54918 920  --out PEDL_predictions
  ```
  searches for interactions between 29126 and 54918, and for interactions between 29126 and 920


* #### Read protein lists from files
  ```bash
  pedl predict --p1 proteins.txt --p2  54918 920  --out PEDL_predictions
  ```
  searches for interactions between the proteins in `proteins.txt` and 54918, as well as interactions between proteins in `proteins.txt` and 920
  

* #### Search for multiple species at once
  If the provided gene ids are from human, mouse, rat or zebrafish, PEDL can automatically
  search for interactions in the other model species (currently human, mouse, rat and zebrafish)
  via homology classes defined by the [Alliance of Genome Resources](http://www.informatics.jax.org/homology.shtml):
  
    ```bash
    pedl predict --p1 29126 --p2 54918 --out PEDL_predictions --expand_species mouse zebrafish
    ```
    would also include interactions in mouse and zebrafish.


* #### Interactions from pathway databases
  It is also possible to query PathwayCommons for interactions. 
  This requires the python package `indra` to be installed, which can be achieved
  via `pip install indra`:
  ```bash
    pedl predict --p1 29126 --p2 54918 --out PEDL_predictions --dbs pid reactome kegg
  ```
  to query `pid` `reactome` and `kegg`. See `--help` for the full list of available
  databases.
  
  
* #### Large gene lists
  If you need to test for more than 100 interactions at once, you have to use a local copy 
  of PubTatorCentral, which can be downloaded [here](https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/PubTatorCentral_BioCXML/).
  Unpack the PubTatorCentral files and point PEDL towards the files:
    ```bash
    pedl predict --p1 large_protein_list1.txt --p2 large_protein_list2 --out PEDL_predictions --pubtator [PATH_TO_PUBTATOR]
    ```

  In this case, it is also strongly advised to use a CUDA-compatible GPU to speed up the machine reading:
    ```bash
    pedl predict --p1 large_protein_list1.txt --p2 large_protein_list2 --out PEDL_predictions
      --pubtator [PATH_TO_PUBTATOR]--device cuda
    ```
  
### summarize
```bash
pedl summarize PEDL_predictions [--out PATH_TO_SUMMARY_CSV]
```
* By default, PEDL will create the summary CSV in the directory containing `PEDL_predictions`




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


