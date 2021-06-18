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


### Prediction

* Interactions between single proteins:
    ```bash
    pedl --p1 5052 --p2 7099 --out PEDL_predictions
    ```
  Results:
  ```bash
  $ ls PEDL_predictions/
  PRDX1-TLR4.txt  TLR4-PRDX1.txt
  
  $ head -n1 PEDL_predictions/PRDX1-TLR4.txt
  in-complex-with 0.93    4872721 <e1>PRDX1</e1> functioned as a ligand for <e2>Toll-like receptor 4</e2> to enhance HIF-1alpha expression and HIF-1 binding to the promoter of the VEGF gene in endothelial cells, thereby potentiating VEGF expression.    PEDL
  ```



* Pairwise interactions between multiple proteins:
  ```bash
  pedl --p1 5052 --p2  7099 222344  --out PEDL_predictions
  ```
  searches for interactions between 5052 and 7099, and for interactions between 5052 and 222344


* Read protein lists from files:
  ```bash
  pedl --p1 proteins.txt --p2  7099 222344  --out PEDL_predictions
  ```
  searches for interactions between the proteins in `proteins.txt` and 7099, as well as interactions between proteins in `proteins.txt` and 222344
  

* Search for interactions in multiple species via the orthologs of the provided proteins:
    ```bash
    pedl --p1 5052 --p2 7099 --out PEDL_predictions --expand_species 10090 10116
    ```
    would also include interactions in Mouse and Rat


### Prediction for large gene lists  
If you need to test for more than 100 interactions at once, you have to use a local copy 
of PubTatorCentral, which can be downloaded [here](https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/PubTatorCentral_BioCXML/).
Unpack the PubTatorCentral files and point PEDL towards the file:
  ```bash
  pedl --p1 large_protein_list1.txt --p2 large_protein_list2 --out PEDL_predictions --pubtator [PATH_TO_PUBTATOR]
  ```

In this case, it is also strongly advised to use a CUDA-compatible GPU to speed up the machine reading:
  ```bash
  pedl --p1 large_protein_list1.txt --p2 large_protein_list2 --out PEDL_predictions
    --pubtator [PATH_TO_PUBTATOR]--device cuda
  ```




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


