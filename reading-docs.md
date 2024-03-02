# Development Environment

Uruchomienie WSL (koniecznie dla GPU i TensorFlow)

- uruchomić Windows PowerShell
- zainstalować WSL (windows subsystem for Linux)
- zainstalować Conda
- utworzyć środowisko wirtualne dla projektu

Instalacja zależności

- conda install --file requirements.txt

Uruchomić pycharm i podpiąć interpreter ze środowiska wirtualnego dla projektu

-  Settings - Project - Python Interpreter - Add Interpreter - On WSL





# Projekt BCNB

### The Early Breast Cancer Core-Needle Biopsy WSI (BCNB)

To jest projekt, z którego użyjemy dane do pierwszych eksperymentów - 1058 wsi (x200) - Masks + Clinical data

Tutaj są informacje o projekcie:
- https://bupt-ai-cz.github.io/BCNB/

Publikacja do tego zbioru
- https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2021.759007/full

Link do danych (sprawdzić czy działa dla Aleksandry i Kuby)
- https://drive.google.com/drive/folders/1PnZ1kND4bjnKaD6VxSdD7dZFB3upUke_



# Podstawy deep learning w histopatologii

- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4977982/



# Przegląd literatury

A Survey of Convolutional Neural Network in Breast Cancer (2023)
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7614504/



# Normalizacja 
Opis zagadnienia oraz przegląd metod.
- https://www.sciencedirect.com/science/article/pii/S1566253523003135

Biblioteka do normalizacji
- https://www.biorxiv.org/content/10.1101/2022.05.17.492245v1.full
- https://github.com/sebastianffx/stainlib/blob/main/stainlib_normalization.ipynb



# Czyszczenie tła/obrazu

Biblioteka do czyszczenia zanieczyszczeń na skanach WSI
- https://github.com/lucasrla/wsi-tile-cleanup?tab=readme-ov-file



# Tutoriale i przykłady

**Bardzo dobry tutorial ZEISS**

To jest jeden z filmów ale w serii jest więcej dobrych. Warto przeglądnąć.

- https://www.youtube.com/watch?v=85gWhlZiZ4E



Simple (Extremlty) UNET - identifying cell nuclei from histology images

https://github.com/VikramShenoy97/Histology-Image-Segmentation-using-UNet/tree/master



The same as above on U-Net++ - nuclei segmentation (Kaggle)

https://medium.com/mlearning-ai/unet-implementation-of-the-unet-architecture-on-tensorflow-for-segmentation-of-cell-nuclei-528b5b6e6ffd



U-Net explained

https://towardsdatascience.com/image-segmentation-unet-and-deep-supervision-loss-using-keras-model-f21a9856750a



# Snakemake

Repozytorium

- https://snakemake.github.io/

Opis snakemake

- https://f1000research.com/articles/10-33/v2



# Przykłady

Przykład klasyfikacji:
- https://www.kaggle.com/code/thesnak/breast-cancer-classification-96-89

Przykładowa publikacja: Boosting Breast Cancer Detection Using Convolutional Neural Network

- https://www.hindawi.com/journals/jhe/2021/5528622/



# Zbiory danych

Zebrane repozytoria danych histopatologicznych 
- https://github.com/maduc7/Histopathology-Datasets

Breast Cancer Histopathological Database (BreakHis) ## Do sprawdzenia (WAŻNE)
- https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

