# INSTALACJA WSL
Z tego linka robimy do zakończenia instalacji WSL i minicondy
https://medium.com/@mass.thanapol/tensorflow-with-gpu-on-linux-or-wsl2-10b02fd19924
## INSTALACJA CUDA I TENSORFLOW
Z poniższego linka praktycznie wklejamy wszystko po kolei zaczynając od sekcji Installing TensorFlow 2.13 and CUDA libraries manually inside Conda environment, należy pamietać o zmianie cudzysłowów na " lub '.
https://medium.com/@dev-charodeyka/tensorflow-conda-nvidia-gpu-on-ubuntu-22-04-3-lts-ad61c1d9ee32

# INSTALACJA PYCHARM
Po zainstalowaniu PyCharm Pro wybieramy WSL i odpowiednią dystrybujce Linuxa, następnie w ustawieniach szukamy Project: {Nazwa projektu} i wybieramy interpreter z venv condy gdzie mamy zainstalowanego Tensorflow'a.
![obraz](https://github.com/michalmadera/path-works/assets/85565440/93e93b50-d772-470a-adff-2a59b8392bf1)
Następnie należy dodać ścieżkę do CUDA aby PyCharm użył GPU. W opcjach Run/Debug Configuration dodajemy nowy wpis do Environment variables i wpisujemy tam wynik polecenia "echo $LD_LIBRARY_PATH" (to w wsl na odpowiednim venv condy wpisujemy).
![obraz](https://github.com/michalmadera/path-works/assets/85565440/ffe06e01-796e-4b9a-b328-6614a3fbeece)
Po sprawdzeniu powinno działać (przynajmniej u mnie działa).
![obraz](https://github.com/michalmadera/path-works/assets/85565440/772176f4-28cc-452c-8354-119068a5071f)
