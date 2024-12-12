Le tree du project :

```
.
├── Dockerfile
├── data
│   ├── predictions
│   ├── test
│   │   ├── images
│   │   └── labels
│   ├── to_predict
│   └── train
│       ├── images
│       └── labels
├── models
└── src
    ├── data_preprocessing.py
    ├── predict.py
    └── train.py
````

Lancer les script depuis la racine (ex : python3 src/predict.py)

Les param de tous les scripts ont normalement des valeurs par défaut (pas encore testé pour data_preprocessing.py et train.py)