# NeuRec
Using Neural Collaborative Filtering to recommend OTT content to users. A solution to Sony inc RISE hackathon

![Neural Collaborative Filtering](https://miro.medium.com/max/1400/1*aP-Mx266ExwoWZPSdHtYpA.png)

A Link to the model checkpoint can be found on [Google Drive](https://drive.google.com/drive/folders/1--3T3Mn0L0UCAH0thAkINIL2I-hdKNNA?usp=sharing)

# Directory Structure
```commandline
.
├── Data.zip
├── LICENSE
├── README.md
├── Sony_NeuRec.ipynb
├── requirements.txt
├── src
│   ├── configs.py
│   ├── inference.py
│   ├── model.py
│   └── training.py
└── utils
    └── json2csv.py

```

## Recreating Results
To recreate the results in the repository. Unzip `Data.zip` and run `src/training.py`
