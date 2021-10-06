# intent_viz
Maruta and Kato. Intent-aware Visualization Recommendation for Tabular Data. WISE 2021.

## Setup
Clone our repository
```
git clone https://github.com/kasys-lab/intent-viz.git
cd intent-viz
```
### Download large files
Our model requires word embedding model and trained model
#### Download pre-trained word embedding vector
To get word embedding model, please download `glove.6B.zip`.

And put `glove.6B.100d.txt` file in unzipped `glove.6B.zip` file to `data/`.
```
wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.100d.txt ./data/
rm glove*
```
#### Download trained model
To get trained model, please access https://drive.google.com/drive/folders/1G2WTiNGCOzu2W4E3IFB6WT62RJXJkFeZ?usp=sharing and download `predict_vis_columns.pt`.

And put `predict_vis_columns.pt` to `saved_models/`

### Installation of Poetry (skip this step if Poetry has already been installed)
```
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```
### Installation of required Python packages
```
$ source ~/.bash_profile
$ poetry install
```
## Prediction
Predict a visualization type and visualized column percents
```
$ poetry run python demo.py <data path>
$ poetry run python demo.py "./data/input_data.json"
```

### Input data
Json format pair data of a visualization intent and tabular data.
```
{   "visualization_intent": "Population trends in Italy",
    "data": 
    [
        {
            "Year": 2015,
            "population": 100,
            "GDP": 2.3
        },{
            "Year": 2016,
            "population": 110,
            "GDP": 2.4
        },{
            "Year": 2017,
            "population": 130,
            "GDP": 2.9
        },{
            "Year": 2018,
            "population": 170,
            "GDP": 3.0
        },{
            "Year": 2019,
            "population": 200,
            "GDP": 3.1
        },{
            "Year": 2020,
            "population": 220,
            "GDP": 3.6
        }
    ]
}
```
### Output
Output visualuzation type and visualized column percent
```
predict visualization type : Line chart
predict visualized columns percent
header  :  percent
Year  :  1.0
population  :  0.9150764707515349
GDP  :  0.0
```


