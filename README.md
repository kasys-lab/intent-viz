# intent_viz
Maruta and Kato. Intent-aware Visualization Recommendation for Tabular Data. WISE 2021.

## Setup
### Install git lfs and clone
Git lfs is used because of large size file
```
brew install git-lfs
git lfs clone https://github.com/kasys-lab/intent-viz.git
```

### Installation of Poetry (skip this step if Poetry has already been installed)
```
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```
### Installation of required Python packages
```
$ source ~/.bash_profile
$ poetry install
```
## Prediction visualizatoin type and visualized columns for input data file
```
$ poerty run python demo.py <data path>
$ poerty run python demo.py "./data/input_data.json"
```

### Input data
json format pair data of a visualization intent and tabular data.
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
output visualuzation type and visualized column percent
```
predict visualization type : Line chart
predict visualized columns percent
header  :  percent
Year  :  1.0
population  :  0.9150764707515349
GDP  :  0.0
```


