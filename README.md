# intent_viz
Maruta and Kato. Intent-aware Visualization Recommendation for Tabular Data. WISE 2021.

## prediction visualizatoin type and visualized columns
```
python demo.py
```

## input data
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
