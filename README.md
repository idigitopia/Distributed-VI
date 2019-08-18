# Distributed-VI
Distributed Implementation of Value Iteration, built on top of Ray Framework. 


### Prerequisites
```
matplotlib==3.0.3
numpy==1.16.3
ray==0.7.2
plotly==4.1.0
```

### Installing
```
pip install -r requirements.txt
```


## Running  Experiments 
The repo currently supports two VI-Engines.
- Simple single worker Value Iteration 
- Distributed Value Iteration with varying workers 


The provided Frozen_Lake Environment has been used for testing/Benchmarking purposes. (Note that the state space is equal to the map_size to the power of 2)

Sample run example for solving frozen_lake environment of map_size 100X100 using distributed VI engine with consuming 8 workesrs. (experiment repeated 5 times for avg runtime and experiment folder to be padded with test1)
```
python run.py -e frozen_lake -exp_id test1 -vi distributed -w 8 -m 100 -r 5
```

### Benchmarking Code
```
python run.py -e frozen_lake -exp_id test1 -vi simple -m 10,32,72,100,225,320,500,708,868,1000 -r 10
python run.py -e frozen_lake -exp_id test1 -vi distributed -w 2 -m 10,32,72,100,225,320,500,708,868,1000 -r 10
python run.py -e frozen_lake -exp_id test1 -vi distributed -w 4 -m 10,32,72,100,225,320,500,708,868,1000 -r 10
python run.py -e frozen_lake -exp_id test1 -vi distributed -w 6 -m 10,32,72,100,225,320,500,708,868,1000 -r 10
python run.py -e frozen_lake -exp_id test1 -vi distributed -w 8 -m 10,32,72,100,225,320,500,708,868,1000 -r 10
python run.py -e frozen_lake -exp_id test1 -vi distributed -w 12 -m 10,32,72,100,225,320,500,708,868,1000 -r 10
python run.py -e frozen_lake -exp_id test1 -vi distributed -w 16 -m 10,32,72,100,225,320,500,708,868,1000 -r 10
python run.py -e frozen_lake -exp_id test1 -vi distributed -w 20 -m 10,32,72,100,225,320,500,708,868,1000 -r 10
python run.py -e frozen_lake -exp_id test1 -vi distributed -w 24 -m 10,32,72,100,225,320,500,708,868,1000 -r 10
```


## Acknowledgements
- Mostly inspired by Synchronous Value Iteration Assignment from CS533 [INTELLIGENT AGENTS & DECISION MAKING(CS_533 - Spring 2018)] 