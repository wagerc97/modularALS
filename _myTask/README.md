# To make a simple ALS procedure

### Get training values out of problem
- Compute problem solution on random X-input in interval 
- Store input/output values in csv file

### Build ML model on random set of problem values 
- Train machine learning model in specified number of randomly drawn value pairs from csv file 
- Visualize results 
- Assess and visualize model score and fit 
- Store ml model
- build Python class to wrap model generation

<---- here ----> 

### Optimize model in procedure
- __Prediction with ml model__ on new data
- Use __Optimization algo__ to generate pareto front on ML model predictions
- Visualize pareto front

### Evaluate pareto candidates
- define small set of promising pareto candidates
- Simuliere Mikromagentische Rechnung mit echten Problem-Daten (langsam)
- Visualize error between predicted values and "true" values

### Generalize and automate training 
- Apply procedure on bigger problems
- Incorporate MOO to automate training 

### EXTRAS TODO ...
- more ML models
- higher dimensional problem 
- use different optimization algos 
