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
  <br> <---- here ----> <br>
- Overfitting?


### Optimize model in procedure
- __Prediction with ml model__ on new data
- Use __Optimization algo__ to generate (pareto front) best candidate on ML model  
  - newLabel = opitmizer(algo="nga2", data=(myModel.predict(selber werte bereich ))
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
- apply finished ALS to different test/standard problems 
  - https://en.wikipedia.org/wiki/Test_functions_for_optimization 
- NN einbauen -> problem von Harry verwendent CNN mit Keras (später wenn Zeit)
- CNN einbauen (später wenn Zeit)


## Forschungsfragen ?
- Benefit von ALS gegenbüber herkömmllicher optimierung 
- Anzahl von teuren Funktionsaufrufen/-auswertungen bei problem optimierung VS. reine optimierungs sw
- extra: wenn es rein passt... Machen die ursprünglichen Trainingsdaten einen unterschied im ALS?
  - haben wir ein bias?
  - kann sich das bias weg lernen?
- 
