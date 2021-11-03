# Daniele Domenichelli
## Artificial Intelligence in Industry - Project:

## *Mid-Long temporal relations prediction in business time-series*

### [**Report**](report.pdf)

### **Abstract**

A very common and growing problem in many financial applications, is the capacity to forecast
business time series to better predict the behaviour that a seller should employ with their prices to maximize the selling income.

The target of this project is to explore and compare different approaches to analyse and forecast time related distributions in the business time-series prediction context, by extracting and constraining temporal relations intrinsic in the selling behaviour (seasonalities). 

The study was conducted to produce a simple baseline model, for this reason the models were limited to use just as little inputs as possible. In particular, only the amount of sold units and the relative month of the year were used for the training phase and to produce the output.

### **Program**

Available programs are:
- `data.extraction.py` - This program will preprocess the full Dataset selecting just a minimal part of it. For huge dataset it will take some time to process
- `model.lattice.py` - This program train a Lattice Model from an input dataset and outputs a metric evaluation of the trained model
- `model.prob.py` - This program train a MLP (probsbilstic) Model from an input dataset and outputs a metric evaluation of the trained model

Launch each python program with `--help` or `-h` flag to discover more informations and input falgs about it.
