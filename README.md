# Signals-Transforms
The objective of this project is to predict the tide levels in a given location using the old data. We used data over 1 year which includes eater level measurements every
10 minutes. This data is provided in the excel file. 
The main code reads this data from the excel file and does the calculations with it. We use Fourier transform to construct the prediction graph. Once the graph is built, 
it is optimized by fitting it to the actual data. In the end, the program gives a graph of the prediction for the coming dates. Also, it prints the time intervals when the
prediction is above a threshold value.
