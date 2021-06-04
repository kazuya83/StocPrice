# StocPrice


Library installation
```python
streamlit==0.82.0
streamlit-folium==0.4.0
pandas==1.2.4
pandas-datareader==0.9.0
numpy==1.20.1
scikit-learn==0.24.1
```

Stock price forecasts are made using LSTM (Long Short Term Memory). As the calculation method, "mean square misuse method" and "stochastic gradient descent method" are used. The problem is that we have not performed technical analysis, and the data is limited in about 5 years, so extrapolation is likely to occur. We will continue to search for solutions to this problem, and in the future we will perform regression analysis instead of deep learning to increase the calculation processing speed and improve the accuracy by complicating the model.
