# financial-instruments-forecasting
This repository contains a web application built with Dash, designed to support investment decisions through data visualization. 
The dashboard facilitates analysis of selected financial instruments using data fetched from Yahoo Finance using the `yfinance` library in Python. 
Key functionalities include:

- Selection of desired financial instruments.
- Prediction of future values using an RNN (Recurrent Neural Network) implemented with TensorFlow.
- Selection of date ranges to generate analysis based on user inputs.

### Important Information:

- **Prediction Limitations:** The dashboard allows predictions up to ten days into the future.
- **Date Range:** The available data ranges from January 1, 2008, to today's date.

### Usage Instructions:

To effectively use the dashboard:
- Use the **'Generate results'** button to generate visualizations based on your inputs. Note that this process may take some time.
- Use the **'Reset'** button to reset the dashboard to its initial state, clearing any visualizations.


## Using the Application

1. Clone the repository to your local machine:

 ```sh
    git clone https://github.com/stormborn333/financial-instruments-forecasting
    cd exchange-rate-forecasting
```

2. Build Docker Image:

```sh
    make docker-build
```

3. Run Docker Container:

```sh
    make docker-run
```

4. Open the following address in your web browser:

**http://localhost:8080**
