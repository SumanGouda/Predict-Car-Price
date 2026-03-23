# 🚗 Second Hand Car Feature Prediction 
A Machine Learning web application that predicts **price**, **power**, **mileage** of a used car.

## 📌 Overview
1. The model is trained over the data of a famous second hand car selling platform **"CarDekho".**

2. Collected data by webscraping tech. using the **'beautifulsoup'** python package.
3. To train the model with more city's data, just go to the **WebScraping** folder, and in the scraper file add the city name and the number of car's data available in the website (mentioned at the top of the webpage of that city) and then run the function it will create a json file, store that inside the folder(create if does not exist) same name as the city you scraped.  
4. Now in the different features prediction folders there are different models just enter the new city name in the city variable and run the entire notebook, it will automatically update the model, with the new cities data. 

5. To do the prediction, open the **webpage** folder run the **'app.py'** file and try out your prediction.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install decencies.

```bash
pip install -r requirements.txt
``` 

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)