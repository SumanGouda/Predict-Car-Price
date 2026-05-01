# 🚗 Second Hand Car Feature Prediction

A Machine Learning web application that predicts **price**, **power**, and **mileage** of a used car — served through a clean dark-themed web interface built with Flask.

---

## 📌 Overview

1. The model is trained on data from a famous second hand car selling platform **"CarDekho".**

2. Data was collected using web scraping with the **`beautifulsoup`** Python package.

3. To train the model with more city data, go to the **`WebScraping`** folder. In the scraper file, add the city name and the number of cars available on that city's page (shown at the top of the webpage), then run the function. It will create a JSON file — store it inside a folder with the same name as the city you scraped (create the folder if it doesn't exist).

4. In each feature prediction folder (`Price`, `Power`, `Mileage`), there are separate model notebooks. Add the new city name to the `city` variable and run the entire notebook — it will automatically retrain and update the model with the new city's data.

5. To run predictions, navigate to the **`website`** folder, run **`app.py`**, and open `127.0.0.1:5000` in your browser.

---

## 🧠 ML Models

| Model | Target | Key Features |
|---|---|---|
| Price | Market value (₹) | 27 features — brand, model, city (target encoded), fuel, drive type, engine specs |
| Power | Engine output (bhp) | 32 features — engine type (22 OHE categories), cylinders, engine density |
| Mileage | Fuel efficiency (kmpl) | 11 features — engine, kerb weight, power, fuel type, registration year |

---

## 🌐 Web Interface

- Built with **Flask** (Python backend)
- Three vertical sections — one form per prediction model
- Dark-themed UI written in **HTML & CSS**
- Each form maps directly to the model's exact feature vector
- Unknown brand/model inputs fall back to global mean (target encoding)

---

## 📁 Project Structure
```
CAR PRICE PREDICTION/
├── WebScraping/         # Scraper scripts (BeautifulSoup)
├── Price/
│   ├── model/           # price_model.pkl, brand_mapping.pkl,
│   │                    # model_mapping.pkl, city_mapping.pkl
│   └── model.ipynb
├── Power/
│   ├── model/           # power_model.pkl
│   └── model.ipynb
├── Mileage/
│   ├── model/           # mileage_model.pkl
│   └── model.ipynb
└── website/
├── app.py
├── static/
│   └── style.css
└── templates/
└── index.html
```

---

## ⚙️ Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install -r requirements.txt
```

Then run the web app:

```bash
cd website
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

---

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)