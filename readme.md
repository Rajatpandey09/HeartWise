# HeartWise

A minimalist web app that estimates the probability of heart disease from clinical readings using a trained K-Nearest Neighbours (KNN) model. It shows a live ECG trace that reacts to the inputs, gives personalised tips, and lets people download their report as a PDF.

> **Educational project. Not a medical device and not a diagnosis.** Always talk to a qualified doctor about your heart health.

---

## Features

- **Live ECG hero**: heartbeat speed follows max heart rate; higher risk makes the rhythm uneven, depresses the ST segment and flips the T wave.
- **Instant risk estimate**: a large percentage with a low, moderate or high band.
- **Personalised tips**: up to four suggestions (blood pressure, cholesterol, chest pain, fitness, check-ups) based on the entered readings.
- **Download report**: saves a PDF with the risk, readings and tips. Falls back to a `.txt` file if the PDF library cannot load.
- **Light and dark mode**, remembered between visits.
- **Mobile friendly**: responsive layout that works on phones, tablets and desktops.
- **Animations**: intro sequence, heartbeat pulse and scroll reveals, all disabled when the visitor has "reduce motion" turned on.
- **FastAPI backend** with input validation and a "what-if" list showing which improvements would lower the score.

---

## Project structure

```
HeartWise/
├── index.html              # Frontend: HTML, CSS and JS in one file
├── main.py                 # FastAPI backend
├── HeartdiseaseFinal.ipynb # Model training notebook
├── knn_heart_model.pkl     # Trained KNN model
├── heart_scaler.pkl        # Fitted scaler
├── heart_columns.pkl       # Column order used in training
├── requirements.txt        # Python dependencies
├── .python-version         # Python version for deployment
└── readme.md
```

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/Rajatpandey09/HeartWise.git
cd HeartWise
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS or Linux use `source venv/bin/activate` instead.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Use the same `scikit-learn` version that trained the model, otherwise loading the `.pkl` files may fail or show warnings.

### 4. Run the app

```bash
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. The backend serves `index.html` at `/`.

### 5. Try it on your phone (same Wi-Fi)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then visit `http://<your-computer-IP>:8000` on the phone. You may need to allow port 8000 through Windows Firewall.

---

## API

### `POST /predict`

| Field | Type | Allowed values |
|---|---|---|
| `Age` | int | 18 to 100 |
| `Sex` | string | `M`, `F` |
| `ChestPainType` | string | `ATA`, `NAP`, `TA`, `ASY` |
| `RestingBP` | int | 80 to 200 |
| `Cholesterol` | int | 100 to 600 |
| `FastingBS` | int | `0`, `1` |
| `RestingECG` | string | `Normal`, `ST`, `LVH` |
| `MaxHR` | int | 60 to 220 |
| `ExerciseAngina` | string | `Y`, `N` |
| `Oldpeak` | float | 0 to 6 |
| `ST_Slope` | string | `Up`, `Flat`, `Down` |

Unknown fields are rejected with a `422` error.

Example response:

```json
{
  "probability": 0.6,
  "prediction": 1,
  "risk_band": "high",
  "levers": [
    { "label": "All of these together", "drop": 0.4, "actionable": true },
    { "label": "Bring cholesterol down to 200", "drop": 0.2, "actionable": true }
  ]
}
```

- `levers` lists what-if changes that would lower the score. Each one only appears when it is a real improvement for that patient.
- `actionable` is `false` for test findings (such as ST slope) that a person cannot change directly.

### Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Checks that the service and model loaded |
| `GET /` | Serves `index.html` |
| `GET /docs` | Interactive API docs (Swagger UI) |

---

## Connecting the frontend to the model

`index.html` has a constant near the top of its `<script>` block:

```js
const API_URL = null;
```

While it is `null`, the page uses a built-in placeholder formula instead of your KNN model. To use the real model, point it at the backend:

```js
const API_URL = "/predict";
```

The request must contain all eleven fields from the table above (for example `Male` becomes `M`, `Atypical` becomes `ATA`, `Yes` becomes `Y`), and the page reads `probability` from the response.

---

## Customisation

| What | Where |
|---|---|
| Colours and dark mode | CSS variables at the top of the `<style>` block |
| Fonts | The Google Fonts `<link>` in `<head>` and the `font-family` rules |
| Form inputs | The `FIELDS` array in the script |
| Tip wording and thresholds | The `buildTips` function |
| Report layout | The `downloadReport` function (uses [jsPDF](https://github.com/parallax/jsPDF)) |
| Risk bands | `pct<30` and `pct<60` in `index.html`, and `band()` in `main.py` |

---

## Notes and limitations

- **KNN gives coarse probabilities.** With `k=5` the model can only output 0, 20, 40, 60, 80 or 100%, so the number moves in steps. A larger `k` or `weights="distance"` gives smoother values.
- **Privacy**: the PDF is built in the visitor's browser. When the API is connected, entered readings are sent to your server, so use HTTPS in production.
- **Internet needed for extras**: Google Fonts and the PDF library load from a CDN. Offline, the page falls back to system fonts and a text report.
- The tips use general public-health thresholds. Have a medical professional review the wording before sharing the app widely.

---

## Deploying

Any host that runs Python web apps works, for example Render, Railway or Fly.io. Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Make sure the three `.pkl` files and `index.html` are committed alongside `main.py`, and that `.gitignore` does not exclude them.

---

## Disclaimer

HeartWise is built for learning and demonstration. Its estimates come from a machine learning model trained on a limited dataset and must not be used to make medical decisions.