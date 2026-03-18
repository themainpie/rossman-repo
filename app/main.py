from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from pathlib import Path
from src.load_data import load_model, load_store
from src.preprocess import prepare_data
import pandas as pd
import numpy as np
import yaml
import io

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

config_path = PROJECT_ROOT / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

app = FastAPI()
model = load_model()
meta_data = load_store()


@app.get("/", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
        <head>
            <title>File Upload for Prediction</title>
        </head>
        <body>
            <h2>Upload your CSV file for prediction</h2>
            <form action="/handle_upload" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".csv" required>
                <button type="submit" name="action" value="preview">Preview Predictions</button>
                <button type="submit" name="action" value="download">Download CSV</button>
            </form>
        </body>
    </html>
    """

@app.post("/handle_upload")
async def handle_upload(file: UploadFile = File(...), action: str = Form(...)):
    df = pd.read_csv(file.file)
    preped_data = prepare_data(df, meta_data, config)
    pred_sqrt = model.predict(preped_data)**2
    preds = np.round(pred_sqrt)

    if action == "preview":
        preds_str = [f"{p}" for p in preds]
        return {"prediction_preview": preds_str[:20], "total_predictions": len(preds_str)}
    
    elif action == "download":
        result_df = pd.DataFrame({"prediction": preds})
        stream = io.StringIO()
        result_df.to_csv(stream, index=False)
        stream.seek(0)
        return StreamingResponse(
            stream,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )