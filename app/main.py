# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from src.load_data import load_model, load_store
import pandas as pd
import numpy as np
import yaml
import io

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

config_path = PROJECT_ROOT / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()
meta_data = load_store()

@app.get("/", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
        <head>
            <title>File Upload for Prediction</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                table { border-collapse: collapse; margin-top: 10px; }
                th, td { border: 1px solid #999; padding: 6px 12px; text-align: center; }
                th { background-color: #f2f2f2; }
                h2 { color: #333; }
                #preview { margin-top: 20px; }
            </style>
        </head>
        <body>
            <h2>Upload your CSV file for prediction</h2>
            <form id="uploadForm" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".csv" required>
                <button type="submit" name="action" value="preview">Preview Predictions</button>
                <button type="submit" name="action" value="download">Download CSV</button>
            </form>
            <div id="preview"></div>
            
            <script>
                const form = document.getElementById('uploadForm');
                const previewDiv = document.getElementById('preview');
                
                form.addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const formData = new FormData(form);
                    const action = e.submitter.value; // preview or download
                    formData.append("action", action); // include action for backend

                    if(action === "preview") {
                        const response = await fetch('/handle_upload', {
                            method: 'POST',
                            body: formData
                        });
                        const data = await response.json();

                        // Build table preview
                        let tableHTML = "<h3>Preview (" + data.total_predictions + " predictions)</h3>";
                        tableHTML += "<table><tr><th>#</th><th>Prediction</th></tr>";
                        data.prediction_preview.forEach((p, idx) => {
                            tableHTML += "<tr><td>" + (idx+1) + "</td><td>" + p + "</td></tr>";
                        });
                        tableHTML += "</table>";
                        previewDiv.innerHTML = tableHTML;
                    } else if (action === "download") {
                        // For download, send via fetch and trigger CSV download manually
                        const response = await fetch('/handle_upload', {
                            method: 'POST',
                            body: formData
                        });
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = "predictions.csv";
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                        window.URL.revokeObjectURL(url);
                    }
                });
            </script>
        </body>
    </html>
    """

@app.post("/handle_upload")
async def handle_upload(file: UploadFile = File(...), action: str = Form(...)):
    df = pd.read_csv(file.file)

    pred_sqrt = model.predict(df)**2
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