from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np
import io
import cv2
import uuid
import os

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def read_root():
    # Читаем HTML-файл из папки frontend
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


# --- CORS настройки ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Конфигурация ---
UPLOAD_FOLDER = "uploads"
THUMBNAIL_SIZE = (200, 200)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Цвета и подписи классов ---
CLASS_COLORS = {
    0: ((255, 50, 50), 'пора'),
    1: ((0, 255, 0), 'включение'),
    2: ((0, 0, 255), 'подрез'),
    3: ((255, 255, 0), 'прожог'),
    4: ((255, 0, 255), 'трещина'),
    5: ((0, 255, 255), 'наплыв'),
    6: ((255, 64, 64), 'эталон1'),
    7: ((64, 255, 64), 'эталон2'),
    8: ((0, 0, 128), 'эталон3'),
    9: ((128, 128, 0), 'пора-скрытая'),
    10: ((128, 0, 128), 'утяжина'),
    11: ((0, 128, 128), 'несплавление'),
    12: ((192, 192, 192), 'непровар корня'),
}

# --- Загрузка модели YOLO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("model.pt")    # убедитесь, что файл model.pt лежит рядом
model.to(device)
model.eval()

# --- Pydantic-модели для predict/ ---
class PredictionItem(BaseModel):
    class_id: int

class PredictResponse(BaseModel):
    predictions: List[PredictionItem]

# --- Pydantic-модель для predict_multiple/ ---
class PredictionResult(BaseModel):
    id: str
    image_url: str
    thumbnail_url: str
    classes: List[int]
    metrics: dict

# --- Вспомогательные функции ---
def save_uploaded_file(file: UploadFile) -> str:
    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{file.filename}")
    with open(path, "wb") as buf:
        buf.write(file.file.read())
    return path

def generate_thumbnail(image_path: str) -> str:
    thumb_path = os.path.join(UPLOAD_FOLDER, f"thumb_{os.path.basename(image_path)}")
    img = Image.open(image_path)
    img.thumbnail(THUMBNAIL_SIZE)
    img.save(thumb_path)
    return thumb_path

async def predict_classes(data: bytes) -> List[int]:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img)
    result = model.predict(arr, conf=0.25)[0]
    return [int(box.cls.item()) for box in result.boxes]

# --- Эндпоинт: один файл, возвращает JSON для фронтенда ---
@app.post("/predict/", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Возвращает:
    {
      "predictions": [
        {"class_id": 0},
        {"class_id": 12},
        ...
      ]
    }
    """
    try:
        data = await file.read()
        classes = await predict_classes(data)
        preds = [PredictionItem(class_id=cls) for cls in classes]
        return PredictResponse(predictions=preds)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# --- Эндпоинт: множественные файлы ---
@app.post("/predict_multiple/")
async def predict_multiple(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        results = []
        for f in files:
            # Сохраняем оригинал и генерируем превью
            path = save_uploaded_file(f)
            thumb = generate_thumbnail(path)

            # Предсказания классов
            data = open(path, "rb").read()
            cls_list = await predict_classes(data)

            results.append(PredictionResult(
                id=str(uuid.uuid4()),
                image_url=f"/uploads/{os.path.basename(path)}",
                thumbnail_url=f"/uploads/{os.path.basename(thumb)}",
                classes=cls_list,
                metrics={"count": len(cls_list)}
            ).dict())

        return {"success": True, "results": results}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# --- Эндпоинт: получение результатов с фильтрацией/пагинацией ---
@app.get("/results")
async def get_results(page: int = 1, filter_class: Optional[int] = None):
    try:
        # TODO: подключите логику чтения из БД или файлов
        filtered = []  # заглушка
        return {"results": filtered}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# --- Точка входа ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

