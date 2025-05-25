'''from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np
import io
import cv2

app = FastAPI()

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


# Разрешение всех CORS (можно ограничить для продакшена)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели YOLO
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("app/model.pt")  
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Ошибка при загрузке модели: {e}")
'''
'''@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Конвертация в numpy и предсказание
        results = model(image)

        # Получение данных о предсказаниях
        predictions = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    b = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    c = int(box.cls[0].item())  # класс
                    conf = float(box.conf[0].item())  # вероятность
                    predictions.append({
                        "box": b,
                        "class_id": c,
                        "confidence": conf
                    })

        return JSONResponse(content={"success": True, "predictions": predictions})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})'''
        
'''
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        results = model(image)[0]

        # Визуализация боксов
        for box in results.boxes:
            cls = int(box.cls.item())
            color, _ = CLASS_COLORS.get(cls, ((255, 255, 255), 'неизвестно'))
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)
            label = CLASS_COLORS.get(cls, (None, ''))[1]
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Преобразуем обратно в изображение и в поток
        img_pil = Image.fromarray(image_np)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/predict_json/")
async def predict_json(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = model(image)

        predictions = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    b = box.xyxy[0].tolist()
                    c = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    predictions.append({
                        "box": b,
                        "class_id": c,
                        "confidence": conf
                    })

        return JSONResponse(content={"success": True, "predictions": predictions})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)}) '''
'''
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List
from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np
import io
import cv2

app = FastAPI()

# Разрешение всех CORS (можно ограничить для продакшена)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Цвета и подписи классов
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

# Загрузка модели YOLO
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("app/model.pt")  # Убедись, что путь к модели корректен
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Ошибка при загрузке модели: {e}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        results = model.predict(image, conf=0.25)

        used_classes = set()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    xyxy = box.xyxy.cpu().numpy().astype(int)[0]
                    x1, y1, x2, y2 = xyxy

                    color, label = CLASS_COLORS.get(cls, ((255, 255, 255), 'неизвестно'))
                    used_classes.add((label, color))

                    # Отрисовка прямоугольника и подписи
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Добавим список классов снизу изображения
        legend_height = 30 * len(used_classes) + 20
        new_image = np.ones((image_np.shape[0] + legend_height, image_np.shape[1], 3), dtype=np.uint8) * 255
        new_image[:image_np.shape[0], :, :] = image_np

        y_offset = image_np.shape[0] + 30
        for label, color in sorted(used_classes):
            cv2.putText(new_image, f"{label}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 30

        # Преобразуем в PIL и отдадим через StreamingResponse
        img_pil = Image.fromarray(new_image)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/predict_json/")
async def predict_json(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = model.predict(image, conf=0.25)

        predictions = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    b = box.xyxy[0].tolist()
                    c = int(box.cls.item())
                    conf = float(box.conf.item())
                    predictions.append({
                        "box": b,
                        "class_id": c,
                        "class_name": CLASS_COLORS.get(c, (None, 'неизвестно'))[1],
                        "confidence": conf
                    })

        return JSONResponse(content={"success": True, "predictions": predictions})

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})



class PredictionResult(BaseModel):
    id: str
    image_url: str
    thumbnail_url: str
    classes: List[int]
    metrics: dict

@app.post("/predict_multiple/")
async def predict_multiple(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        # Ваша логика обработки
        result = PredictionResult(
            id=str(uuid.uuid4()),
            image_url=upload_image(file),
            thumbnail_url=create_thumbnail(file),
            classes=[0, 12], # Пример классов
            metrics={"accuracy": 0.95}
        )
        results.append(result)
    
    return {"success": True, "results": results}

@app.get("/results")
async def get_results(page: int = 1, filter_class: int = None):
    # Логика пагинации и фильтрации
    return {"results": filtered_results}
    '''



#latest
'''
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация
UPLOAD_FOLDER = "uploads"
THUMBNAIL_SIZE = (200, 200)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Цвета и подписи классов (исправленный формат)
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

# Загрузка модели YOLO
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("model.pt")  # Убедитесь в правильности пути
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {e}")

# Вспомогательные функции
def save_uploaded_file(file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path

def generate_thumbnail(image_path: str) -> str:
    thumbnail_path = os.path.join(UPLOAD_FOLDER, f"thumb_{os.path.basename(image_path)}")
    img = Image.open(image_path)
    img.thumbnail(THUMBNAIL_SIZE)
    img.save(thumbnail_path)
    return thumbnail_path

# Pydantic модели
class PredictionResult(BaseModel):
    id: str
    image_url: str
    thumbnail_url: str
    classes: List[int]
    metrics: dict

# Эндпоинты
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)  # Конвертация в numpy array

        # Исправленный вызов predict
        results = model.predict(image_np, conf=0.25)

        used_classes = set()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    xyxy = box.xyxy.cpu().numpy().astype(int)[0]
                    x1, y1, x2, y2 = xyxy

                    color, label = CLASS_COLORS.get(cls, ((255, 255, 255), 'неизвестно'))
                    used_classes.add((label, color))

                    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(image_np, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Генерация изображения с легендой
        legend_height = 30 * len(used_classes) + 20
        new_image = np.ones((image_np.shape[0] + legend_height, image_np.shape[1], 3), 
                          dtype=np.uint8) * 255
        new_image[:image_np.shape[0], :, :] = image_np

        y_offset = image_np.shape[0] + 30
        for label, color in sorted(used_classes):
            cv2.putText(new_image, f"{label}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 30

        img_pil = Image.fromarray(new_image)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/predict_multiple/")
async def predict_multiple(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        results = []
        for file in files:
            file_path = save_uploaded_file(file)
            thumbnail_path = generate_thumbnail(file_path)
            
            # Здесь должна быть ваша логика предсказания
            # Пример временной реализации:
            result = PredictionResult(
                id=str(uuid.uuid4()),
                image_url=f"/uploads/{os.path.basename(file_path)}",
                thumbnail_url=f"/uploads/{os.path.basename(thumbnail_path)}",
                classes=[0, 12],  # Замените реальными классами
                metrics={"accuracy": 0.95}  # Замените реальными метриками
            )
            results.append(result.dict())

        return {"success": True, "results": results}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/results")
async def get_results(page: int = 1, filter_class: Optional[int] = None):
    try:
        # Здесь должна быть логика получения и фильтрации результатов
        # Пример временной реализации:
        filtered_results = []  # Замените реальными данными
        return {"results": filtered_results}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    '''
####################

'''
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
import torch, io, os, uuid
import numpy as np
import cv2
from PIL import Image
import uvicorn

app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
THUMBNAIL_SIZE = (200, 200)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("model.pt").to(device).eval()

class Prediction(BaseModel):
    id: str
    image_url: str
    thumbnail_url: str
    classes: List[int]


def save_file(file: UploadFile) -> str:
    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{file.filename}")
    with open(path, "wb") as buf:
        buf.write(file.file.read())
    return path


def make_thumbnail(path: str) -> str:
    thumb = os.path.join(UPLOAD_FOLDER, f"thumb_{os.path.basename(path)}")
    img = Image.open(path)
    img.thumbnail(THUMBNAIL_SIZE)
    img.save(thumb)
    return thumb

@app.post("/predict/", response_class=StreamingResponse)
async def predict_single(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)
        res = model.predict(arr, conf=0.25)[0]
        used = set()
        for box in res.boxes:
            cls = int(box.cls.item())
            used.add(cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            color, label = CLASS_COLORS[cls]
            cv2.rectangle(arr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(arr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # Legend
        h = arr.shape[0] + 30 * len(used)
        canvas = np.ones((h, arr.shape[1], 3), dtype=np.uint8) * 255
        canvas[:arr.shape[0]] = arr
        y = arr.shape[0] + 20
        for cls in sorted(used):
            _, label = CLASS_COLORS[cls]
            color, _ = CLASS_COLORS[cls]
            cv2.putText(canvas, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30
        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, 'JPEG')
        buf.seek(0)
        return StreamingResponse(buf, media_type='image/jpeg')
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/predict_multiple/")
async def predict_multiple(files: List[UploadFile] = File(...)):
    items = []
    for f in files:
        path = save_file(f)
        thumb = make_thumbnail(path)
        # Здесь вызывается такой же predict, как в /predict/, чтобы получить реальные классы
        data = await f.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)
        res = model.predict(arr, conf=0.25)[0]
        used = set(int(box.cls.item()) for box in res.boxes)
        items.append(Prediction(
            id=str(uuid.uuid4()),
            image_url=f"/uploads/{os.path.basename(path)}",
            thumbnail_url=f"/uploads/{os.path.basename(thumb)}",
            classes=list(used)
        ).dict())
    return {"results": items}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

'''

###########
'''from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
import torch, io, os, uuid
import numpy as np
import cv2
from PIL import Image
import uvicorn

import os
print("Current working dir:", os.getcwd())

# Создаем папку uploads до монтирования
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
THUMBNAIL_SIZE = (200, 200)

app = FastAPI()
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_COLORS = {i:COL for i,COL in enumerate([
    ((255,50,50),'пора'),((0,255,0),'включение'),((0,0,255),'подрез'),
    ((255,255,0),'прожог'),((255,0,255),'трещина'),((0,255,255),'наплыв'),
    ((255,64,64),'эталон1'),((64,255,64),'эталон2'),((0,0,128),'эталон3'),
    ((128,128,0),'пора-скрытая'),((128,0,128),'утяжина'),((0,128,128),'несплавление'),
    ((192,192,192),'непровар корня')])}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("model.pt").to(device).eval()

class PredictJSON(BaseModel):
    classes: List[int]

class Prediction(BaseModel):
    id: str
    image_url: str
    thumbnail_url: str
    classes: List[int]

async def predict_classes(data_bytes: bytes) -> List[int]:
    img = Image.open(io.BytesIO(data_bytes)).convert("RGB")
    arr = np.array(img)
    res = model.predict(arr, conf=0.25)[0]
    return [int(box.cls.item()) for box in res.boxes]

@app.post("/predict/", response_class=StreamingResponse)
async def predict_single(file: UploadFile = File(...)):
    data = await file.read()
    classes = await predict_classes(data)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img)
    for cls in classes:
        box = next(b for b in model.predict(arr, conf=0.25)[0].boxes if int(b.cls)==cls)
        x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
        color, label = CLASS_COLORS[cls]
        cv2.rectangle(arr,(x1,y1),(x2,y2),color,2)
        cv2.putText(arr,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
    # Рисуем легенду
    h = arr.shape[0] + 30*len(classes)
    canvas = np.ones((h,arr.shape[1],3),dtype=np.uint8)*255
    canvas[:arr.shape[0]] = arr
    y = arr.shape[0] + 20
    for cls in classes:
        color, label = CLASS_COLORS[cls]
        cv2.putText(canvas,label,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
        y +=30
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf,'JPEG'); buf.seek(0)
    return StreamingResponse(buf, media_type='image/jpeg')

@app.post("/predict_json/", response_model=PredictJSON)
async def get_classes(file: UploadFile = File(...)):
    data = await file.read()
    classes = await predict_classes(data)
    return {"classes": classes}

@app.post("/predict_multiple/", response_model=List[Prediction])
async def predict_multiple(files: List[UploadFile] = File(...)):
    items = []
    for f in files:
        data = await f.read()
        cls_list = await predict_classes(data)
        path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{f.filename}")
        with open(path,'wb') as buf: buf.write(data)
        thumb_path = os.path.join(UPLOAD_FOLDER, f"thumb_{os.path.basename(path)}")
        img = Image.open(path); img.thumbnail(THUMBNAIL_SIZE); img.save(thumb_path)
        items.append(Prediction(id=str(uuid.uuid4()), image_url=f"/uploads/{os.path.basename(path)}", thumbnail_url=f"/uploads/{os.path.basename(thumb_path)}", classes=cls_list))
    return items

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)'''



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

