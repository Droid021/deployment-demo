from fastai import *
from fastai.vision import *
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

import aiohttp
import asyncio
import uvicorn

from io import BytesIO

model = 'export.pkl'
classes = ['basketball', 'football', 'rugby', 'roses', 'lilies']

path = Path(__file__).parent

app = Starlette()

app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])

app.mount('/static', StaticFiles(directory='app/static'))

def setup_learner():
    try:
        learn = load_learner('models', model)
        return learn
    except RuntimeError as e:
        print("Couldn't load model")

# loop = asyncio.get_event_loop()
# tasks = [asyncio.ensure_future(setup_learner())]
# learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
#loop.close()
learn = setup_learner()

@app.route('/')
async def homepage(request):
    html_file = path/'view'/'index.html'
    return HTMLResponse(html_file.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]

    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level='info')