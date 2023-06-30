from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from .routers import tasks
from .utils.fastapi import resp_fail

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return resp_fail(msg=exc.errors())


app.include_router(
    tasks.router,
    prefix="/tasks",
)


@app.get("/")
async def root():
    return {"code": 0, "msg": "Welcome to xroute_env!"}
