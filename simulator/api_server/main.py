from fastapi import FastAPI

from .routers import tasks

app = FastAPI()

app.include_router(
    tasks.router,
    prefix="/tasks",
)


@app.get("/")
async def root():
    return {"code": 0, "msg": "Welcome to xroute_env!"}
