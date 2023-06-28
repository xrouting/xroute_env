from fastapi import APIRouter
from fastapi.responses import ORJSONResponse

router = APIRouter()


@router.get("/")
async def get_tasks():
    return ORJSONResponse([{"foo": "bar"}])


@router.post("/")
async def create_tasks():
    return ORJSONResponse([{"foo": "bar"}])
