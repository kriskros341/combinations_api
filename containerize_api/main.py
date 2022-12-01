import container_script
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from typing import List
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def streamWrapper(sums: int, args: List[int]):
    generator = container_script.stream_containers(args, sums)
    for result in generator:
        value = json.dumps(result, cls=container_script.NumpyEncoder)
        yield value + ","


@app.get("/stream/")
async def streamResults(sums: int, args: List[int] = Query(None)):
    return StreamingResponse(streamWrapper(sums, args))


@app.get("/")
async def root(sums: int, args: List[int] = Query(None)):
    serialized = container_script.containerize(args, sums)
    data = json.dumps(serialized, cls=container_script.NumpyEncoder)
    return {"result": data}