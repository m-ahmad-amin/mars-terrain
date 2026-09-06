from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.mission import get_mission

HERE = Path(__file__).resolve().parent
app = FastAPI(title="mars-terrain")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


class StartBody(BaseModel):
    index: int | None = None
    mode: str = "autonav"
    continuous: bool = True
    use_model: bool = False


class GoalBody(BaseModel):
    y: int
    x: int


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/scenes")
def scenes() -> dict:
    m = get_mission()
    return {
        "total": len(m.pairs),
        "index": m.index,
        "stems": [p[0] for p in m.pairs],
    }


@app.post("/api/start")
def start(body: StartBody) -> dict:
    m = get_mission()
    if body.use_model:
        m.enable_model()
    else:
        m.use_model = False
    return m.start(body.index, body.mode, body.continuous)


@app.post("/api/step")
def step() -> dict:
    return get_mission().step()


@app.post("/api/goal")
def goal(body: GoalBody) -> dict:
    return get_mission().set_goal(body.y, body.x)


@app.get("/api/state")
def state() -> dict:
    return get_mission().snapshot()


def main() -> None:
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
