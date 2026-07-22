from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from anyio.to_thread import current_default_thread_limiter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from neurapy.robot import Robot
except Exception:
    Robot = None

BASE = Path(__file__).resolve().parent
INDEX_FILE = BASE / "index.html"
STATIC_DIR = BASE / "static"
if not INDEX_FILE.exists() and (STATIC_DIR / "index.html").exists():
    INDEX_FILE = STATIC_DIR / "index.html"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

robot = None
try:
    robot = Robot() if Robot is not None else None
except Exception:
    robot = None

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

class Command(BaseModel):
    action: str
    data: dict=Field(default_factory=dict)

current_step_mm=10
current_velocity=50
current_jog_mode="base"
current_started= False


def gen_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        jpg = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')


def require_robot():
    if robot is None:
        raise RuntimeError("NeuraPy is not available in this environment.")
    return robot

def step_meters():
    return current_step_mm/1000.0


def safe_cartesian_nudge(offset):
    r = require_robot()
    if not r.is_robot_in_automatic_mode():
        raise RuntimeError("Robot is not in Automatic mode on the pendant.")
    r.init_program()
    try:
        r.move_linear_relative(cartesian_offset=offset)
    finally:
        r.stop()


def safe_gripper(action):
    r = require_robot()
    if not r.is_robot_in_automatic_mode():
        raise RuntimeError("Robot is not in Automatic mode on the pendant.")
    r.init_program()
    try:
        if action == "close":
            r.grasp()
            return "closed"
        r.release()
        return "opened"
    finally:
        r.stop()

def safe_joint_nudge(joint,direction):
    r=require_robot()
    if not r.is_robot_in_automatic_mode():
        raise RuntimeError("Robot is not in Automatic mode on the pendant.")
    r.init_program()
    try:
        step = step_meters() * (1 if direction > 0 else -1)
        current = list(r.get_current_joint_angles())
        idx = int(joint) - 1
        if idx < 0 or idx >= len(current):
            raise RuntimeError(f"Invalid joint: {joint}")
        current[idx] = current[idx] + step
        if hasattr(r, "move_joints_absolute"):
            r.move_joints_absolute(current)
        elif hasattr(r, "move_joint_absolute"):
            r.move_joint_absolute(current)
        else:
            raise RuntimeError("No joint movement method available in this NeuraPy version.")
    finally:
        r.stop()
def safe_orientation_nudge(axis, direction):
    r = require_robot()
    if not r.is_robot_in_automatic_mode():
        raise RuntimeError("Robot is not in Automatic mode on the pendant.")
    r.init_program()
    try:
        step = step_meters() * (1 if direction > 0 else -1)
        offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        axis = axis.upper()
        if axis == "RX":
            offset[3] = step
        elif axis == "RY":
            offset[4] = step
        elif axis == "RZ":
            offset[5] = step
        else:
            raise RuntimeError(f"Invalid orientation axis: {axis}")
        r.move_linear_relative(cartesian_offset=offset)
    finally:
        r.stop()

@app.on_event("shutdown")
def shutdown_event():
    try:
        pipeline.stop()
    except Exception:
        pass
    try:
        if robot is not None:
            robot.stop()
    except Exception:
        pass


@app.get("/")
def index():
    if INDEX_FILE.exists():
        return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/video")
def video():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/command")
def api_command(cmd: Command):
    global current_step_mm, current_velocity, current_jog_mode, current_started
    try:
        action = cmd.action.upper()
        data=cmd.data or{}
        step = current_step_mm / 1000.0

        print(f"COMMAND: {action}", flush=True)
        if action == "START":
            print("START received", flush=True)
            current_started = True
            return {"status": "started"}

        if action == "STOP":
            print("STOP received", flush=True)
            current_started = False
            if robot is not None:
                robot.stop()
            return {"status": "stopped"}

        if action == "SET_STEP":
            current_step_mm = float(data.get("step", current_step_mm))
            print(f"Step changed to: {current_step_mm} mm", flush=True)
            return {"ok": True, "step": current_step_mm}

        if action == "SET_VELOCITY":
            current_velocity = int(data.get("velocity", current_velocity))
            print(f"Velocity changed to: {current_velocity}%", flush=True)
            return {"ok": True, "velocity": current_velocity}
        if action == "HOME":
            return {"status": "ok", "moved": "HOME"}
        if action == "SYNC_FRAME":
            return {"status": "ok", "synced": True}
        if action == "UP":
            print("UP received", flush=True)
            safe_cartesian_nudge([0.0, 0.0, step, 0.0, 0.0, 0.0])
            return {"status": "ok", "moved": "UP"}
        if action == "DOWN":
            print("DOWN received", flush=True)
            safe_cartesian_nudge([0.0, 0.0, -step, 0.0, 0.0, 0.0])
            return {"status": "ok", "moved": "DOWN"}
        if action == "LEFT":
            print("LEFT received", flush=True)
            safe_cartesian_nudge([0.0, -step, 0.0, 0.0, 0.0, 0.0])
            return {"status": "ok", "moved": "LEFT"}
        if action == "RIGHT":
            print("RIGHT received", flush=True)
            safe_cartesian_nudge([0.0, step, 0.0, 0.0, 0.0, 0.0])
            return {"status": "ok", "moved": "RIGHT"}
        if action == "FORWARD":
            print("FORWARD received", flush=True)
            safe_cartesian_nudge([step, 0.0, 0.0, 0.0, 0.0, 0.0])
            return {"status": "ok", "moved": "FORWARD"}
        if action == "BACKWARD":
            print("BACKWARD received", flush=True)
            safe_cartesian_nudge([-step, 0.0, 0.0, 0.0, 0.0, 0.0])
            return {"status": "ok", "moved": "BACKWARD"}
        if action in ("RX+", "RX-"):
            safe_orientation_nudge("RX", 1 if action.endswith("+") else -1)
            return {"status": "ok", "moved": action}
        if action in ("RY+", "RY-"):
            safe_orientation_nudge("RY", 1 if action.endswith("+") else -1)
            return {"status": "ok", "moved": action}
        if action in ("RZ+", "RZ-"):
            safe_orientation_nudge("RZ", 1 if action.endswith("+") else -1)
            return {"status": "ok", "moved": action}
        if action == "JOGJOINT":
            joint = int(data.get("joint", 1))
            direction = int(data.get("dir", 1))
            safe_joint_nudge(joint, direction)
            return {"status": "ok", "moved": f"J{joint}{'+' if direction > 0 else '-'}"}
        if action == "GRIP_CLOSE":
            print("GRIP_CLOSE received", flush=True)
            if robot is None:
                return {"status": "mock", "gripper": "closed"}
            return {"status": "ok", "gripper": safe_gripper("close")}
        if action == "GRIP_OPEN":
            print("GRIP_OPEN received", flush=True)
            if robot is None:
                return {"status": "mock", "gripper": "opened"}
            return {"status": "ok", "gripper": safe_gripper("open")}
        if action == "SET_JOG_MODE":
            mode = str(data.get("mode", current_jog_mode)).lower()
            if mode not in ("base", "cartesian", "joint"):
                raise HTTPException(status_code=400, detail=f"Invalid jog mode: {mode}")
            current_jog_mode = mode
            print(f"Jog mode changed to {current_jog_mode}", flush=True)
            return {"ok": True, "jog_mode": current_jog_mode}
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
def api_status():
    print("STATUS requested", flush=True)
    try:
        if robot is None:
            return {
                "automatic_mode": False,
                "joints": [0,0,0,0,0,0],
                "started": current_started,
                "automatic": False,
                "gripper": "open",
                "inputs": {"DI 01": 0, "DI 02": 0, "Tool DI 1": 0},
                "outputs": {"DO 01": 0, "DO 02": 0, "Tool DO 1": 0},
                "cartesian": {"X": 0, "Y": 0, "Z": 0, "Rx": 0, "Ry": 0, "Rz": 0},
            }
        joints = robot.get_current_joint_angles()
        auto = robot.is_robot_in_automatic_mode()
        return {"automatic_mode": auto, "joints": joints, "started": current_started, "automatic": auto}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)