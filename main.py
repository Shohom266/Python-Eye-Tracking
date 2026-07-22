from pathlib import Path
import math
import time

import cv2
import numpy as np
import pyrealsense2 as rs
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
    data: dict = Field(default_factory=dict)


current_step_mm = 10.0
current_rot_step_deg = 5.0
current_velocity = 50
current_jog_mode = "base"   # base | tcp | joint
robot_started = False


def gen_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        jpg = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )


def require_robot():
    if robot is None:
        raise RuntimeError("NeuraPy is not available in this environment.")
    return robot


def step_meters():
    return float(current_step_mm) / 1000.0


def rot_step_radians():
    return math.radians(float(current_rot_step_deg))


def is_motion_state_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "3104" in msg
        or "Robot state does not allow motion to execute" in msg
        or "Please try restarting the program" in msg
    )


def raise_clean_motion_error(exc: Exception):
    if is_motion_state_error(exc):
        raise RuntimeError(
            "Robot program is not ready for motion. Restart the program on the pendant, "
            "make sure the robot is in Automatic mode, then press Start again."
        )
    raise exc


def get_robot_debug_state():
    r = require_robot()
    state = {}

    for name in ("program_status", "motion_status", "get_errors"):
        if hasattr(r, name):
            try:
                state[name] = getattr(r, name)()
            except Exception as e:
                state[name] = f"error: {e}"

    try:
        state["automatic"] = r.is_robot_in_automatic_mode()
    except Exception as e:
        state["automatic"] = f"error: {e}"

    return state


def try_recover_robot_state():
    r = require_robot()
    recovery = {}

    if hasattr(r, "reset_errors"):
        try:
            recovery["reset_errors"] = r.reset_errors()
        except Exception as e:
            recovery["reset_errors"] = f"error: {e}"

    if hasattr(r, "reset_collision"):
        try:
            recovery["reset_collision"] = r.reset_collision()
        except Exception as e:
            recovery["reset_collision"] = f"error: {e}"

    return recovery


def ensure_motion_ready(r):
    if not r.is_robot_in_automatic_mode():
        raise RuntimeError("Robot is not in Automatic mode on the pendant.")
    r.init_program()


def stop_motion_program(r):
    try:
        r.stop()
    except Exception:
        pass


def ensure_valid_jog_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode in ("cartesian", "tool", "tcp"):
        return "tcp"
    if mode in ("base", "joint"):
        return mode
    raise RuntimeError(f"Invalid jog mode: {mode}")


def velocity_to_linear_speed():
    v = max(1, min(100, int(current_velocity)))
    return max(0.01, min(1.0, v / 100.0))


def safe_cartesian_nudge(offset, frame="Base"):
    r = require_robot()
    try:
        ensure_motion_ready(r)
        try:
            r.move_linear_relative(
                speed=velocity_to_linear_speed(),
                acceleration=min(1.0, max(0.05, velocity_to_linear_speed())),
                jerk=500.0,
                rotation_speed=0.5,
                rotation_acceleration=1.57,
                rotation_jerk=500.0,
                blocking=True,
                offset_frame=frame,
                cartesian_offset=offset,
                current_joint_angles=r.get_current_joint_angles(),
            )
        finally:
            stop_motion_program(r)
    except Exception as e:
        raise_clean_motion_error(e)


def safe_orientation_nudge(axis, direction, frame="Base"):
    try:
        step = rot_step_radians()
        offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        signed = step if direction >= 0 else -step

        axis = axis.upper()
        if axis == "RX":
            offset[3] = signed
        elif axis == "RY":
            offset[4] = signed
        elif axis == "RZ":
            offset[5] = signed
        else:
            raise RuntimeError(f"Invalid orientation axis: {axis}")

        safe_cartesian_nudge(offset, frame=frame)
    except Exception as e:
        raise_clean_motion_error(e)


def safe_joint_step(joint, direction):
    r = require_robot()
    try:
        if not r.is_robot_in_automatic_mode():
            raise RuntimeError("Robot is not in Automatic mode on the pendant.")

        joint = int(joint)
        if joint < 1 or joint > 6:
            raise RuntimeError(f"Invalid joint index: {joint}")

        current = list(r.get_current_joint_angles())
        before = current.copy()

        step_rad = max(0.02, min(0.12, current_step_mm / 1000.0))
        if direction < 0:
            step_rad = -step_rad

        current[joint - 1] += step_rad

        r.init_program()
        try:
            ok = r.move_joint(
                speed=10.0,
                acceleration=10.0,
                target_joint=[current],
                current_joint_angles=before,
                enable_safety=True,
            )
        finally:
            r.stop()

        after = list(r.get_current_joint_angles())

        return {
            "ok": ok,
            "joint": joint,
            "direction": direction,
            "before": before,
            "after": after,
            "motion_status": r.motion_status(),
            "program_status": r.program_status(),
            "errors": r.get_errors(),
        }

    except Exception as e:
        raise_clean_motion_error(e)


def safe_gripper(action):
    r = require_robot()
    try:
        ensure_motion_ready(r)
        try:
            if action == "close":
                r.grasp()
                return "closed"
            elif action == "open":
                r.release()
                return "opened"
            else:
                raise RuntimeError(f"Invalid gripper action: {action}")
        finally:
            stop_motion_program(r)
    except Exception as e:
        raise_clean_motion_error(e)


def cartesian_frame_from_mode():
    if current_jog_mode == "base":
        return "Base"
    if current_jog_mode == "tcp":
        return "Tool"
    raise RuntimeError("Cartesian jogging is disabled in joint mode. Use J1-J6 buttons.")


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
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/command")
def api_command(cmd: Command):
    global current_step_mm, current_rot_step_deg, current_velocity, current_jog_mode, robot_started

    try:
        action = str(cmd.action or "").upper().strip()
        data = cmd.data or {}

        print(f"COMMAND {action}", flush=True)

        if action == "START":
            print("START received", flush=True)

            if robot is not None:
                try:
                    debug_before = get_robot_debug_state()
                    print(f"DEBUG BEFORE START: {debug_before}", flush=True)
                except Exception:
                    pass

                try:
                    try_recover_robot_state()
                except Exception:
                    pass
                try:
                    robot.stop()
                except Exception:
                    pass
                try:
                    robot.init_program()
                except Exception as e:
                    raise_clean_motion_error(e)
                finally:
                    try:
                        robot.stop()
                    except Exception:
                        pass
                try:
                    debug_after = get_robot_debug_state()
                    print(f"DEBUG AFTER START: {debug_after}", flush=True)
                except Exception:
                    pass

            robot_started = True
            return {"status": "started", "started": True}

        if action in ("STOP", "EMERGENCYSTOP"):
            robot_started = False
            print(f"{action} received", flush=True)
            if robot is not None:
                try:
                    robot.stop()
                except Exception:
                    pass
                try:
                    robot.turn_off_jog()
                except Exception:
                    pass
            return {"status": "stopped", "started": False}

        if action in ("SET_STEP", "SETSTEP"):
            current_step_mm = float(
                data.get("step", data.get("step_mm", current_step_mm))
            )
            return {"ok": True, "step": current_step_mm}

        if action in ("SET_ROT_STEP", "SETROTSTEP"):
            current_rot_step_deg = float(
                data.get("rot_step_deg", data.get("step_deg", current_rot_step_deg))
            )
            return {"ok": True, "rot_step_deg": current_rot_step_deg}

        if action in ("SET_VELOCITY", "SETVELOCITY"):
            current_velocity = int(data.get("velocity", current_velocity))
            return {"ok": True, "velocity": current_velocity}

        if action == "SET_JOG_MODE":
            mode = ensure_valid_jog_mode(data.get("mode", current_jog_mode))
            current_jog_mode = mode
            return {"ok": True, "jog_mode": current_jog_mode}

        if action == "GET_JOG_MODE":
            return {"ok": True, "jog_mode": current_jog_mode}

        if action == "HOME":
            r = require_robot()
            try:
                ensure_motion_ready(r)
                try:
                    r.move_joint("Home")
                finally:
                    stop_motion_program(r)
            except Exception as e:
                raise_clean_motion_error(e)
            return {"status": "ok", "moved": "HOME"}

        if action == "SYNCFRAME":
            return {"status": "ok", "synced": True, "jog_mode": current_jog_mode}

        step = step_meters()

        cartesian_map = {
            "UP":        [0.0, 0.0,  step, 0.0, 0.0, 0.0],
            "DOWN":      [0.0, 0.0, -step, 0.0, 0.0, 0.0],
            "LEFT":      [0.0,  step, 0.0, 0.0, 0.0, 0.0],
            "RIGHT":     [0.0, -step, 0.0, 0.0, 0.0, 0.0],
            "FORWARD":   [ step, 0.0, 0.0, 0.0, 0.0, 0.0],
            "BACKWARD":  [-step, 0.0, 0.0, 0.0, 0.0, 0.0],
            "X":         [ step, 0.0, 0.0, 0.0, 0.0, 0.0],
            "X-":        [-step, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Y":         [0.0,  step, 0.0, 0.0, 0.0, 0.0],
            "Y-":        [0.0, -step, 0.0, 0.0, 0.0, 0.0],
            "Z":         [0.0, 0.0,  step, 0.0, 0.0, 0.0],
            "Z-":        [0.0, 0.0, -step, 0.0, 0.0, 0.0],
        }

        if action in cartesian_map:
            frame = cartesian_frame_from_mode()
            print(f"{action} received in {current_jog_mode} mode -> {frame}", flush=True)
            safe_cartesian_nudge(cartesian_map[action], frame=frame)
            return {
                "status": "ok",
                "moved": action,
                "jog_mode": current_jog_mode,
                "frame": frame,
            }

        if action in ("RX", "RX-"):
            frame = cartesian_frame_from_mode()
            safe_orientation_nudge("RX", 1 if action == "RX" else -1, frame=frame)
            return {"status": "ok", "moved": action, "jog_mode": current_jog_mode, "frame": frame}

        if action in ("RY", "RY-"):
            frame = cartesian_frame_from_mode()
            safe_orientation_nudge("RY", 1 if action == "RY" else -1, frame=frame)
            return {"status": "ok", "moved": action, "jog_mode": current_jog_mode, "frame": frame}

        if action in ("RZ", "RZ-"):
            frame = cartesian_frame_from_mode()
            safe_orientation_nudge("RZ", 1 if action == "RZ" else -1, frame=frame)
            return {"status": "ok", "moved": action, "jog_mode": current_jog_mode, "frame": frame}

        if action == "JOGJOINT":
            joint = int(data.get("joint", 1))
            direction = int(data.get("dir", 1))

            result = safe_joint_step(joint, direction)
            result["status"] = "ok"
            result["moved"] = f"J{joint}{'+' if direction >= 0 else '-'}"
            result["jog_mode"] = current_jog_mode
            return result

        if action == "GRIPCLOSE":
            print("GRIPCLOSE received", flush=True)
            if robot is None:
                return {"status": "mock", "gripper": "closed"}
            return {"status": "ok", "gripper": safe_gripper("close")}

        if action == "GRIPOPEN":
            print("GRIPOPEN received", flush=True)
            if robot is None:
                return {"status": "mock", "gripper": "opened"}
            return {"status": "ok", "gripper": safe_gripper("open")}

        if action == "TOGGLEDO":
            name = str(data.get("name", "")).strip()
            return {"status": "ok", "output": name, "toggled": True}

        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
def api_status():
    try:
        if robot is None:
            return {
                "automaticMode": False,
                "automatic": False,
                "started": robot_started,
                "jogMode": current_jog_mode,
                "stepMm": current_step_mm,
                "rotStepDeg": current_rot_step_deg,
                "velocity": current_velocity,
                "joints": [0, 0, 0, 0, 0, 0],
                "cartesian": {"X": 0, "Y": 0, "Z": 0, "Rx": 0, "Ry": 0, "Rz": 0},
                "gripper": "open",
                "inputs": {"DI 01": 0, "DI 02": 0, "Tool DI 1": 0},
                "outputs": {"DO 01": 0, "DO 02": 0, "Tool DO 1": 0},
            }

        joints = robot.get_current_joint_angles()
        automatic = robot.is_robot_in_automatic_mode()

        tcp = [0, 0, 0, 0, 0, 0]
        try:
            tcp = robot.get_tcp_pose()
        except Exception:
            try:
                tcp = robot.get_tcp_pose_quaternion()[:6]
            except Exception:
                pass

        io_status = {}
        try:
            io_status = robot.get_io_status()
        except Exception:
            io_status = {}

        return {
            "automaticMode": automatic,
            "automatic": automatic,
            "started": robot_started,
            "jogMode": current_jog_mode,
            "stepMm": current_step_mm,
            "rotStepDeg": current_rot_step_deg,
            "velocity": current_velocity,
            "joints": joints,
            "cartesian": {
                "X": tcp[0] if len(tcp) > 0 else 0,
                "Y": tcp[1] if len(tcp) > 1 else 0,
                "Z": tcp[2] if len(tcp) > 2 else 0,
                "Rx": tcp[3] if len(tcp) > 3 else 0,
                "Ry": tcp[4] if len(tcp) > 4 else 0,
                "Rz": tcp[5] if len(tcp) > 5 else 0,
            },
            "gripper": "unknown",
            "inputs": {
                "DI 01": io_status.get("DI 01", 0) if isinstance(io_status, dict) else 0,
                "DI 02": io_status.get("DI 02", 0) if isinstance(io_status, dict) else 0,
                "Tool DI 1": io_status.get("Tool DI 1", 0) if isinstance(io_status, dict) else 0,
            },
            "outputs": {
                "DO 01": io_status.get("DO 01", 0) if isinstance(io_status, dict) else 0,
                "DO 02": io_status.get("DO 02", 0) if isinstance(io_status, dict) else 0,
                "Tool DO 1": io_status.get("Tool DO 1", 0) if isinstance(io_status, dict) else 0,
            },
        }
    except Exception as e:
        return {
            "automaticMode": False,
            "automatic": False,
            "started": robot_started,
            "jogMode": current_jog_mode,
            "stepMm": current_step_mm,
            "rotStepDeg": current_rot_step_deg,
            "velocity": current_velocity,
            "error": str(e),
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)