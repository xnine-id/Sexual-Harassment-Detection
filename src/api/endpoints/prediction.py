from fastapi import Depends
from src.api.middleware.auth import verify_token
from src.services.video_sexual_harassment_tracker import VideoSexualHarassmentTracker
import mimetypes
import logging
import os
import shutil
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, status, BackgroundTasks
from fastapi.responses import FileResponse
from src.core.camera_processor import CameraProcessor
from src.core.sexual_harassment_detector import SexualHarassmentDetector
from src.services.frame_renderer import FrameRenderer
from src.utils.config_loader import Config
from src.api.schemas import JobCreateResponse, JobStatusResponse
from src.database.entity.camera import Camera
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("API_PREDICTION")

# In-memory jobs state
jobs = {}

def get_prediction_router(
    config: Config,
    sexual_detector: SexualHarassmentDetector,
    renderer: FrameRenderer,
):
    router = APIRouter(tags=["Prediction"])

    output_dir = config.detection_settings.output_dir

    def process_image_sync(input_path: str, output_path: str) -> dict:
        """Process image file, run detection and return result"""
        import cv2
        frame = cv2.imread(input_path)
        if frame is None:
            raise Exception("Invalid or corrupted image file")

        result = sexual_detector.run(frame)
        rendered_frame = renderer.render(frame.copy(), result)
        cv2.imwrite(output_path, rendered_frame)

        return {
            'harassment_detected': result['class'] == 1,
            'max_score': result['score'],
            'min_score': result['score'],
            'avg_score': result['score'],
            'harassment_frequency': 1 if result['class'] == 1 else 0
        }

    def process_video_sync(input_path: str, output_path: str):
        """Process video file using the CameraProcessor engine"""
        cam_config = Camera(
            name="video_upload",
            url=input_path,
            detect_fps=config.detection_settings.detect_fps,
            is_enabled=True,
        )
        tracker = VideoSexualHarassmentTracker()

        processor = CameraProcessor(
            cam_config=cam_config,
            sexual_harassment_detector=sexual_detector,
            frame_renderer=renderer,
            tracker=tracker,
            output_path=output_path,
        )

        try:
            processor.run()
        except Exception as e:
            processor.stop_event.set()
            raise e

        return {
            'harassment_detected': len(tracker.get_scores()) > 0,
            'max_score': tracker.get_highest_score(),
            'min_score': tracker.get_lowest_score(),
            'avg_score': tracker.get_avg_scores(),
            'harassment_frequency': tracker.get_frequency()
        }

    async def bg_process_video(job_id: str, temp_input: str, output_path: str, output_filename: str):
        jobs[job_id] = {"status": "processing"}
        try:
            result = await run_in_threadpool(process_video_sync, temp_input, output_path)
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {"filename": output_filename, "url": f"/api/result/{output_filename}", "detections": result}
        except Exception as e:
            logger.exception(f"Error processing video job {job_id}: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
        finally:
            if os.path.exists(temp_input):
                os.remove(temp_input)

    async def bg_process_image(job_id: str, temp_input: str, output_path: str, output_filename: str):
        jobs[job_id] = {"status": "processing"}
        try:
            result = await run_in_threadpool(process_image_sync, temp_input, output_path)
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {"filename": output_filename, "url": f"/api/result/{output_filename}", "detections": result}
        except Exception as e:
            logger.exception(f"Error processing image job {job_id}: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
        finally:
            if os.path.exists(temp_input):
                os.remove(temp_input)

    @router.post(
        "/predict/video",
        summary="Predict sexual harassment from uploaded video",
        response_model=JobCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_token)]
    )
    async def predict_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
        """
        Upload a video, start background job for sexual harassment detection, and return job id.
        """
        if not output_dir or not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 1. Save uploaded file temporarily
        temp_input = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Prepare output path
        output_filename = f"processed_{uuid.uuid4()}_{file.filename}"
        if not output_filename.endswith(".mp4"):
            output_filename += ".mp4"
        output_path = os.path.join(output_dir, output_filename)

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending"}

        background_tasks.add_task(bg_process_video, job_id, temp_input, output_path, output_filename)

        return {"message": "Job created", "data": {"job_id": job_id}}

    @router.post(
        "/predict/image",
        summary="Predict sexual harassment from uploaded image",
        response_model=JobCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_token)]
    )
    async def predict_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
        """
        Upload an image, start background job for sexual harassment detection, and return job id.
        """
        if not output_dir or not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 1. Save uploaded file temporarily
        temp_input = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Prepare output path
        ext = os.path.splitext(file.filename)[1]
        if not ext:
            ext = ".jpg"
        output_filename = f"processed_{uuid.uuid4()}{ext}"
        output_path = os.path.join(output_dir, output_filename)

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending"}

        background_tasks.add_task(bg_process_image, job_id, temp_input, output_path, output_filename)

        return {"message": "Job created", "data": {"job_id": job_id}}

    @router.get(
        "/jobs/{job_id}",
        summary="Get job status",
        response_model=JobStatusResponse,
        dependencies=[Depends(verify_token)]
    )
    async def get_job_status(job_id: str):
        """
        Get the status of a background prediction job.
        """
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {"data": {"job_id": job_id, **jobs[job_id]}}

    @router.get("/result/{filename}", summary="Get result file")
    async def get_result(filename: str):
        """
        Serve result file.
        """
        if not output_dir:
            raise HTTPException(status_code=500, detail="Output directory not configured")

        base_dir = os.path.abspath(output_dir)
        requested_path = os.path.abspath(os.path.join(base_dir, filename))

        if not requested_path.startswith(base_dir):
            raise HTTPException(status_code=403, detail="Access denied")

        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="File not found")

        mime_type, _ = mimetypes.guess_type(requested_path)

        return FileResponse(requested_path, media_type=mime_type, filename=filename)

    return router
