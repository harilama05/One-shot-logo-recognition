import torch

from .config import PipelineConfig
from .models.arcface_model import load_arcface_model, load_embeddings
from .models.yolo_model import YoloModel
from .utils.logger import get_logger
from .workers.detect_worker import DetectWorker
from .workers.input_worker import InputWorker
from .workers.output_worker import OutputWorker
from .workers.postprocess_detect_worker import PostprocessDetectWorker
from .workers.postprocess_recog_worker import PostprocessRecogWorker
from .workers.recog_worker import RecogWorker


def run_pipeline(config: PipelineConfig) -> None:
    logger = get_logger(__name__)
    device = (
        torch.device(config.device)
        if config.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    input_worker = InputWorker(config.video_path)
    fps, width, height = input_worker.get_video_props()
    if fps <= 0 or width <= 0 or height <= 0:
        input_worker.release()
        raise ValueError("Invalid video properties for output writer.")

    yolo_model = YoloModel(config.yolo_weights, device)
    detect_worker = DetectWorker(yolo_model, conf_threshold=config.conf_threshold)
    postprocess_detect_worker = PostprocessDetectWorker()

    arcface_model = load_arcface_model(config.recog_weights, device)
    db_embeddings, db_labels = load_embeddings(config.embed_db_path)
    recog_worker = RecogWorker(
        arcface_model,
        db_embeddings,
        db_labels,
        device,
        threshold=config.recog_threshold,
    )
    postprocess_recog_worker = PostprocessRecogWorker()

    output_worker = OutputWorker(config.output_path, fps, width, height)
    logger.info("Processing started.")

    try:
        while True:
            input_item = input_worker.read()
            if input_item is None:
                break
            detected_item = detect_worker.process(input_item)
            postprocessed_item = postprocess_detect_worker.process(detected_item)
            recognized_item = recog_worker.process(postprocessed_item)
            postprocessed_recog_item = postprocess_recog_worker.process(
                input_item,
                recognized_item,
                threshold=config.recog_threshold,
            )
            output_worker.write(postprocessed_recog_item.frame_output)
    finally:
        input_worker.release()
        output_worker.release()
        logger.info("Processing finished.")
