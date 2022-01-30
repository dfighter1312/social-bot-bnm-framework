from tabnanny import verbose
from src.example.deep import DeepAccountLevelPipeline, DeepTweetLevelPipeline
from src.pipeline import BaseDetectorPipeline
from src.example.scalable import ScalablePipeline


if __name__ == "__main__":
    # pipeline = ScalablePipeline()
    # pipeline.run()
    # pipeline = DeepAccountLevelPipeline()
    # pipeline.run()
    pipeline = DeepTweetLevelPipeline()
    pipeline.run()