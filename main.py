from src.pipeline import BaseDetectorPipeline
from src.example.scalable import ScalablePipeline


if __name__ == "__main__":
    pipeline = ScalablePipeline(verbose=True)
    pipeline.run()