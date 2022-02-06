from src.example.basic import BasicPipeline
from src.example.deep import DeepAccountLevelPipeline, DeepTweetLevelPipeline
from src.pipeline import BaseDetectorPipeline
from src.example.scalable import ScalablePipeline


if __name__ == "__main__":
    # pipeline = BasicPipeline(dataset_name='MIB')
    # pipeline.run()
    # pipeline = BasicPipeline(dataset_name='TwiBot-20')
    # pipeline.run()
    # pipeline = ScalablePipeline(dataset_name='MIB')
    # pipeline.run()
    # pipeline = ScalablePipeline(dataset_name='TwiBot-20')
    # pipeline.run()
    # pipeline = DeepAccountLevelPipeline(dataset_name='MIB')
    # pipeline.run()
    # pipeline = DeepAccountLevelPipeline(dataset_name='TwiBot-20')
    # pipeline.run()
    # pipeline = DeepTweetLevelPipeline(dataset_name='MIB')
    # pipeline.run()
    pipeline = DeepTweetLevelPipeline(dataset_name='TwiBot-20')
    pipeline.run()