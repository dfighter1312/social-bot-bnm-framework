from src.example.basic import BasicPipeline
from src.example.deep import DeepAccountLevelPipeline, DeepTweetLevelPipeline
from src.example.scalable import ScalablePipeline


if __name__ == "__main__":
    # pipeline = BasicPipeline()
    pipeline = ScalablePipeline()
    # pipeline = DeepAccountLevelPipeline()
    # pipeline = DeepTweetLevelPipeline()

    pipeline.run(dataset_name='MIB')
    pipeline.run(dataset_name='TwiBot-20')