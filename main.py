from src.example.basic import BasicPipeline
from src.example.bilstm import BidirectionalLSTMPipeline
from src.example.classification import ClassificationPipeline
from src.example.deep import DeepAccountLevelPipeline, DeepTweetLevelPipeline
from src.example.digital_dna import DnaPipeline
from src.example.scalable import ScalablePipeline


if __name__ == "__main__":
    # pipeline = BasicPipeline()
    # pipeline = ScalablePipeline()
    # pipeline = DeepAccountLevelPipeline()
    # pipeline = DeepTweetLevelPipeline()
    # pipeline = ClassificationPipeline()
    # pipeline = DnaPipeline()
    pipeline = BidirectionalLSTMPipeline()

    pipeline.run(dataset_name='MIB')
    pipeline.run(dataset_name='TwiBot-20')