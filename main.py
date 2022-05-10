from src.example.ablation import AblationPipeline
from src.example.basic import BasicPipeline
from src.example.bilstm import BidirectionalLSTMPipeline
from src.example.classification import ClassificationPipeline
from src.example.deep import DeepAccountLevelPipeline, DeepTweetLevelPipeline
from src.example.digital_dna import DnaPipeline
from src.example.scalable import ScalablePipeline
from src.example.turing import TuringPipeline


if __name__ == "__main__":
    # pipeline = BasicPipeline()
    # pipeline = ScalablePipeline()
    # pipeline = DeepAccountLevelPipeline()
    # pipeline = DeepTweetLevelPipeline()
    # pipeline = ClassificationPipeline()
    # pipeline = DnaPipeline()
    # pipeline = BidirectionalLSTMPipeline()
    # pipeline = TuringPipeline()
    pipeline = AblationPipeline(
        units=300,
        dl_types='dense',
        use_tweet=True,
        encoder='tfidf',
        num_layers=10
    )

    # Set nrows to any number to receive a subset of that data
    # or None to get the whole dataset.

    # pipeline.run(dataset_name='MIB')
    # pipeline.run(dataset_name='MIB-2')
    pipeline.run(dataset_name='TwiBot-20', nrows=10)
    pass