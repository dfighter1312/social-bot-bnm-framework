from src.example.ablation import AblationPipeline
from src.example.basic import BasicPipeline
from src.example.bilstm import BidirectionalLSTMPipeline
from src.example.classification import ClassificationPipeline
from src.example.deep import DeepAccountLevelPipeline, DeepTweetLevelPipeline
from src.example.digital_dna import DnaPipeline
from src.example.scalable import ScalablePipeline
from src.example.turing import TuringPipeline
from src.example.botrgcn import BotRGCNPipeline


if __name__ == "__main__":
    # pipeline = BasicPipeline()
    # pipeline = ScalablePipeline()
    # pipeline = DeepAccountLevelPipeline()
    # pipeline = DeepTweetLevelPipeline()
    # pipeline = ClassificationPipeline()
    # pipeline = DnaPipeline()
    # pipeline = BidirectionalLSTMPipeline()
    # pipeline = TuringPipeline()
    # pipeline = BotRGCNPipeline()
    pipeline = AblationPipeline(
        units=32,
        dl_types='lstm',
        use_tweet=True,
        encoder='word2vec',
        num_layers=2,
        use_tweet_metadata=False,
        use_users=False
    )

    # Set nrows to any number to receive a subset of that data
    # or None to get the whole dataset.

    # pipeline.run(dataset_name='MIB', nrows=None)
    # pipeline.run(dataset_name='MIB-2', nrows=10000)
    pipeline.run(dataset_name='TwiBot-20', nrows=None)
    pass