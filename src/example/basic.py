from sklearn.ensemble import RandomForestClassifier
from src.pipeline import BaseDetectorPipeline

class BasicPipeline(BaseDetectorPipeline):
    
    def __init__(self, **kwargs):
        super().__init__(
            user_features='all',
            **kwargs
        )

    def classify(self, X_train, X_dev, y_train, y_dev):
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)