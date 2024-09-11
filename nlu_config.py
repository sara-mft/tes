import yaml
import os

class Config:
    def __init__(self, config_path="config.yaml"):
        # Load the configuration file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        # Initialize all parameters
        self._initialize_params()

    def _initialize_params(self):
        """Initialize parameters from the configuration file."""
        # General settings
        general = self.config.get("general", {})
        self.project_name = general.get("project_name", "TextIntentDetection")
        self.log_level = general.get("log_level", "INFO")
        self.save_model_path = general.get("save_model_path", "models/")
        self.data_path = general.get("data_path", "data/")
        
        # Data settings
        data = self.config.get("data", {})
        self.train_file = os.path.join(self.data_path, data.get("train_file", "train_data.csv"))
        self.validation_file = os.path.join(self.data_path, data.get("validation_file", "validation_data.csv"))
        self.test_file = os.path.join(self.data_path, data.get("test_file", "test_data.csv"))
        self.label_column = data.get("label_column", "intent")
        self.text_column = data.get("text_column", "text")
        self.text_column = data.get("negative_example_column", "negative_example")
        self.test_split_ratio = data.get("test_split_ratio", 0.2)
        
        # Model settings
        model = self.config.get("model", {})
        self.model_type = model.get("type", "cross_encoder")
        self.use_pretrained = model.get("use_pretrained", True)
        self.pretrained_model_name = model.get("pretrained_model_name", "bert-base-uncased")
        
        # Training settings
        training = self.config.get("training", {})
        self.batch_size = training.get("batch_size", 32)
        self.learning_rate = training.get("learning_rate", 0.001)
        self.epochs = training.get("epochs", 10)
        self.optimizer = training.get("optimizer", "adam")
        self.loss_function = training.get("loss_function", "cross_entropy")
        
        # Evaluation settings
        evaluation = self.config.get("evaluation", {})
        self.save_results = evaluation.get("save_results", True)
        self.results_path = evaluation.get("results_path", "results/")
    
        
        # Open-source model settings
        open_source = self.config.get("open_source", {})
        self.tokenizer_max_length = open_source.get("tokenizer_max_length", 128)
        self.max_seq_length = open_source.get("max_seq_length", 128)
        self.model_save_format = open_source.get("model_save_format", "pytorch")

        
        # Bi-Encoder settings
        bi_encoder = self.config.get("bi_encoder", {})
        self.embedding_dimension = bi_encoder.get("embedding_dimension", 768)
        self.use_faiss = bi_encoder.get("use_faiss", True)
        self.faiss_index_path = bi_encoder.get("faiss_index_path", "faiss/index")
        
        # Cross-Encoder settings
        cross_encoder = self.config.get("cross_encoder", {})
        self.pool_strategy = cross_encoder.get("pool_strategy", "mean")
        self.use_gpu = cross_encoder.get("use_gpu", True)
        
        # Miscellaneous settings
        misc = self.config.get("misc", {})
        self.random_seed = misc.get("random_seed", 42)
        self.verbose = misc.get("verbose", True)

    def __repr__(self):
        return f"Config(project_name={self.project_name}, model_type={self.model_type}, epochs={self.epochs})"

