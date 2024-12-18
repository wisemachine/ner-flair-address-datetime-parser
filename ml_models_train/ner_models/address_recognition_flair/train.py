import os
import boto3
import torch
from dotenv import load_dotenv
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

load_dotenv()  # Loads environment variables from .env (if present)

def setup_environment(input_dir, model_dir):
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

def download_files_from_s3(s3_bucket, s3_prefix, input_dir):
    """
    Downloads train/test/dev data files from the specified S3 bucket and prefix 
    into the local 'input_dir' directory.
    Expects 'train_new.txt', 'test_new.txt', and 'dev_new.txt' to exist under s3_prefix.
    """
    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, f'{s3_prefix}/train_new.txt', os.path.join(input_dir, 'train_new.txt'))
    s3.download_file(s3_bucket, f'{s3_prefix}/test_new.txt', os.path.join(input_dir, 'test_new.txt'))
    s3.download_file(s3_bucket, f'{s3_prefix}/dev_new.txt', os.path.join(input_dir, 'dev_new.txt'))
    print(f"Files in input directory: {os.listdir(input_dir)}")

def initialize_corpus(data_folder, columns):
    """
    Creates a flair Corpus object from train/test/dev files (column format).
    """
    return ColumnCorpus(
        data_folder,
        columns,
        train_file='train_new.txt',
        test_file='test_new.txt',
        dev_file='dev_new.txt'
    )

def initialize_embeddings():
    flair_news_forward_embedding = FlairEmbeddings('news-forward')
    flair_news_backward_embedding = FlairEmbeddings('news-backward')
    return StackedEmbeddings([
        flair_news_forward_embedding,
        flair_news_backward_embedding
    ])

def initialize_tagger(embeddings, tag_dictionary, tag_type, device):
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
        dropout=0.1,
        word_dropout=0.1,
        locked_dropout=0.0
    )
    tagger.to(device)
    return tagger

def train_model(tagger, corpus, model_dir):
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(
        model_dir,
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=50
    )

def main():
    # Directories for input data and model output
    input_dir = '/opt/ml/input/data/training'
    model_dir = '/opt/ml/model'
    setup_environment(input_dir, model_dir)

    # Fetch S3 config from environment variables
    # Make sure your .env has S3_BUCKET and S3_PREFIX set
    s3_bucket = os.getenv('S3_BUCKET', 'your-default-bucket')
    s3_prefix = os.getenv('S3_PREFIX', 'your-default-prefix')

    # Download data files from S3
    download_files_from_s3(s3_bucket, s3_prefix, input_dir)

    # Define columns for the ColumnCorpus (0: token text, 1: NER tag)
    columns = {0: 'text', 1: 'ner'}
    data_folder = input_dir
    corpus = initialize_corpus(data_folder, columns)
    print(f"Number of training sentences: {len(corpus.train)}")

    tag_type = 'ner'
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
    print(f"Tag Dictionary: {tag_dictionary}")

    embeddings = initialize_embeddings()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tagger = initialize_tagger(embeddings, tag_dictionary, tag_type, device)
    train_model(tagger, corpus, model_dir)

if __name__ == "__main__":
    main()
