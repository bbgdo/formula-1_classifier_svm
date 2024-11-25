# Formula-1 SMV classifier

## Dataset
[F1 Image Classification Updated](https://www.kaggle.com/datasets/loveymishra/f1-image-classification-updated)

Only Mercedes and Redbull images were used.

## Usage
```bash
pip install -r requirements.txt
```

### Prepare data for training
```bash
python prepare_data.py
```

### Train model
```bash
python train_svm_classifier_f1.py
```

### Use classifier
```bash
python classifier.py
```

