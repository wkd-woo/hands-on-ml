```
docker run -it --rm -p 8500:
8500 -p 8501:8501 -v "C:/workspace/HKNU/machine_learning/my_mnist_model:/models/my
_mnist_model" -e MODEL_NAME=my_mnist_model tensorflow/serving
```
