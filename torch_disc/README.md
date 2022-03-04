# Torch-DISC

## How to build

1. Development Docker image

```bash
nvidia-docker run --rm -it -
```

1. install requirements

``` bash
bash prepare.sh
```

1. build pytorch and torch-ltc

``` bash
(cd pytorch && python setup.py install)
(cd pytorch/lazy_tensor_core && python setup.py develop)
(cd setup.py develop)
```

1. Try to load `torch_disc` module

``` bash
$ python
> import torch_disc
```
