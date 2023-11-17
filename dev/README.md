# Development

## Create a development environment

```bash
conda env create --file dev/environment-dev.yaml --name irag-dev
conda run --name irag-dev pip install -e .
```

This will create an environment with the name `irag-dev`, that can be activated with `conda activate irag-dev`.
