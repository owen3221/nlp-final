## Install Conda

參考 https://vocus.cc/article/63c16c72fd89780001276284

## Create Virtual Environment via conda

```shell
curl -sSL https://install.python-poetry.org | python3 -
conda create -n nlp-final python=3.10 -y
conda activate nlp-final
poetry install --no-root
```

## Add .env file

- Make a copy of `env_template` and rename it to `.env` and then fill you keys accordingly.
  - MISTRAL_API_KEY: api key on mistralai
  - NV_API_KEY: Skip this, we are not going to use this.
  - API_KEY_1, API_KEY_2, ...: the API key from google ai studio, apply as much as possible. And then go to `gen.py` to set `KEY_NUMS` as the number of keys you've applied.

## Run a quick test to see whether the environment is ready

```shell
python test/test.py --model google
```

## Run pride_method

```shell
python src/exp.py --exp taskc --model google --prompt_method pride
```
