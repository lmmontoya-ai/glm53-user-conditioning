FROM runpod/pytorch@sha256:f40e33a190d6823439541d1dde52003fbed66539a7af998f38e29f499ca5bdd6

ARG PROJECT_REF=glm53-user-eval-v8-preregistered-v1.7
ARG PROJECT_REPO=https://github.com/lmmontoya-ai/Non-verbal-Eval-Awareness.git
ARG TRANSLUCE_COMMIT=d1b9c3573470f50495202795c044bd72f72ee6e5

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --ignore-installed cryptography==45.0.7 \
    && python -m pip install --no-cache-dir \
    runpod==1.7.13 \
    boto3 \
    accelerate==1.11.0 \
    safetensors \
    tokenizers \
    sentencepiece \
    huggingface-hub \
    numpy \
    scipy \
    scikit-learn \
    pyyaml \
    pydantic \
    typer \
    orjson \
    kernels==0.16.0 \
    "transformers @ git+https://github.com/huggingface/transformers.git@805a9e939fa8c1bff8d8ffdf041c051b71a914aa" \
    && python -m pip install --no-cache-dir --no-deps --force-reinstall triton==3.5.1 \
    && python -c 'import torch; assert torch.__version__ == "2.8.0+cu128", torch.__version__'

RUN git clone "$PROJECT_REPO" /app/repo \
    && git -C /app/repo checkout --detach "$PROJECT_REF" \
    && test "$(git -C /app/repo rev-parse HEAD)" = "$(git -C /app/repo rev-list -n 1 "$PROJECT_REF")" \
    && test -z "$(git -C /app/repo status --porcelain)" \
    && git clone https://github.com/TransluceAI/user-awareness /app/reference/transluce-user-awareness \
    && git -C /app/reference/transluce-user-awareness checkout --detach "$TRANSLUCE_COMMIT"

COPY infra/runpod/glm53_v8_serverless/handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/repo \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app/repo
CMD ["python", "-u", "/app/handler.py"]
