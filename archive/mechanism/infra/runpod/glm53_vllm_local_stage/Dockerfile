FROM vllm/vllm-openai@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703

COPY stage_and_serve.py /usr/local/bin/glm53-stage-and-serve
RUN chmod 0755 /usr/local/bin/glm53-stage-and-serve

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/usr/local/bin/glm53-stage-and-serve"]
