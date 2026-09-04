# V1.12 unreachable Pod

RunPod allocated Pod `g24y8c67qpqp1o` with two B300 GPUs at $15.78/hour. The
pinned image downloaded and the CUDA container started, but TCP port 22 never
opened. Container logs showed no model work.

The cause was the launcher's `--docker-args "sleep infinity"` option. It
replaced the image's normal startup command, so the image did not start SSH.
The infrastructure README already warned that publishing port 22 does not
start `sshd`.

The Pod was deleted before repository bootstrap, checkpoint download, model
load, model forward, activation extraction, or scientific-row generation.
The RunPod balance after deletion was $81.3784905484. V1.13 removes the Docker
argument and permits one same-topology retry.
