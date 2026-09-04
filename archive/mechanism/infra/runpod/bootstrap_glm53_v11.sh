#!/usr/bin/env bash
set -euo pipefail

export V11_ROOT="/workspace/mats-glm53-v11"
export PROJECT_ROOT="${V11_ROOT}/repo/Non-verbal-Eval-Awareness"
export V11_ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts/glm53_user_eval/v11"
export UV_CACHE_DIR="/tmp/glm53-v11-uv-cache"
export XDG_CACHE_HOME="/tmp/glm53-v11-xdg-cache"
export TORCH_EXTENSIONS_DIR="/tmp/glm53-v11-torch-extensions"
export TMPDIR="/tmp/glm53-v11-tmp"
export UV_PROJECT_ENVIRONMENT="/tmp/glm53-v11-venv"
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CACHE="/tmp/glm53-v11-hf-xet-cache"
export GLM53_PROJECT_REPO="${GLM53_PROJECT_REPO:-https://github.com/lmmontoya-ai/Non-verbal-Eval-Awareness.git}"
export GLM53_PROJECT_REF="glm53-user-eval-v11-preregistered"
export GLM53_MODEL_REVISION="04c4e9e95c5da8862dced7e5056455116f83a7e0"
export GLM53_MODEL_PATH="/workspace/mats-glm53/models/GLM-5.3-Flash/${GLM53_MODEL_REVISION}"
export GLM53_LOCAL_MODEL_PATH="/runpod-local/GLM-5.3-Flash/${GLM53_MODEL_REVISION}"
export GLM53_V11_S3_ENDPOINT="https://s3api-us-ks-2.runpod.io"
export GLM53_V11_S3_BUCKET="a9diryunoj"
export GLM53_V11_S3_PREFIX="glm53-v11-results"
export AWS_DEFAULT_REGION="US-KS-2"

for name in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY \
  RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC GLM53_V11_RUN_ID \
  GLM53_V11_DEADLINE_UTC GLM53_V11_AGGREGATE_RATE_USD \
  GLM53_V11_BALANCE_FLOOR_USD GLM53_V11_INPUT_PREFIX RUNPOD_POD_ID \
  RUNPOD_API_KEY; do
  if [[ -z "${!name:-}" ]]; then
    printf 'Required transient environment variable is absent: %s\n' "$name" >&2
    exit 1
  fi
done

# Keep credentials out of apt, git, uv, pip, model staging, and science child
# environments. Only the narrow S3, HMAC, and Pod-delete calls below receive
# them explicitly.
s3_access_key="${AWS_ACCESS_KEY_ID}"
s3_secret_key="${AWS_SECRET_ACCESS_KEY}"
pod_api_key="${RUNPOD_API_KEY}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY RUNPOD_API_KEY
readonly s3_access_key s3_secret_key pod_api_key

python3 - <<'PY'
from datetime import datetime, timezone
import os

attested = datetime.fromisoformat(
    os.environ["RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC"].replace("Z", "+00:00")
)
if attested.tzinfo is None:
    raise SystemExit("S3 credential session attestation must include a timezone")
age = datetime.now(timezone.utc) - attested.astimezone(timezone.utc)
if age.total_seconds() < -300 or age.total_seconds() > 86400:
    raise SystemExit("S3 credential session attestation is outside the 24-hour window")
PY

python3 - <<'PY'
from datetime import datetime, timezone
from decimal import Decimal
import os

deadline = datetime.fromisoformat(
    os.environ["GLM53_V11_DEADLINE_UTC"].replace("Z", "+00:00")
)
if deadline.tzinfo is None:
    raise SystemExit("v11 deadline must include a timezone")
remaining = deadline.astimezone(timezone.utc) - datetime.now(timezone.utc)
if remaining.total_seconds() <= 0 or remaining.total_seconds() > 6600:
    raise SystemExit("v11 paid deadline is absent, expired, or exceeds 110 minutes")
rate = Decimal(os.environ["GLM53_V11_AGGREGATE_RATE_USD"])
floor = Decimal(os.environ["GLM53_V11_BALANCE_FLOOR_USD"])
if rate <= 0 or rate > Decimal("15.78"):
    raise SystemExit("v11 live aggregate rate exceeds the preregistered cap")
if floor < Decimal("15.00"):
    raise SystemExit("v11 balance floor does not preserve the minimum reserve")
PY

command -v curl >/dev/null
credential_probe="${GLM53_V11_S3_ENDPOINT}/${GLM53_V11_S3_BUCKET}?list-type=2&max-keys=0"
curl --config - <<EOF
silent
show-error
fail
connect-timeout = 20
max-time = 120
retry = 3
retry-all-errors
user = "${s3_access_key}:${s3_secret_key}"
aws-sigv4 = "aws:amz:${AWS_DEFAULT_REGION}:s3"
output = "/dev/null"
url = "${credential_probe}"
EOF

heartbeat_pid=""
deadline_delete_pid=""
deadline_backup_pid=""
terminal_written=0
s3_operation_max_seconds=900
main_pid="$$"
on_exit_marker="/tmp/glm53-v11-on-exit-active"
declare -A uploaded_artifact_sha256=()

s3_put() {
  local source="$1"
  local key="$2"
  curl --config - <<EOF
silent
show-error
fail
connect-timeout = 20
max-time = ${s3_operation_max_seconds}
retry = 3
retry-all-errors
user = "${s3_access_key}:${s3_secret_key}"
aws-sigv4 = "aws:amz:${AWS_DEFAULT_REGION}:s3"
upload-file = "${source}"
url = "${GLM53_V11_S3_ENDPOINT}/${GLM53_V11_S3_BUCKET}/${key}"
EOF
}

s3_get() {
  local key="$1"
  local destination="$2"
  curl --config - <<EOF
silent
show-error
fail
connect-timeout = 20
max-time = ${s3_operation_max_seconds}
retry = 3
retry-all-errors
user = "${s3_access_key}:${s3_secret_key}"
aws-sigv4 = "aws:amz:${AWS_DEFAULT_REGION}:s3"
output = "${destination}"
url = "${GLM53_V11_S3_ENDPOINT}/${GLM53_V11_S3_BUCKET}/${key}"
EOF
}

write_heartbeat() {
  local phase="$1"
  local temporary="/tmp/glm53-v11-heartbeat-${BASHPID}.json"
  python3 - "${temporary}" "${phase}" <<'PY'
from datetime import datetime, timezone
import json
import os
import sys

payload = {
    "schema_version": "glm53_v11_heartbeat_v1",
    "run_id": os.environ["GLM53_V11_RUN_ID"],
    "pod_id": os.environ["RUNPOD_POD_ID"],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "phase": sys.argv[2],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
  s3_put "${temporary}" \
    "${GLM53_V11_S3_PREFIX}/${GLM53_V11_RUN_ID}/heartbeat.json"
  rm -f -- "${temporary}"
}

start_heartbeat() {
  local phase="$1"
  (
    while true; do
      write_heartbeat "${phase}" || exit 1
      sleep 60
    done
  ) &
  heartbeat_pid="$!"
}

stop_heartbeat() {
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
    heartbeat_pid=""
  fi
}

upload_terminal() {
  local exit_code="$1"
  local state="$2"
  local temporary="/tmp/glm53-v11-terminal-${BASHPID}.json"
  python3 - "${temporary}" "${exit_code}" "${state}" <<'PY'
from datetime import datetime, timezone
import json
import os
import sys

payload = {
    "schema_version": "glm53_v11_terminal_v1",
    "run_id": os.environ["GLM53_V11_RUN_ID"],
    "pod_id": os.environ["RUNPOD_POD_ID"],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "exit_code": int(sys.argv[2]),
    "state": sys.argv[3],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
  s3_put "${temporary}" \
    "${GLM53_V11_S3_PREFIX}/${GLM53_V11_RUN_ID}/terminal.json"
  rm -f -- "${temporary}"
  terminal_written=1
}

upload_current_artifacts() {
  if [[ ! -d "${V11_ARTIFACT_ROOT}" ]]; then
    return 0
  fi
  while IFS= read -r -d '' artifact; do
    local relative="${artifact#${V11_ARTIFACT_ROOT}/}"
    local digest
    digest="$(sha256sum "${artifact}" | cut -d' ' -f1)"
    if [[ "${uploaded_artifact_sha256[${relative}]:-}" == "${digest}" ]]; then
      continue
    fi
    s3_put "${artifact}" \
      "${GLM53_V11_S3_PREFIX}/${GLM53_V11_RUN_ID}/artifacts/${relative}" || return 1
    uploaded_artifact_sha256["${relative}"]="${digest}"
  done < <(find "${V11_ARTIFACT_ROOT}" -type f ! -name '*.tmp' ! -name '*.partial' -print0)
}

delete_current_pod() {
  local http_code
  # RunPod injects a Pod-scoped RUNPOD_API_KEY. Feed the authorization header
  # through curl stdin so the key never appears in argv, logs, or artifacts.
  http_code="$(curl --config - <<EOF
request = "DELETE"
url = "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}"
header = "Authorization: Bearer ${pod_api_key}"
silent
show-error
connect-timeout = 10
max-time = 30
output = "/dev/null"
write-out = "%{http_code}"
EOF
)" || return 1
  [[ "${http_code}" == "204" || "${http_code}" == "404" ]]
}

write_delete_failure_marker() {
  local reason="$1"
  local marker="${V11_ARTIFACT_ROOT}/infrastructure/pod_delete_failure.json"
  mkdir -p -- "$(dirname "${marker}")"
  python3 - "${marker}" "${reason}" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

target = Path(sys.argv[1])
temporary = target.with_suffix(target.suffix + ".partial")
temporary.write_text(json.dumps({
    "schema_version": "glm53_v11_pod_delete_failure_v1",
    "run_id": os.environ["GLM53_V11_RUN_ID"],
    "pod_id": os.environ["RUNPOD_POD_ID"],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "reason": sys.argv[2],
}, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
  s3_put "${marker}" \
    "${GLM53_V11_S3_PREFIX}/${GLM53_V11_RUN_ID}/infrastructure/pod_delete_failure.json" || true
}

start_deadline_guards() {
  local seconds_remaining
  seconds_remaining="$(python3 - <<'PY'
from datetime import datetime, timezone
import os

deadline = datetime.fromisoformat(
    os.environ["GLM53_V11_DEADLINE_UTC"].replace("Z", "+00:00")
).astimezone(timezone.utc)
print(max(1, int((deadline - datetime.now(timezone.utc)).total_seconds())))
PY
)"
  (
    sleep "$(( seconds_remaining > 30 ? seconds_remaining - 30 : 1 ))"
    s3_operation_max_seconds=20
    upload_current_artifacts || true
    upload_terminal 124 "hard_deadline_imminent" || true
  ) &
  deadline_backup_pid="$!"
  (
    sleep "${seconds_remaining}"
    if [[ ! -e "${on_exit_marker}" ]]; then
      kill -TERM "${main_pid}" 2>/dev/null || true
    fi
    if ! delete_current_pod_with_retry 120; then
      s3_operation_max_seconds=20
      write_delete_failure_marker "pod_scoped_deadline_delete_unconfirmed"
    fi
  ) &
  deadline_delete_pid="$!"
}

stop_deadline_guards() {
  local pid
  for pid in "${deadline_backup_pid}" "${deadline_delete_pid}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  deadline_backup_pid=""
  deadline_delete_pid=""
}

delete_current_pod_with_retry() {
  local retry_seconds="${1:-120}"
  local stop_at=$(( SECONDS + retry_seconds ))
  while (( SECONDS < stop_at )); do
    if delete_current_pod; then
      return 0
    fi
    sleep 5
  done
  return 1
}

on_exit() {
  local exit_code="$?"
  : > "${on_exit_marker}"
  s3_operation_max_seconds=60
  if [[ "${terminal_written}" -eq 0 ]]; then
    upload_current_artifacts || true
    stop_heartbeat
    upload_terminal "${exit_code}" "bootstrap_or_science_exited" || true
  else
    stop_heartbeat
  fi
  if delete_current_pod_with_retry 120; then
    stop_deadline_guards
    return
  fi
  write_delete_failure_marker "normal_terminal_delete_unconfirmed"
  if [[ -n "${deadline_delete_pid}" ]] && kill -0 "${deadline_delete_pid}" 2>/dev/null; then
    wait "${deadline_delete_pid}" 2>/dev/null || true
  fi
  if delete_current_pod_with_retry 120; then
    stop_deadline_guards
    return
  fi
  write_delete_failure_marker "post_deadline_delete_unconfirmed_container_exiting"
  stop_deadline_guards
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_heartbeat "bootstrap"
start_deadline_guards

mkdir -p "${V11_ROOT}/repo" "${V11_ROOT}/exports" \
  "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TORCH_EXTENSIONS_DIR}" \
  "${TMPDIR}" "${HF_XET_CACHE}" "$(dirname "${GLM53_LOCAL_MODEL_PATH}")"

apt-get update
apt-get install -y git git-lfs jq rsync tmux htop nvtop zstd curl
curl -LsSf --connect-timeout 20 --max-time 120 --retry 3 --retry-all-errors \
  https://astral.sh/uv/0.9.26/install.sh | \
  env UV_UNMANAGED_INSTALL="${HOME}/.local/bin" sh
export PATH="${HOME}/.local/bin:${PATH}"
test "$(uv --version)" = "uv 0.9.26 (ee4f00362 2026-01-15)"

if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  git clone "${GLM53_PROJECT_REPO}" "${PROJECT_ROOT}"
fi
git -C "${PROJECT_ROOT}" fetch --tags origin
git -C "${PROJECT_ROOT}" checkout --detach "${GLM53_PROJECT_REF}"
test "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" = \
  "$(git -C "${PROJECT_ROOT}" rev-list -n 1 "${GLM53_PROJECT_REF}")"
test -z "$(git -C "${PROJECT_ROOT}" status --porcelain)"

input_bundle="/tmp/glm53-v11-input-bundle"
rm -rf -- "${input_bundle}"
mkdir -p "${input_bundle}/objects"
s3_get "${GLM53_V11_INPUT_PREFIX}/input_manifest.json" \
  "${input_bundle}/input_manifest.json"
s3_get "${GLM53_V11_INPUT_PREFIX}/input_manifest.hmac-sha256" \
  "${input_bundle}/input_manifest.hmac-sha256"
python3 - "${input_bundle}" "${PROJECT_ROOT}" 3<<<"${s3_secret_key}" <<'PY'
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys

bundle = Path(sys.argv[1])
project = Path(sys.argv[2]).resolve()
manifest_bytes = (bundle / "input_manifest.json").read_bytes()
expected_signature = (bundle / "input_manifest.hmac-sha256").read_text().strip()
actual_signature = hmac.new(
    os.read(3, 1024 * 1024).rstrip(b"\n"),
    b"glm53-v11-input-manifest-v1\0" + manifest_bytes,
    hashlib.sha256,
).hexdigest()
if not hmac.compare_digest(actual_signature, expected_signature):
    raise SystemExit("v11 S3 input-manifest HMAC verification failed")
manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
if manifest.get("schema_version") != "glm53_v11_signed_input_manifest_v1":
    raise SystemExit("unexpected v11 S3 input-manifest schema")
if manifest.get("run_id") != os.environ["GLM53_V11_RUN_ID"]:
    raise SystemExit("v11 S3 input manifest names another run")
if manifest.get("object_prefix") != os.environ["GLM53_V11_INPUT_PREFIX"]:
    raise SystemExit("v11 S3 input manifest names another object prefix")
if manifest.get("prereg_tag") != "glm53-user-eval-v11-preregistered":
    raise SystemExit("v11 S3 input manifest names another preregistration tag")
expected_commit = subprocess.check_output(
    ["git", "-C", str(project), "rev-list", "-n", "1", "glm53-user-eval-v11-preregistered"],
    text=True,
).strip()
if manifest.get("git_commit") != expected_commit:
    raise SystemExit("v11 S3 input manifest commit differs from the preregistration tag")
expected_targets = {
    "text_decision.json": "artifacts/glm53_user_eval/v11/offline_audit/decision.json",
    "samples.jsonl": "artifacts/datasets/contrastive_prompts_v3/samples.jsonl",
    "dataset_manifest.json": "artifacts/datasets/contrastive_prompts_v3/manifest.json",
    "tokenizer_audit.json": "artifacts/datasets/contrastive_prompts_v3/tokenizer_audit.json",
    "structural_audit.json": "artifacts/glm53_user_eval/v11/offline_audit/structural_audit.json",
    "development_analysis.json": "artifacts/glm53_user_eval/v11/offline_audit/development_analysis.json",
    "final_text_analysis.json": "artifacts/glm53_user_eval/v11/offline_audit/final_text_analysis.json",
    "FINAL_TEXT_HOLDOUT_OPENED.json": "artifacts/glm53_user_eval/v11/offline_audit/FINAL_TEXT_HOLDOUT_OPENED.json",
    "lexical_decision.json": "artifacts/glm53_user_eval/v11/offline_audit/lexical_decision.json",
    "semantic_validation.json": "artifacts/glm53_user_eval/v11/offline_audit/semantic_validation.json",
    "manual_audit.json": "artifacts/glm53_user_eval/v11/offline_audit/manual_audit.json",
    "offline_verification.json": "artifacts/glm53_user_eval/v11/offline_audit/verification.json",
    "downstream_manifest_v1.json": "pipelines/glm53_user_eval/v11/configs/downstream_manifest_v1.json",
    "downstream_preflight.json": "artifacts/glm53_user_eval/v11/downstream_inputs/preflight.json",
    "v7_transcripts_all100.jsonl": "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100.jsonl",
    "v7_transcripts_all100_manifest.json": "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100_manifest.json",
    "parent_proxy_surface_v1.json": "pipelines/glm53_user_eval/v11/configs/parent_proxy_surface_v1.json",
    "proxy_token_contract_v2.json": "pipelines/glm53_user_eval/v11/configs/proxy_token_contract_v2.json",
    "proxy_codebooks_v2.json": "pipelines/glm53_user_eval/v11/configs/proxy_codebooks_v2.json",
    "causal_schedule_v1.json": "pipelines/glm53_user_eval/v8/configs/causal_schedule_v1.json",
    "user_prompt_templates_v1.jsonl": "pipelines/glm53_user_eval/v8/configs/user_prompt_templates_v1.jsonl",
    "identity_selection_v1.json": "pipelines/glm53_user_eval/configs/identity_selection_v1.json",
    "v7_analysis.json": "artifacts/glm53_user_eval/reports/transluce_interaction_v7/analysis.json",
    "v7_decision.json": "artifacts/glm53_user_eval/reports/transluce_interaction_v7/decision.json",
    "v7_final_evidence.json": "artifacts/glm53_user_eval/reports/transluce_interaction_v7/final_evidence.json",
    "transluce_personas2.json": "artifacts/glm53_user_eval/v11/downstream_inputs/personas2.json",
}
files = manifest.get("files")
if not isinstance(files, dict) or set(files) != set(expected_targets):
    raise SystemExit("v11 S3 input manifest has the wrong file set")
for name, target_relative in expected_targets.items():
    record = files[name]
    if record.get("object_name") != name:
        raise SystemExit(f"v11 S3 object name mismatch: {name}")
    if record.get("target_relative_path") != target_relative:
        raise SystemExit(f"v11 S3 target path mismatch: {name}")
    digest = record.get("sha256", "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SystemExit(f"v11 S3 input lacks literal SHA-256: {name}")
    target = (project / target_relative).resolve()
    if project not in target.parents:
        raise SystemExit(f"v11 S3 target escapes project root: {name}")
(bundle / "verified_manifest.json").write_text(
    json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
)
PY

while IFS= read -r object_name; do
  s3_get "${GLM53_V11_INPUT_PREFIX}/${object_name}" \
    "${input_bundle}/objects/${object_name}.partial"
done < <(jq -r '.files | keys[]' "${input_bundle}/verified_manifest.json")
python3 - "${input_bundle}" "${PROJECT_ROOT}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

bundle = Path(sys.argv[1])
project = Path(sys.argv[2]).resolve()
manifest = json.loads((bundle / "verified_manifest.json").read_text())
for name, record in manifest["files"].items():
    partial = bundle / "objects" / f"{name}.partial"
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise SystemExit(f"v11 S3 input hash mismatch: {name}")
    if partial.stat().st_size != int(record["bytes"]):
        raise SystemExit(f"v11 S3 input size mismatch: {name}")
    target = (project / record["target_relative_path"]).resolve()
    if project not in target.parents:
        raise SystemExit(f"v11 S3 target escapes project root: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_bytes(partial.read_bytes())
    os.replace(temporary, target)
PY

cd "${PROJECT_ROOT}"
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

decision = json.loads(
    Path("artifacts/glm53_user_eval/v11/offline_audit/decision.json").read_text()
)
input_manifest = json.loads(
    Path("/tmp/glm53-v11-input-bundle/verified_manifest.json").read_text()
)
if decision.get("passed") is not True:
    raise SystemExit("v11 text gate did not pass")
if decision.get("decision") != "source_text_instrument_valid_for_activation_test":
    raise SystemExit("v11 text decision does not unlock source extraction")
if decision.get("schema_version") != "glm53_v11_text_gate_decision_v1" or decision.get(
    "project_id"
) != "glm53_user_eval_source_instrument_v11":
    raise SystemExit("v11 text decision schema or project differs")
expected_check_keys = {
    "structural",
    "tokenizer",
    "metadata_ordinary_le_060",
    "metadata_final_le_060",
    "keyword_final_le_060",
    "word_final_le_065",
    "char_final_le_065",
    "deleted_word_final_le_060",
    "deleted_char_final_le_060",
    "lexical_decision",
    "semantic",
    "manual",
    "independent_verification",
}
if set(decision.get("checks", {})) != expected_check_keys or not all(
    value is True for value in decision["checks"].values()
):
    raise SystemExit("v11 text decision contains a failed or malformed check")
for field in ("new_glm_forwards", "runpod_compute", "source_activation_extraction"):
    if decision.get("authorization", {}).get(field) is not True:
        raise SystemExit(f"v11 text decision does not authorize {field}")
expected_decision_inputs = {
    "prereg",
    "samples",
    "dataset_manifest",
    "tokenizer_audit",
    "builder",
    "spec",
    "runtime_config",
    "structural",
    "development",
    "final_text",
    "final_text_marker",
    "lexical_decision",
    "semantic",
    "manual",
    "verification",
}
decision_inputs = decision.get("inputs", {})
if set(decision_inputs) != expected_decision_inputs or not all(
    isinstance(value, str) and len(value) == 64 for value in decision_inputs.values()
):
    raise SystemExit("v11 text decision input-hash set differs from the paid contract")
if input_manifest.get("decision_inputs") != decision_inputs:
    raise SystemExit("v11 signed input manifest differs from the text decision inputs")
paths = {
    "prereg": Path("pipelines/glm53_user_eval/v11/configs/prereg_v11_source_instrument.yaml"),
    "samples": Path("artifacts/datasets/contrastive_prompts_v3/samples.jsonl"),
    "dataset_manifest": Path("artifacts/datasets/contrastive_prompts_v3/manifest.json"),
    "tokenizer_audit": Path("artifacts/datasets/contrastive_prompts_v3/tokenizer_audit.json"),
    "builder": Path("src/glm53_user_eval/v11/builder.py"),
    "spec": Path("src/glm53_user_eval/v11/spec.py"),
    "runtime_config": Path("pipelines/glm53_user_eval/v11/configs/runtime_v11.yaml"),
    "structural": Path("artifacts/glm53_user_eval/v11/offline_audit/structural_audit.json"),
    "development": Path("artifacts/glm53_user_eval/v11/offline_audit/development_analysis.json"),
    "final_text": Path("artifacts/glm53_user_eval/v11/offline_audit/final_text_analysis.json"),
    "final_text_marker": Path("artifacts/glm53_user_eval/v11/offline_audit/FINAL_TEXT_HOLDOUT_OPENED.json"),
    "lexical_decision": Path("artifacts/glm53_user_eval/v11/offline_audit/lexical_decision.json"),
    "semantic": Path("artifacts/glm53_user_eval/v11/offline_audit/semantic_validation.json"),
    "manual": Path("artifacts/glm53_user_eval/v11/offline_audit/manual_audit.json"),
    "verification": Path("artifacts/glm53_user_eval/v11/offline_audit/verification.json"),
}
for field, path in paths.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != decision["inputs"][field]:
        raise SystemExit(f"v11 reconstructed input differs from decision hash: {field}")
PY

if [[ -e "${UV_PROJECT_ENVIRONMENT}" ]]; then
  printf 'Refusing to reuse an existing v11 virtual environment\n' >&2
  exit 1
fi
uv sync --frozen
uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
  torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip uninstall --python "${UV_PROJECT_ENVIRONMENT}/bin/python" torchaudio || true
uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" --reinstall \
  -r infra/runpod/glm53_runtime_overlay_v2.requirements.txt

"${UV_PROJECT_ENVIRONMENT}/bin/python" - <<'PY'
import importlib.metadata
import json
import torch

distribution = importlib.metadata.distribution("transformers")
direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
commit = direct_url.get("vcs_info", {}).get("commit_id")
expected_commit = "805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
if commit != expected_commit:
    raise SystemExit(f"transformers commit {commit!r} differs from {expected_commit}")
if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
    raise SystemExit(f"unexpected torch runtime: {torch.__version__}, {torch.version.cuda}")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"expected two GPUs, found {torch.cuda.device_count()}")
gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
if any("B300" not in name for name in gpu_names):
    raise SystemExit(f"expected only B300 GPUs, found {gpu_names}")
print(json.dumps({
    "transformers_commit": commit,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu_count": len(gpu_names),
    "gpu_names": gpu_names,
}, indent=2))
PY

available_bytes="$(df --output=avail -B1 /runpod-local | tail -n 1 | tr -d ' ')"
if (( available_bytes < 360000000000 )); then
  printf 'Container disk has fewer than 360 GB free before model staging\n' >&2
  exit 1
fi

stage_report="${V11_ARTIFACT_ROOT}/infrastructure/model_stage.json"
"${UV_PROJECT_ENVIRONMENT}/bin/python" infra/runpod/stage_glm53_v8_hf_local.py \
  --manifest artifacts/glm53_user_eval/runtime/g2/model_stage.json \
  --target "${GLM53_LOCAL_MODEL_PATH}" \
  --runtime-link "${GLM53_MODEL_PATH}" \
  --report "${stage_report}" \
  --workers 8
s3_put "${stage_report}" \
  "${GLM53_V11_S3_PREFIX}/${GLM53_V11_RUN_ID}/infrastructure/model_stage.json"

if [[ "${GLM53_V11_RUN_SCIENCE:-0}" != "1" ]]; then
  upload_terminal 0 "bootstrap_only_complete"
  exit 0
fi

stop_heartbeat
start_heartbeat "source_instrument"
runner=("${UV_PROJECT_ENVIRONMENT}/bin/python" pipelines/glm53_user_eval/v11/run.py)
"${runner[@]}" validate-downstream
"${runner[@]}" paid-ladder --model-path "${GLM53_MODEL_PATH}" \
  --permutation-reps 1000 --confirm-spend &
ladder_pid="$!"
while kill -0 "${ladder_pid}" 2>/dev/null; do
  sleep 60
  upload_current_artifacts || true
done
wait "${ladder_pid}"

evidence_manifest="${V11_ARTIFACT_ROOT}/infrastructure/source_evidence_manifest.json"
"${UV_PROJECT_ENVIRONMENT}/bin/python" - "${V11_ARTIFACT_ROOT}" "${evidence_manifest}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
output = Path(sys.argv[2])
files = {}
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    if path == output or path.name.endswith((".tmp", ".partial")):
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    files[path.relative_to(root).as_posix()] = digest.hexdigest()
temporary = output.with_suffix(output.suffix + ".partial")
temporary.parent.mkdir(parents=True, exist_ok=True)
temporary.write_text(json.dumps({
    "schema_version": "glm53_v11_source_evidence_manifest_v1",
    "run_id": os.environ["GLM53_V11_RUN_ID"],
    "pod_id": os.environ["RUNPOD_POD_ID"],
    "files": files,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY

upload_current_artifacts

roundtrip="/tmp/glm53-v11-source-evidence-manifest.roundtrip.json"
s3_get \
  "${GLM53_V11_S3_PREFIX}/${GLM53_V11_RUN_ID}/artifacts/infrastructure/source_evidence_manifest.json" \
  "${roundtrip}"
test "$(sha256sum "${evidence_manifest}" | cut -d' ' -f1)" = \
  "$(sha256sum "${roundtrip}" | cut -d' ' -f1)"
rm -f -- "${roundtrip}"
upload_terminal 0 "source_instrument_terminal_decision_written"
