#!/usr/bin/env bash
set -euo pipefail

export V19_ROOT="/workspace/mats-glm53-v19"
export PROJECT_ROOT="${V19_ROOT}/repo/Non-verbal-Eval-Awareness"
export V19_ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts/glm53_user_eval/v19"
export UV_CACHE_DIR="/tmp/glm53-v19-uv-cache"
export XDG_CACHE_HOME="/tmp/glm53-v19-xdg-cache"
export TORCH_EXTENSIONS_DIR="/tmp/glm53-v19-torch-extensions"
export TMPDIR="/tmp/glm53-v19-tmp"
export UV_PROJECT_ENVIRONMENT="/tmp/glm53-v19-venv"
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CACHE="/tmp/glm53-v19-hf-xet-cache"
export GLM53_PROJECT_REPO="${GLM53_PROJECT_REPO:-https://github.com/lmmontoya-ai/Non-verbal-Eval-Awareness.git}"
export GLM53_PROJECT_REF="glm53-user-eval-v19-preregistered-r8"
export GLM53_PROJECT_COMMIT="24f358173193c97103dc5d3fff3c0a14ac7c88b1"
export GLM53_SOURCE_ARCHIVE_SHA256="f60cc5b3b7ef4ae91a927445dd6730a5ec3bf1287f182951fdf550b8a97d105b"
export GLM53_SOURCE_ARCHIVE_VERIFIED=1
export GLM53_SOURCE_COMMIT="${GLM53_PROJECT_COMMIT}"
export GLM53_EXECUTION_TAG="${GLM53_PROJECT_REF}"
export GLM53_ARTIFACT_RELATIVE_ROOT="artifacts/glm53_user_eval/v19"
export GLM53_RATE_CAP_USD_PER_HOUR="16.00"
export GLM53_COMPUTE_CAP_USD="39.51"
export GLM53_MODEL_REVISION="04c4e9e95c5da8862dced7e5056455116f83a7e0"
export GLM53_MODEL_PATH="/workspace/mats-glm53/models/GLM-5.3-Flash/${GLM53_MODEL_REVISION}"
export GLM53_LOCAL_MODEL_PATH="/runpod-local/GLM-5.3-Flash/${GLM53_MODEL_REVISION}"
export GLM53_V19_S3_ENDPOINT="${GLM53_V19_S3_ENDPOINT:-https://s3api-us-ks-2.runpod.io}"
export GLM53_V19_S3_BUCKET="${GLM53_V19_S3_BUCKET:?missing temporary transport volume ID}"
export GLM53_V19_S3_PREFIX="glm53-v19-results"
export AWS_DEFAULT_REGION="US-KS-2"
BOOTSTRAP_LOG="/tmp/glm53-v19-bootstrap.log"
: > "${BOOTSTRAP_LOG}"
exec > >(tee -a "${BOOTSTRAP_LOG}") 2>&1

for name in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY \
  RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC GLM53_V19_RUN_ID \
  GLM53_V19_DEADLINE_UTC GLM53_V19_AGGREGATE_RATE_USD \
  GLM53_V19_BALANCE_FLOOR_USD GLM53_V19_LAUNCH_BALANCE_USD \
  GLM53_V19_INPUT_PREFIX RUNPOD_POD_ID \
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
    os.environ["GLM53_V19_DEADLINE_UTC"].replace("Z", "+00:00")
)
if deadline.tzinfo is None:
    raise SystemExit("v19 deadline must include a timezone")
remaining = deadline.astimezone(timezone.utc) - datetime.now(timezone.utc)
if remaining.total_seconds() <= 0 or remaining.total_seconds() > 9000:
    raise SystemExit("v19 paid deadline is absent, expired, or exceeds 150 minutes")
rate = Decimal(os.environ["GLM53_V19_AGGREGATE_RATE_USD"])
floor = Decimal(os.environ["GLM53_V19_BALANCE_FLOOR_USD"])
if rate <= 0 or rate > Decimal("16.00"):
    raise SystemExit("v19 live aggregate rate exceeds the preregistered cap")
if floor < Decimal("32.00"):
    raise SystemExit("v19 balance floor does not preserve the minimum reserve")
PY

command -v curl >/dev/null
credential_probe="${GLM53_V19_S3_ENDPOINT}/${GLM53_V19_S3_BUCKET}?list-type=2&max-keys=0"
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
export GLM53_V19_S3_TRANSPORT_VERIFIED=1

heartbeat_pid=""
deadline_delete_pid=""
deadline_backup_pid=""
terminal_written=0
s3_operation_max_seconds=900
main_pid="$$"
on_exit_marker="/tmp/glm53-v19-on-exit-active"
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
url = "${GLM53_V19_S3_ENDPOINT}/${GLM53_V19_S3_BUCKET}/${key}"
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
url = "${GLM53_V19_S3_ENDPOINT}/${GLM53_V19_S3_BUCKET}/${key}"
EOF
}

write_heartbeat() {
  local phase="$1"
  local temporary="/tmp/glm53-v19-heartbeat-${BASHPID}.json"
  python3 - "${temporary}" "${phase}" <<'PY'
from datetime import datetime, timezone
import json
import os
import sys

payload = {
    "schema_version": "glm53_v19_heartbeat_v1",
    "run_id": os.environ["GLM53_V19_RUN_ID"],
    "pod_id": os.environ["RUNPOD_POD_ID"],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "phase": sys.argv[2],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
  s3_put "${temporary}" \
    "${GLM53_V19_S3_PREFIX}/${GLM53_V19_RUN_ID}/heartbeat.json"
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
  local temporary="/tmp/glm53-v19-terminal-${BASHPID}.json"
  python3 - "${temporary}" "${exit_code}" "${state}" <<'PY'
from datetime import datetime, timezone
import json
import os
import sys

payload = {
    "schema_version": "glm53_v19_terminal_v1",
    "run_id": os.environ["GLM53_V19_RUN_ID"],
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
    "${GLM53_V19_S3_PREFIX}/${GLM53_V19_RUN_ID}/terminal.json"
  rm -f -- "${temporary}"
  terminal_written=1
}

upload_current_artifacts() {
  if [[ ! -d "${V19_ARTIFACT_ROOT}" ]]; then
    return 0
  fi
  while IFS= read -r -d '' artifact; do
    local relative="${artifact#${V19_ARTIFACT_ROOT}/}"
    local digest
    digest="$(sha256sum "${artifact}" | cut -d' ' -f1)"
    if [[ "${uploaded_artifact_sha256[${relative}]:-}" == "${digest}" ]]; then
      continue
    fi
    s3_put "${artifact}" \
      "${GLM53_V19_S3_PREFIX}/${GLM53_V19_RUN_ID}/artifacts/${relative}" || return 1
    uploaded_artifact_sha256["${relative}"]="${digest}"
  done < <(find "${V19_ARTIFACT_ROOT}" -type f ! -name '*.tmp' ! -name '*.partial' -print0)
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
  local marker="${V19_ARTIFACT_ROOT}/infrastructure/pod_delete_failure.json"
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
    "schema_version": "glm53_v19_pod_delete_failure_v1",
    "run_id": os.environ["GLM53_V19_RUN_ID"],
    "pod_id": os.environ["RUNPOD_POD_ID"],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "reason": sys.argv[2],
}, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
  s3_put "${marker}" \
    "${GLM53_V19_S3_PREFIX}/${GLM53_V19_RUN_ID}/infrastructure/pod_delete_failure.json" || true
}

start_deadline_guards() {
  local seconds_remaining
  seconds_remaining="$(python3 - <<'PY'
from datetime import datetime, timezone
import os

deadline = datetime.fromisoformat(
    os.environ["GLM53_V19_DEADLINE_UTC"].replace("Z", "+00:00")
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
    s3_put "${BOOTSTRAP_LOG}" \
      "${GLM53_V19_S3_PREFIX}/${GLM53_V19_RUN_ID}/infrastructure/bootstrap.log" || true
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

mkdir -p "${V19_ROOT}/repo" "${V19_ROOT}/exports" \
  "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TORCH_EXTENSIONS_DIR}" \
  "${TMPDIR}" "${HF_XET_CACHE}" "$(dirname "${GLM53_LOCAL_MODEL_PATH}")"

apt-get update
apt-get install -y git git-lfs jq rsync tmux htop nvtop zstd curl
curl -LsSf --connect-timeout 20 --max-time 120 --retry 3 --retry-all-errors \
  https://astral.sh/uv/0.9.26/install.sh | \
  env UV_UNMANAGED_INSTALL="${HOME}/.local/bin" sh
export PATH="${HOME}/.local/bin:${PATH}"
uv --version | grep -Eq '^uv 0\.9\.26($| )'

if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  source_archive="/tmp/glm53-v19-science-repo.tar.gz"
  s3_get "${GLM53_V19_INPUT_PREFIX}/v19_science_repo.tar.gz" \
    "${source_archive}.partial"
  printf '%s  %s\n' "${GLM53_SOURCE_ARCHIVE_SHA256}" \
    "${source_archive}.partial" | sha256sum --check --strict -
  mv "${source_archive}.partial" "${source_archive}"
  mkdir -p "${PROJECT_ROOT}"
  tar -xzf "${source_archive}" -C "${PROJECT_ROOT}" --strip-components=1
fi

input_bundle="/tmp/glm53-v19-input-bundle"
rm -rf -- "${input_bundle}"
mkdir -p "${input_bundle}/objects"
s3_get "${GLM53_V19_INPUT_PREFIX}/input_manifest.json" \
  "${input_bundle}/input_manifest.json"
s3_get "${GLM53_V19_INPUT_PREFIX}/input_manifest.hmac-sha256" \
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
    b"glm53-v19-input-manifest-v1\0" + manifest_bytes,
    hashlib.sha256,
).hexdigest()
if not hmac.compare_digest(actual_signature, expected_signature):
    raise SystemExit("v19 S3 input-manifest HMAC verification failed")
manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
if manifest.get("schema_version") != "glm53_v19_signed_input_manifest_v1":
    raise SystemExit("unexpected v19 S3 input-manifest schema")
if manifest.get("run_id") != os.environ["GLM53_V19_RUN_ID"]:
    raise SystemExit("v19 S3 input manifest names another run")
if manifest.get("object_prefix") != os.environ["GLM53_V19_INPUT_PREFIX"]:
    raise SystemExit("v19 S3 input manifest names another object prefix")
if manifest.get("prereg_tag") != "glm53-user-eval-v19-preregistered-r8":
    raise SystemExit("v19 S3 input manifest names another preregistration tag")
expected_commit = "24f358173193c97103dc5d3fff3c0a14ac7c88b1"
if manifest.get("git_commit") != expected_commit:
    raise SystemExit("v19 S3 input manifest commit differs from the preregistration tag")
expected_targets = {
    "downstream_preflight.json": "artifacts/glm53_user_eval/v11/downstream_inputs/preflight.json",
    "v7_transcripts_all100.jsonl": "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100.jsonl",
    "v7_transcripts_all100_manifest.json": "artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100_manifest.json",
    "transluce_personas2.json": "artifacts/glm53_user_eval/v11/downstream_inputs/personas2.json",
    "v15_parent_decision.json": "artifacts/glm53_user_eval/v15/reports/codex_cohort/decision.json",
    "v15_parent_verification.json": "artifacts/glm53_user_eval/v15/reports/codex_cohort/verification.json",
    "immutable_v7_decision.json": "artifacts/glm53_user_eval/reports/transluce_interaction_v7/decision.json",
    "immutable_v7_analysis.json": "artifacts/glm53_user_eval/reports/transluce_interaction_v7/analysis.json",
    "immutable_v19_prereg.yaml": "pipelines/glm53_user_eval/v19/configs/prereg_v19_lean_hua.yaml",
    "immutable_v18_decision.json": "artifacts/glm53_user_eval/v18/reports/decision.json",
    "immutable_hua16.yaml": "manifests/task_sources/contrastive_prompts_v2/hua16_exact_raw_v1.yaml",
    "immutable_v15_dataset.jsonl": "artifacts/datasets/contrastive_prompts_v5/samples.jsonl",
    "immutable_proxy_token_contract.json": "pipelines/glm53_user_eval/v11/configs/proxy_token_contract_v2.json",
    "immutable_proxy_codebooks.json": "pipelines/glm53_user_eval/v11/configs/proxy_codebooks_v2.json",
    "immutable_downstream_manifest.json": "pipelines/glm53_user_eval/v17/configs/downstream_manifest_v17.json",
    "immutable_formality_pairs.json": "pipelines/glm53_user_eval/v17/configs/formality_pairs_v1.json",
    "immutable_neutral_damage_prompts.json": "pipelines/glm53_user_eval/v17/configs/neutral_damage_prompts_v1.json",
    "immutable_positive_control_manifest.json": "pipelines/glm53_user_eval/v17/configs/positive_control_manifest_v1.json",
    "immutable_positive_control_selection.json": "pipelines/glm53_user_eval/v19/configs/positive_control_selection_v19.json",
    "immutable_causal_schedule.json": "pipelines/glm53_user_eval/v8/configs/causal_schedule_v1.json",
    "immutable_user_prompt_templates.jsonl": "pipelines/glm53_user_eval/v8/configs/user_prompt_templates_v1.jsonl",
    "immutable_identity_selection.json": "pipelines/glm53_user_eval/configs/identity_selection_v1.json",
    "immutable_v19_design.json": "pipelines/glm53_user_eval/v19/configs/design_v19.json",
    "immutable_v19_runtime.yaml": "pipelines/glm53_user_eval/v19/configs/runtime_v19.yaml",
    "immutable_model_stage.json": "artifacts/glm53_user_eval/runtime/g2/model_stage.json",
    "immutable_transformers_source.tar.gz": "artifacts/glm53_user_eval/v17/infrastructure/transformers_805a9e939fa8c1bff8d8ffdf041c051b71a914aa.tar.gz",
}
files = manifest.get("files")
if not isinstance(files, dict) or set(files) != set(expected_targets):
    raise SystemExit("v19 S3 input manifest has the wrong file set")
for name, target_relative in expected_targets.items():
    record = files[name]
    if record.get("object_name") != name:
        raise SystemExit(f"v19 S3 object name mismatch: {name}")
    if record.get("target_relative_path") != target_relative:
        raise SystemExit(f"v19 S3 target path mismatch: {name}")
    digest = record.get("sha256", "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SystemExit(f"v19 S3 input lacks literal SHA-256: {name}")
    target = (project / target_relative).resolve()
    if project not in target.parents:
        raise SystemExit(f"v19 S3 target escapes project root: {name}")
(bundle / "verified_manifest.json").write_text(
    json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
)
PY

while IFS= read -r object_name; do
  s3_get "${GLM53_V19_INPUT_PREFIX}/${object_name}" \
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
        raise SystemExit(f"v19 S3 input hash mismatch: {name}")
    if partial.stat().st_size != int(record["bytes"]):
        raise SystemExit(f"v19 S3 input size mismatch: {name}")
    target = (project / record["target_relative_path"]).resolve()
    if project not in target.parents:
        raise SystemExit(f"v19 S3 target escapes project root: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_bytes(partial.read_bytes())
    os.replace(temporary, target)
PY

# The preregistration hashes bind the exact workstation bytes. Some JSON and
# YAML files therefore differ from Git's normalized blob bytes only in newline
# representation. Their signed hashes are the scientific integrity check; hide
# those expected transport substitutions from the later clean-tree guard.
cd "${PROJECT_ROOT}"
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

decision_path = Path("artifacts/glm53_user_eval/v15/reports/codex_cohort/decision.json")
verification_path = Path("artifacts/glm53_user_eval/v15/reports/codex_cohort/verification.json")
decision = json.loads(decision_path.read_text())
input_manifest = json.loads(
    Path("/tmp/glm53-v19-input-bundle/verified_manifest.json").read_text()
)
if decision.get("passed") is not True or decision.get("decision") != (
    "fresh_control_bank_validated_by_both_codex_judges"
):
    raise SystemExit("V15 parent decision did not pass")
if decision.get("schema_version") != "glm53_v15_fresh_controls_decision_v1":
    raise SystemExit("V15 parent decision schema differs")
for field in ("exact_fp8_source_extraction", "runpod_compute"):
    if decision.get("authorization", {}).get(field) is not True:
        raise SystemExit(f"V15 parent does not authorize {field}")
for field in ("local_proxy_parity", "prompt_recruitment", "first_cot_transfer", "steering"):
    if decision.get("authorization", {}).get(field) is not False:
        raise SystemExit(f"V15 parent must not pre-authorize {field}")
expected_hashes = {
    decision_path: "6546853ea1be45f179a4a396c621be375f15ba0ff5412d9775fbadbb4725b9fc",
    verification_path: "da4f4d7d3e6fd5d33a34f14f99eae33f05d43f9ca6af18dea062646daf25509e",
    Path("artifacts/datasets/contrastive_prompts_v5/samples.jsonl"): "a9c29d1a10a3eb7c431d8097808e9df10f888aa1585cc27638f44bb0aca1352e",
    Path("artifacts/datasets/contrastive_prompts_v5/manifest.json"): "f06b5d01727f29d5ccb55de2aa18de442aef6fff8aabd7e4443ebb54589c4e5b",
}
for path, expected in expected_hashes.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"V19 immutable input hash mismatch: {path}")
parent = input_manifest.get("parent_decision", {})
if parent.get("decision") != decision["decision"] or parent.get("sha256") != expected_hashes[decision_path]:
    raise SystemExit("V19 signed input manifest differs from the V15 parent")
paths = {
    "downstream_preflight.json": Path("artifacts/glm53_user_eval/v11/downstream_inputs/preflight.json"),
    "v7_transcripts_all100.jsonl": Path("artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100.jsonl"),
    "v7_transcripts_all100_manifest.json": Path("artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100_manifest.json"),
    "transluce_personas2.json": Path("artifacts/glm53_user_eval/v11/downstream_inputs/personas2.json"),
}
for object_name, path in paths.items():
    expected = input_manifest["files"][object_name]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"V19 reconstructed input differs from signed hash: {path}")
PY

if [[ -e "${UV_PROJECT_ENVIRONMENT}" ]]; then
  printf 'Refusing to reuse an existing v19 virtual environment\n' >&2
  exit 1
fi
uv sync --frozen
uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
  torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip uninstall --python "${UV_PROJECT_ENVIRONMENT}/bin/python" torchaudio || true
transformers_source="${PROJECT_ROOT}/artifacts/glm53_user_eval/v17/infrastructure/transformers_805a9e939fa8c1bff8d8ffdf041c051b71a914aa.tar.gz"
printf '%s  %s\n' \
  '17890f68cae495a88b51db8105fd9bca43d5357f671fce925e3fe1f63c3cac0a' \
  "${transformers_source}" | sha256sum --check --strict -
uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" --reinstall \
  "${transformers_source}" kernels==0.16.0

"${UV_PROJECT_ENVIRONMENT}/bin/python" - <<'PY'
import importlib.metadata
import json
import torch

distribution = importlib.metadata.distribution("transformers")
direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
expected_archive_hash = "17890f68cae495a88b51db8105fd9bca43d5357f671fce925e3fe1f63c3cac0a"
expected_archive_name = "transformers_805a9e939fa8c1bff8d8ffdf041c051b71a914aa.tar.gz"
source_url = direct_url.get("url", "").replace("\\", "/")
if not source_url.endswith(f"/{expected_archive_name}"):
    raise SystemExit(f"transformers direct URL does not name the pinned source: {source_url!r}")
if distribution.version != "5.16.0.dev0":
    raise SystemExit(f"unexpected transformers version: {distribution.version}")
if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
    raise SystemExit(f"unexpected torch runtime: {torch.__version__}, {torch.version.cuda}")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"expected two GPUs, found {torch.cuda.device_count()}")
gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
if any(name != "NVIDIA B300 SXM6 AC" for name in gpu_names):
    raise SystemExit(
        f"expected only NVIDIA B300 SXM6 AC GPUs, found {gpu_names}"
    )
print(json.dumps({
    "transformers_commit": "805a9e939fa8c1bff8d8ffdf041c051b71a914aa",
    "transformers_source_sha256": expected_archive_hash,
    "transformers_version": distribution.version,
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

stage_report="${V19_ARTIFACT_ROOT}/infrastructure/model_stage.json"
"${UV_PROJECT_ENVIRONMENT}/bin/python" infra/runpod/stage_glm53_v8_hf_local.py \
  --manifest artifacts/glm53_user_eval/runtime/g2/model_stage.json \
  --target "${GLM53_LOCAL_MODEL_PATH}" \
  --runtime-link "${GLM53_MODEL_PATH}" \
  --report "${stage_report}" \
  --workers 8
s3_put "${stage_report}" \
  "${GLM53_V19_S3_PREFIX}/${GLM53_V19_RUN_ID}/infrastructure/model_stage.json"

if [[ "${GLM53_V19_RUN_SCIENCE:-0}" != "1" ]]; then
  upload_terminal 0 "bootstrap_only_complete"
  exit 0
fi

stop_heartbeat
start_heartbeat "hua_positive_control"
runner=("${UV_PROJECT_ENVIRONMENT}/bin/python" pipelines/glm53_user_eval/v19/run.py)
"${runner[@]}" paid-supervisor --model-path "${GLM53_MODEL_PATH}" \
  --runtime pipelines/glm53_user_eval/v19/configs/runtime_v19.yaml \
  --confirm-spend &
ladder_pid="$!"
while kill -0 "${ladder_pid}" 2>/dev/null; do
  sleep 60
  upload_current_artifacts || true
done
wait "${ladder_pid}"

evidence_manifest="${V19_ARTIFACT_ROOT}/infrastructure/evidence_manifest.json"
"${UV_PROJECT_ENVIRONMENT}/bin/python" - "${V19_ARTIFACT_ROOT}" "${evidence_manifest}" <<'PY'
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
    "schema_version": "glm53_v19_evidence_manifest_v1",
    "run_id": os.environ["GLM53_V19_RUN_ID"],
    "pod_id": os.environ["RUNPOD_POD_ID"],
    "files": files,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY

upload_current_artifacts

roundtrip="/tmp/glm53-v19-source-evidence-manifest.roundtrip.json"
s3_get \
  "${GLM53_V19_S3_PREFIX}/${GLM53_V19_RUN_ID}/artifacts/infrastructure/evidence_manifest.json" \
  "${roundtrip}"
test "$(sha256sum "${evidence_manifest}" | cut -d' ' -f1)" = \
  "$(sha256sum "${roundtrip}" | cut -d' ' -f1)"
rm -f -- "${roundtrip}"
upload_terminal 0 "v19_terminal_decision_written"
