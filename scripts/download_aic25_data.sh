#!/usr/bin/env bash
# Download and safely merge AIC25 B1 archives into the repository data layout.
# Source manifest: https://docs.google.com/spreadsheets/d/1rfn1fieTThS_Ki3SIoJ6uXOx2AhMq7wGCak6W4jZyZM/edit?gid=0#gid=0

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_ROOT="${PROJECT_ROOT}/data"
DOWNLOAD_DIR="${DATA_ROOT}/.downloads"
KEEP_ARCHIVES=false
OVERWRITE=false
DOWNLOAD_ONLY=false
EXTRACT_ONLY=false
ONLY_GROUPS=""

readonly -a ARCHIVES=(
  "keyframes|Keyframes_L21.zip|https://aic-data.ledo.io.vn/Keyframes_L21.zip"
  "keyframes|Keyframes_L22.zip|https://aic-data.ledo.io.vn/Keyframes_L22.zip"
  "keyframes|Keyframes_L23.zip|https://aic-data.ledo.io.vn/Keyframes_L23.zip"
  "keyframes|Keyframes_L24.zip|https://aic-data.ledo.io.vn/Keyframes_L24.zip"
  "keyframes|Keyframes_L25.zip|https://aic-data.ledo.io.vn/Keyframes_L25.zip"
  "keyframes|Keyframes_L26_a.zip|https://aic-data.ledo.io.vn/Keyframes_L26_a.zip"
  "keyframes|Keyframes_L26_b.zip|https://aic-data.ledo.io.vn/Keyframes_L26_b.zip"
  "keyframes|Keyframes_L26_c.zip|https://aic-data.ledo.io.vn/Keyframes_L26_c.zip"
  "keyframes|Keyframes_L26_d.zip|https://aic-data.ledo.io.vn/Keyframes_L26_d.zip"
  "keyframes|Keyframes_L26_e.zip|https://aic-data.ledo.io.vn/Keyframes_L26_e.zip"
  "keyframes|Keyframes_L27.zip|https://aic-data.ledo.io.vn/Keyframes_L27.zip"
  "keyframes|Keyframes_L28.zip|https://aic-data.ledo.io.vn/Keyframes_L28.zip"
  "keyframes|Keyframes_L29.zip|https://aic-data.ledo.io.vn/Keyframes_L29.zip"
  "keyframes|Keyframes_L30.zip|https://aic-data.ledo.io.vn/Keyframes_L30.zip"
  "video|Videos_L21_a.zip|https://aic-data.ledo.io.vn/Videos_L21_a.zip"
  "video|Videos_L22_a.zip|https://aic-data.ledo.io.vn/Videos_L22_a.zip"
  "video|Videos_L23_a.zip|https://aic-data.ledo.io.vn/Videos_L23_a.zip"
  "video|Videos_L24_a.zip|https://aic-data.ledo.io.vn/Videos_L24_a.zip"
  "video|Videos_L25_a.zip|https://aic-data.ledo.io.vn/Videos_L25_a.zip"
  "video|Videos_L26_a.zip|https://aic-data.ledo.io.vn/Videos_L26_a.zip"
  "video|Videos_L26_b.zip|https://aic-data.ledo.io.vn/Videos_L26_b.zip"
  "video|Videos_L26_c.zip|https://aic-data.ledo.io.vn/Videos_L26_c.zip"
  "video|Videos_L26_d.zip|https://aic-data.ledo.io.vn/Videos_L26_d.zip"
  "video|Videos_L26_e.zip|https://aic-data.ledo.io.vn/Videos_L26_e.zip"
  "video|Videos_L27_a.zip|https://aic-data.ledo.io.vn/Videos_L27_a.zip"
  "video|Videos_L28_a.zip|https://aic-data.ledo.io.vn/Videos_L28_a.zip"
  "video|Videos_L29_a.zip|https://aic-data.ledo.io.vn/Videos_L29_a.zip"
  "video|Videos_L30_a.zip|https://aic-data.ledo.io.vn/Videos_L30_a.zip"
  "clip-features-32|clip-features-32-aic25-b1.zip|https://aic-data.ledo.io.vn/clip-features-32-aic25-b1.zip"
  "map-keyframes|map-keyframes-aic25-b1.zip|https://aic-data.ledo.io.vn/map-keyframes-aic25-b1.zip"
  "media-info-aic25-b1|media-info-aic25-b1.zip|https://aic-data.ledo.io.vn/media-info-aic25-b1.zip"
  "objects-aic25-b1|objects-aic25-b1.zip|https://aic-data.ledo.io.vn/objects-aic25-b1.zip"
)

usage() {
  cat <<'EOF'
Usage: scripts/download_aic25_data.sh [options]

Downloads the AIC25 B1 public archives and merges them into data/.

Options:
  --only GROUPS        Comma-separated groups: keyframes,video,clip-features-32,
                       map-keyframes,media-info-aic25-b1,objects-aic25-b1.
  --download-dir PATH  Archive cache directory (default: data/.downloads).
  --keep-archives      Keep verified ZIP archives after extraction.
  --overwrite          Replace files already present in the target data directory.
  --download-only      Download and validate ZIPs; do not extract them.
  --extract-only       Extract archives already present in --download-dir.
  -h, --help           Show this message.

By default existing data files are preserved, verified ZIP archives are removed
after successful extraction, and incomplete downloads resume via curl.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 1
  }
}

group_is_selected() {
  local group=$1
  [[ -z "${ONLY_GROUPS}" ]] && return 0
  [[ ",${ONLY_GROUPS}," == *",${group},"* ]]
}

target_for_group() {
  case "$1" in
    keyframes) printf '%s\n' 'keyframes' ;;
    video) printf '%s\n' 'video' ;;
    clip-features-32) printf '%s\n' 'clip-features-32' ;;
    map-keyframes) printf '%s\n' 'map-keyframes' ;;
    media-info-aic25-b1) printf '%s\n' 'media-info-aic25-b1' ;;
    objects-aic25-b1) printf '%s\n' 'objects-aic25-b1' ;;
    *)
      echo "Unknown archive group: $1" >&2
      exit 1
      ;;
  esac
}

download_archive() {
  local archive_path=$1
  local url=$2
  local partial_path="${archive_path}.part"

  if [[ -f "${archive_path}" ]]; then
    if unzip -tq "${archive_path}" >/dev/null; then
      echo "Using verified archive: ${archive_path}"
      return
    fi
    echo "Existing archive is invalid: ${archive_path}" >&2
    echo "Move or remove it manually, then rerun the script." >&2
    exit 1
  fi

  echo "Downloading: ${url}"
  curl --fail --location --retry 5 --retry-all-errors --continue-at - \
    --output "${partial_path}" "${url}"
  unzip -tq "${partial_path}" >/dev/null
  mv -- "${partial_path}" "${archive_path}"
}

find_payload_root() {
  local extraction_dir=$1
  local target_name=$2
  local direct="${extraction_dir}/${target_name}"
  local found

  if [[ -d "${direct}" ]]; then
    printf '%s\n' "${direct}"
    return
  fi

  found=$(find "${extraction_dir}" -mindepth 1 -type d -name "${target_name}" -print -quit)
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
    return
  fi

  local -a entries=()
  while IFS= read -r entry; do
    entries+=("${entry}")
  done < <(find "${extraction_dir}" -mindepth 1 -maxdepth 1 -print)

  if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
    printf '%s\n' "${entries[0]}"
    return
  fi

  printf '%s\n' "${extraction_dir}"
}

extract_archive() {
  local archive_path=$1
  local group=$2
  local target_name
  target_name=$(target_for_group "${group}")
  local target_dir="${DATA_ROOT}/${target_name}"
  local extraction_dir
  extraction_dir=$(mktemp -d "${DOWNLOAD_DIR}/extract.XXXXXX")

  trap 'rm -rf -- "${extraction_dir}"' RETURN
  echo "Extracting: ${archive_path} -> ${target_dir}"
  unzip -q "${archive_path}" -d "${extraction_dir}"

  local payload_root
  payload_root=$(find_payload_root "${extraction_dir}" "${target_name}")
  mkdir -p "${target_dir}"
  if [[ "${OVERWRITE}" == true ]]; then
    rsync -a "${payload_root}/" "${target_dir}/"
  else
    rsync -a --ignore-existing "${payload_root}/" "${target_dir}/"
  fi

  rm -rf -- "${extraction_dir}"
  trap - RETURN
}

parse_arguments() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --only)
        ONLY_GROUPS=${2:?"--only requires a value"}
        shift 2
        ;;
      --download-dir)
        DOWNLOAD_DIR=${2:?"--download-dir requires a path"}
        shift 2
        ;;
      --keep-archives)
        KEEP_ARCHIVES=true
        shift
        ;;
      --overwrite)
        OVERWRITE=true
        shift
        ;;
      --download-only)
        DOWNLOAD_ONLY=true
        shift
        ;;
      --extract-only)
        EXTRACT_ONLY=true
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done

  if [[ "${DOWNLOAD_ONLY}" == true && "${EXTRACT_ONLY}" == true ]]; then
    echo "--download-only and --extract-only cannot be used together." >&2
    exit 2
  fi
}

main() {
  parse_arguments "$@"
  require_command curl
  require_command unzip
  require_command rsync
  require_command find

  mkdir -p "${DATA_ROOT}" "${DOWNLOAD_DIR}"
  local selected_count=0
  local entry group filename url archive_path
  for entry in "${ARCHIVES[@]}"; do
    IFS='|' read -r group filename url <<< "${entry}"
    group_is_selected "${group}" || continue
    selected_count=$((selected_count + 1))
    archive_path="${DOWNLOAD_DIR}/${filename}"

    if [[ "${EXTRACT_ONLY}" == false ]]; then
      download_archive "${archive_path}" "${url}"
    elif [[ ! -f "${archive_path}" ]]; then
      echo "Missing archive for --extract-only: ${archive_path}" >&2
      exit 1
    fi

    if [[ "${DOWNLOAD_ONLY}" == false ]]; then
      extract_archive "${archive_path}" "${group}"
      if [[ "${KEEP_ARCHIVES}" == false ]]; then
        rm -- "${archive_path}"
      fi
    fi
  done

  if [[ ${selected_count} -eq 0 ]]; then
    echo "No archive groups selected. Check --only." >&2
    exit 2
  fi
  echo "Completed ${selected_count} archive(s)."
}

main "$@"
