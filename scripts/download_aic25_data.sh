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
VERIFY_ONLY=false
ONLY_GROUPS=""
DOWNLOAD_JOBS=1
CONNECTIONS_PER_ARCHIVE=16

EXPECTED_VIDEO_COUNT=873
EXPECTED_KEYFRAME_COUNT=177321
EXPECTED_MAP_COUNT=873
EXPECTED_METADATA_COUNT=873
EXPECTED_FEATURE_COUNT=873
EXPECTED_OBJECT_COUNT=177321

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
  --jobs COUNT         Concurrent archive downloads (default: 1, maximum: 8).
  --connections COUNT  Connections used for each archive by aria2c (default: 16,
                       maximum: 32).
  --keep-archives      Keep verified ZIP archives after extraction.
  --overwrite          Replace files already present in the target data directory.
  --download-only      Download and validate ZIPs; do not extract them.
  --extract-only       Extract archives already present in --download-dir.
  --verify-only        Validate the complete extracted dataset; do not download.
  -h, --help           Show this message.

By default existing data files are preserved, verified ZIP archives are removed
after successful extraction, and incomplete downloads resume via aria2c. A
completion marker prevents already extracted archives from being downloaded
again. A full run validates the final dataset layout and expected file counts.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    if [[ "$1" == "unzip" || "$1" == "rsync" || "$1" == "aria2c" ]]; then
      echo "Ubuntu: sudo apt update && sudo apt install -y aria2 unzip rsync" >&2
    fi
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
    media-info-aic25-b1) printf '%s\n' 'media-info-aic25-b1/media-info' ;;
    objects-aic25-b1) printf '%s\n' 'objects-aic25-b1/objects' ;;
    *)
      echo "Unknown archive group: $1" >&2
      exit 1
      ;;
  esac
}

payload_leaf_for_group() {
  case "$1" in
    keyframes) printf '%s\n' 'keyframes' ;;
    video) printf '%s\n' 'video' ;;
    clip-features-32) printf '%s\n' 'clip-features-32' ;;
    map-keyframes) printf '%s\n' 'map-keyframes' ;;
    media-info-aic25-b1) printf '%s\n' 'media-info' ;;
    objects-aic25-b1) printf '%s\n' 'objects' ;;
    *)
      echo "Unknown archive group: $1" >&2
      exit 1
      ;;
  esac
}

completion_marker() {
  local filename=$1
  printf '%s/.completed/%s.complete\n' "${DOWNLOAD_DIR}" "${filename}"
}

download_archive() {
  local archive_path=$1
  local url=$2
  if [[ -f "${archive_path}" ]]; then
    if unzip -tq "${archive_path}" >/dev/null; then
      echo "Using verified archive: ${archive_path}"
      return
    fi
    echo "Resuming incomplete or invalid archive: ${archive_path}" >&2
  fi

  echo "Downloading: ${url}"
  aria2c \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --continue=true \
    --dir "${DOWNLOAD_DIR}" \
    --file-allocation=none \
    --max-connection-per-server="${CONNECTIONS_PER_ARCHIVE}" \
    --max-tries=5 \
    --min-split-size=4M \
    --out "$(basename -- "${archive_path}")" \
    --retry-wait=3 \
    --split="${CONNECTIONS_PER_ARCHIVE}" \
    --summary-interval=10 \
    "${url}"
  unzip -tq "${archive_path}" >/dev/null
}

find_payload_root() {
  local extraction_dir=$1
  local group=$2
  local target_name
  target_name=$(target_for_group "${group}")
  local payload_leaf
  payload_leaf=$(payload_leaf_for_group "${group}")
  local direct="${extraction_dir}/${target_name}"
  local found

  if [[ -d "${direct}" ]]; then
    printf '%s\n' "${direct}"
    return
  fi

  found=$(find "${extraction_dir}" -mindepth 1 -type d -name "${payload_leaf}" -print -quit)
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
  payload_root=$(find_payload_root "${extraction_dir}" "${group}")
  mkdir -p "${target_dir}"
  if [[ "${OVERWRITE}" == true ]]; then
    rsync -a "${payload_root}/" "${target_dir}/"
  else
    rsync -a --ignore-existing "${payload_root}/" "${target_dir}/"
  fi

  rm -rf -- "${extraction_dir}"
  trap - RETURN
}

count_files() {
  local root=$1
  local pattern=$2
  find "${root}" -type f -name "${pattern}" -print | wc -l
}

verify_exact_count() {
  local label=$1
  local root=$2
  local pattern=$3
  local expected=$4
  local actual=0

  if [[ -d "${root}" ]]; then
    actual=$(count_files "${root}" "${pattern}")
  fi
  printf '%-24s expected=%-8s actual=%s\n' "${label}" "${expected}" "${actual}"
  [[ "${actual}" -eq "${expected}" ]]
}

verify_minimum_count() {
  local label=$1
  local root=$2
  local pattern=$3
  local expected_minimum=$4
  local actual=0

  if [[ -d "${root}" ]]; then
    actual=$(count_files "${root}" "${pattern}")
  fi
  printf '%-24s expected>=%-6s actual=%s\n' "${label}" "${expected_minimum}" "${actual}"
  [[ "${actual}" -ge "${expected_minimum}" ]]
}

write_file_ids() {
  local root=$1
  local pattern=$2
  local output=$3
  find "${root}" -mindepth 1 -maxdepth 1 -type f -name "${pattern}" -printf '%f\n' \
    | sed 's/\.[^.]*$//' \
    | LC_ALL=C sort -u > "${output}"
}

write_directory_ids() {
  local root=$1
  local output=$2
  find "${root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | LC_ALL=C sort -u > "${output}"
}

verify_id_set() {
  local label=$1
  local expected_file=$2
  local actual_file=$3
  local differences
  differences=$(comm -3 "${expected_file}" "${actual_file}" | sed -n '1,10p')
  if [[ -n "${differences}" ]]; then
    echo "${label}: video ID set mismatch (first 10 differences):" >&2
    echo "${differences}" >&2
    return 1
  fi
  echo "${label}: video ID set matches metadata."
}

verify_full_dataset() {
  local failures=0
  local verification_dir
  verification_dir=$(mktemp -d "${DOWNLOAD_DIR}/verify.XXXXXX")
  trap 'rm -rf -- "${verification_dir}"' RETURN

  echo "Validating complete AIC25 B1 dataset under: ${DATA_ROOT}"
  verify_exact_count "videos (.mp4)" "${DATA_ROOT}/video" "*.mp4" "${EXPECTED_VIDEO_COUNT}" || failures=$((failures + 1))
  verify_minimum_count "keyframes (.jpg)" "${DATA_ROOT}/keyframes" "*.jpg" "${EXPECTED_KEYFRAME_COUNT}" || failures=$((failures + 1))
  verify_exact_count "keyframe maps (.csv)" "${DATA_ROOT}/map-keyframes" "*.csv" "${EXPECTED_MAP_COUNT}" || failures=$((failures + 1))
  verify_exact_count "metadata (.json)" "${DATA_ROOT}/media-info-aic25-b1/media-info" "*.json" "${EXPECTED_METADATA_COUNT}" || failures=$((failures + 1))
  verify_exact_count "CLIP features (.npy)" "${DATA_ROOT}/clip-features-32" "*.npy" "${EXPECTED_FEATURE_COUNT}" || failures=$((failures + 1))
  verify_exact_count "object files (.json)" "${DATA_ROOT}/objects-aic25-b1/objects" "*.json" "${EXPECTED_OBJECT_COUNT}" || failures=$((failures + 1))

  if [[ "${failures}" -eq 0 ]]; then
    write_file_ids "${DATA_ROOT}/media-info-aic25-b1/media-info" "*.json" "${verification_dir}/metadata.ids"
    write_file_ids "${DATA_ROOT}/video" "*.mp4" "${verification_dir}/video.ids"
    write_file_ids "${DATA_ROOT}/map-keyframes" "*.csv" "${verification_dir}/map.ids"
    write_file_ids "${DATA_ROOT}/clip-features-32" "*.npy" "${verification_dir}/features.ids"
    write_directory_ids "${DATA_ROOT}/keyframes" "${verification_dir}/keyframes.ids"
    write_directory_ids "${DATA_ROOT}/objects-aic25-b1/objects" "${verification_dir}/objects.ids"

    verify_id_set "videos" "${verification_dir}/metadata.ids" "${verification_dir}/video.ids" || failures=$((failures + 1))
    verify_id_set "keyframe maps" "${verification_dir}/metadata.ids" "${verification_dir}/map.ids" || failures=$((failures + 1))
    verify_id_set "CLIP features" "${verification_dir}/metadata.ids" "${verification_dir}/features.ids" || failures=$((failures + 1))
    verify_id_set "keyframe directories" "${verification_dir}/metadata.ids" "${verification_dir}/keyframes.ids" || failures=$((failures + 1))
    verify_id_set "object directories" "${verification_dir}/metadata.ids" "${verification_dir}/objects.ids" || failures=$((failures + 1))
  fi

  rm -rf -- "${verification_dir}"
  trap - RETURN
  if [[ "${failures}" -ne 0 ]]; then
    echo "Dataset verification failed with ${failures} problem(s)." >&2
    return 1
  fi
  echo "Dataset verification passed."
}

download_selected_archives() {
  local -a entries=("$@")
  local -a process_ids=()
  local entry group filename url archive_path

  for entry in "${entries[@]}"; do
    IFS='|' read -r group filename url <<< "${entry}"
    archive_path="${DOWNLOAD_DIR}/${filename}"
    download_archive "${archive_path}" "${url}" &
    process_ids+=("$!")

    if [[ ${#process_ids[@]} -ge ${DOWNLOAD_JOBS} ]]; then
      for process_id in "${process_ids[@]}"; do
        wait "${process_id}"
      done
      process_ids=()
    fi
  done

  for process_id in "${process_ids[@]}"; do
    wait "${process_id}"
  done
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
      --jobs)
        DOWNLOAD_JOBS=${2:?"--jobs requires a positive integer"}
        shift 2
        ;;
      --connections)
        CONNECTIONS_PER_ARCHIVE=${2:?"--connections requires a positive integer"}
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
      --verify-only)
        VERIFY_ONLY=true
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

  if [[ "${VERIFY_ONLY}" == true && ("${DOWNLOAD_ONLY}" == true || "${EXTRACT_ONLY}" == true) ]]; then
    echo "--verify-only cannot be combined with download/extract modes." >&2
    exit 2
  fi
  if [[ "${DOWNLOAD_ONLY}" == true && "${EXTRACT_ONLY}" == true ]]; then
    echo "--download-only and --extract-only cannot be used together." >&2
    exit 2
  fi
  if ! [[ "${DOWNLOAD_JOBS}" =~ ^[1-8]$ ]]; then
    echo "--jobs must be an integer from 1 to 8." >&2
    exit 2
  fi
  if ! [[ "${CONNECTIONS_PER_ARCHIVE}" =~ ^([1-9]|[12][0-9]|3[0-2])$ ]]; then
    echo "--connections must be an integer from 1 to 32." >&2
    exit 2
  fi
}

main() {
  parse_arguments "$@"
  require_command find
  require_command sort
  require_command comm
  require_command sed
  require_command wc

  mkdir -p "${DATA_ROOT}" "${DOWNLOAD_DIR}" "${DOWNLOAD_DIR}/.completed"
  if [[ "${VERIFY_ONLY}" == true ]]; then
    verify_full_dataset
    exit 0
  fi

  require_command aria2c
  require_command unzip
  require_command rsync

  local -a selected_entries=()
  local entry group filename url archive_path marker_path
  for entry in "${ARCHIVES[@]}"; do
    IFS='|' read -r group filename url <<< "${entry}"
    group_is_selected "${group}" || continue
    marker_path=$(completion_marker "${filename}")
    if [[ "${DOWNLOAD_ONLY}" == false && "${EXTRACT_ONLY}" == false && "${OVERWRITE}" == false && -f "${marker_path}" ]]; then
      echo "Skipping completed archive: ${filename}"
      continue
    fi
    selected_entries+=("${entry}")
  done

  if [[ ${#selected_entries[@]} -eq 0 ]]; then
    if [[ -z "${ONLY_GROUPS}" ]]; then
      echo "All archives have completion markers; validating extracted data."
      verify_full_dataset
      exit 0
    fi
    echo "No pending archives for the selected groups."
    exit 0
  fi

  local start
  local -a batch=()
  for ((start = 0; start < ${#selected_entries[@]}; start += DOWNLOAD_JOBS)); do
    batch=("${selected_entries[@]:start:DOWNLOAD_JOBS}")
    if [[ "${EXTRACT_ONLY}" == false ]]; then
      download_selected_archives "${batch[@]}"
    fi

    for entry in "${batch[@]}"; do
      IFS='|' read -r group filename url <<< "${entry}"
      archive_path="${DOWNLOAD_DIR}/${filename}"

      if [[ "${EXTRACT_ONLY}" == true && ! -f "${archive_path}" ]]; then
        echo "Missing archive for --extract-only: ${archive_path}" >&2
        exit 1
      fi

      if [[ "${DOWNLOAD_ONLY}" == false ]]; then
        extract_archive "${archive_path}" "${group}"
        marker_path=$(completion_marker "${filename}")
        printf 'archive=%s\nurl=%s\ncompleted_at=%s\n' \
          "${filename}" "${url}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${marker_path}"
        if [[ "${KEEP_ARCHIVES}" == false ]]; then
          rm -- "${archive_path}"
        fi
      fi
    done
  done

  echo "Completed ${#selected_entries[@]} archive(s)."
  if [[ -z "${ONLY_GROUPS}" && "${DOWNLOAD_ONLY}" == false ]]; then
    verify_full_dataset
  fi
}

main "$@"
