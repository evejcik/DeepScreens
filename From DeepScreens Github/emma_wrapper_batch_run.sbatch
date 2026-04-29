#!/bin/bash -l

# Name of the job
#SBATCH --job-name=DeepScreensFullPipeline

# Number of compute nodes
#SBATCH --nodes=1

# Request the GPU resources
#SBATCH --gres=gpu:1

# Request memory
#SBATCH --mem=500GB

# Request the GPU partition
#SBATCH --partition=a100

# Walltime (job duration)
#SBATCH --time=18:00:00

# Email notifications
#SBATCH --mail-type=ALL

# Specify account
#SBATCH --account=mep

# specify folder
#SBATCH --mail-type=ALL

# --- Normalize exported env vars: strip a single leading/trailing double-quote if present ---
# This is defensive: some submission environments may incorrectly send values like "mmpose"
# (including literal quote characters). Remove a single surrounding quote if present.
normalize_quotes() {
    local v="$1"
    # remove one leading quote if present, then one trailing quote if present
    v="${v#\"}"
    v="${v%\"}"
    printf '%s' "$v"
}

# Apply normalization to exported variables we care about.
# Add names here as needed.
if [ -n "${ENV_DS_MODEL:-}" ]; then
    ENV_DS_MODEL="$(normalize_quotes "${ENV_DS_MODEL}")"
fi
if [ -n "${NOTIFY_EMAIL:-}" ]; then
    NOTIFY_EMAIL="$(normalize_quotes "${NOTIFY_EMAIL}")"
fi
if [ -n "${CUTFILES:-}" ]; then
    # CUTFILES may contain embedded delimiters and internal quotes; strip only outer quotes.
    CUTFILES="$(normalize_quotes "${CUTFILES}")"
fi
if [ -n "${DRY_RUN_IN_PYTHON:-}" ]; then
    DRY_RUN_IN_PYTHON="$(normalize_quotes "${DRY_RUN_IN_PYTHON}")"
fi
if [ -n "${ENV_DS_QUIET:-}" ]; then
    ENV_DS_QUIET="$(normalize_quotes "${ENV_DS_QUIET}")"
fi
# Normalize ENV_DS_LOGPATH if present
if [ -n "${ENV_DS_LOGPATH:-}" ]; then
    ENV_DS_LOGPATH="$(normalize_quotes "${ENV_DS_LOGPATH}")"
fi
# Normalize ENV_DS_VIDEO if present
if [ -n "${ENV_DS_VIDEO:-}" ]; then
    ENV_DS_VIDEO="$(normalize_quotes "${ENV_DS_VIDEO}")"
fi
# Normalize ENV_DS_CLEANUP if present
if [ -n "${ENV_DS_CLEANUP:-}" ]; then
    ENV_DS_CLEANUP="$(normalize_quotes "${ENV_DS_CLEANUP}")"
fi
# Normalize ENV_DS_MIN_FRAMES if present
if [ -n "${ENV_DS_MIN_FRAMES:-}" ]; then
    ENV_DS_MIN_FRAMES="$(normalize_quotes "${ENV_DS_MIN_FRAMES}")"
fi
# Normalize ENV_DS_SMOOTH_2D if present
if [ -n "${ENV_DS_SMOOTH_2D:-}" ]; then
    ENV_DS_SMOOTH_2D="$(normalize_quotes "${ENV_DS_SMOOTH_2D}")"
fi
# Normalize ENV_DS_SMOOTH_3D if present
if [ -n "${ENV_DS_SMOOTH_3D:-}" ]; then
    ENV_DS_SMOOTH_3D="$(normalize_quotes "${ENV_DS_SMOOTH_3D}")"
fi
# Normalize ENV_DS_ZIPONLY if present
if [ -n "${ENV_DS_ZIPONLY:-}" ]; then
    ENV_DS_ZIPONLY="$(normalize_quotes "${ENV_DS_ZIPONLY}")"
fi


# --- end normalization ---

# Helpers
# Run a command array, optionally suppressing stdout if ENV_DS_QUIET=1
run_cmd() {
    local -n _arr=$1
    if [ "${ENV_DS_QUIET:-0}" = "1" ]; then
        "${_arr[@]}" > /dev/null 2>&1
    else
        "${_arr[@]}"
    fi
}

# shell-quote a single argument for readable printing (single-quote style)
shell_quote() {
    printf "'%s'" "$(printf "%s" "$1" | sed "s/'/'\\\\''/g")"
}

# print an array command with arguments quoted
print_cmd_array() {
    local -n arr=$1
    local out=""
    for a in "${arr[@]}"; do
        if [ -z "$out" ]; then
            out="$(shell_quote "$a")"
        else
            out="$out $(shell_quote "$a")"
        fi
    done
    printf '%s\n' "$out"
}

# Set email for notifications from environment variable, else default to none
if [ -n "$NOTIFY_EMAIL" ]; then
    # This changes the SLURM mail-user at job runtime:
    export SLURM_MAIL_USER="$NOTIFY_EMAIL"
else
    export SLURM_MAIL_USER=""
fi

# Set encoding
export LC_CTYPE=en_US.UTF-8

#
# Recover CUTFILES (joined by the submit script using ASCII RS 0x1E)
#
DELIM=$'\036'
if [ -z "${CUTFILES:-}" ]; then
    echo "Error: CUTFILES not provided in environment. Exiting." >&2
    exit 1
fi

# Split into array on the same delimiter used by the submitter
IFS="$DELIM" read -r -a input_files <<< "$CUTFILES"

# Remove any empty entries (e.g., from a trailing delimiter)
_compact_input_files=()
for f in "${input_files[@]}"; do
    [ -n "$f" ] || continue
    _compact_input_files+=("$f")
done
input_files=("${_compact_input_files[@]}")
unset _compact_input_files

# Validate ENV_DS_MODEL if provided. Allowed values: mmpose, rtmw
if [ -n "${ENV_DS_MODEL:-}" ]; then
    case "${ENV_DS_MODEL}" in
        mmpose|rtmw|hybrid|none)
            # allowed — continue
            ;;
        *)
            echo "Error: ENV_DS_MODEL has unrecognized value: '${ENV_DS_MODEL}'. Allowed: mmpose, rtmw, hybrid, none" >&2
            exit 1
            ;;
    esac
fi

# Map array index (default to 0 if not an array job)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    task_id="${SLURM_ARRAY_TASK_ID}"
else
    task_id=0
fi

if [ -z "${input_files[$task_id]:-}" ]; then
    echo "Error: No input file for SLURM_ARRAY_TASK_ID=${task_id}" >&2
    exit 1
fi

input_file=${input_files[$task_id]}

# Centralized paths and constants
ROOT_MODELS="/isi/mep-rip/working/deepscreens/models"
ROOT_MMPOSE="/isi/mep-rip/working/deepscreens/mmpose"
ROOT_RTMW="/isi/mep-rip/working/deepscreens/rtmw"
ROOT_DEEPSCREENS="/isi/mep-rip/working/deepscreens/python/"
OUTPUT_MMPOSE_FINAL="/isi/mep-rip/working/deepscreens/output/mmpose"
OUTPUT_PYSCENE_FINAL="/isi/mep-rip/working/deepscreens/output/pyscenedetect"
OUTPUT_JSON_FINAL="/isi/mep-rip/working/deepscreens/output/final/"
META_INFO_ROOT="/isi/mep-rip/working/deepscreens/python/film_data.json"
SCRATCH_BASE="/scratch/deepscreens"

# Centralized conda environment paths (used by conda activate later)
# CONDA_SH="/dartfs-hpc/rc/lab/M/MEP/sharedconda/miniconda3/etc/profile.d/conda.sh"
CONDA_SH="/optnfs/common/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_MMPOSE="/isi/mep-rip/working/shared_miniconda/mmpose"
CONDA_ENV_RTMW="/isi/mep-rip/working/shared_miniconda/rtmw" #for both rtmw and hybrid
CONDA_ENV_PYSCENE="/isi/mep-rip/working/shared_miniconda/pyscenedetect"

# verify required model files exist TODO
REQUIRED_MODELS=(
    "${ROOT_MODELS}/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth"
    "${ROOT_MODELS}/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    "${ROOT_MODELS}/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
    "${ROOT_MODELS}/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
    "${ROOT_MODELS}/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth"
    "${ROOT_MODELS}/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
)

for f in "${REQUIRED_MODELS[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Error: required model file not found: $f" >&2
    exit 1
  fi
done

# --- SCRATCH DIRECTORY SETUP ---
# Create base scratch directory if it doesn't exist
if [ ! -d "${SCRATCH_BASE}" ]; then
    mkdir -p "${SCRATCH_BASE}" || {
        echo "Error: Failed to create scratch base directory ${SCRATCH_BASE}" >&2
        exit 1
    }
fi

# Create scratch copy of models directory if it doesn't exist
SCRATCH_MODELS="${SCRATCH_BASE}/models"
if [ ! -d "${SCRATCH_MODELS}" ]; then
    mkdir -p "${SCRATCH_MODELS}" || {
        echo "Error: Failed to create scratch models directory ${SCRATCH_MODELS}" >&2
        exit 1
    }
    # Loop through required model files and copy them to scratch
    for f in "${REQUIRED_MODELS[@]}"; do
        # Check if the model file exists in scratch already
        if [ ! -f "${SCRATCH_MODELS}/$(basename "$f")" ]; then
            cp -r "$f" "${SCRATCH_MODELS}/" || {
                echo "Error: Failed to copy model file to scratch directory: $f" >&2
                exit 1
            }
        fi
    done
fi

# Create job-specific scratch directory
JOB_ID="${SLURM_JOB_ID:-$$}"  # Use PID as fallback if not in SLURM
SCRATCH_JOB_DIR="${SCRATCH_BASE}/${JOB_ID}"
if [ ! -d "${SCRATCH_JOB_DIR}" ]; then
    mkdir -p "${SCRATCH_JOB_DIR}" || {
        echo "Error: Failed to create job scratch directory ${SCRATCH_JOB_DIR}" >&2
        exit 1
    }
fi

# Create local output subdirectories matching the network structure
mkdir -p "${SCRATCH_JOB_DIR}/mmpose" || exit 1
mkdir -p "${SCRATCH_JOB_DIR}/pyscenedetect" || exit 1

# Copy input file to scratch
ORIGINAL_BASENAME=$(basename "${input_file}" .mp4)
SCRATCH_INPUT_FILE="${SCRATCH_JOB_DIR}/original_$(basename "${input_file}")"
echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Copying input file to scratch storage..."
cp "${input_file}" "${SCRATCH_INPUT_FILE}" || {
    echo "Error: Failed to copy input file to scratch: ${input_file} -> ${SCRATCH_INPUT_FILE}" >&2
    exit 1
}

# Update input_file to point to scratch copy
input_file="${SCRATCH_INPUT_FILE}"
echo "Using scratch input file: ${input_file}"

# Scratch-based output paths (for processing)
OUTPUT_MMPOSE="${SCRATCH_JOB_DIR}/mmpose"
OUTPUT_PYSCENE="${SCRATCH_JOB_DIR}/pyscenedetect"

# Prepare per-file variables
# base_name=$(basename "$input_file" .mp4)
base_name=$ORIGINAL_BASENAME

# Define expected mmpose outputs (network paths for checking)
MMPOSE_JSON_FINAL="${OUTPUT_MMPOSE_FINAL}/results_smooth_${base_name}.json"
MMPOSE_JSON_HYBRID_FINAL="${OUTPUT_MMPOSE_FINAL}/results_smooth_hybrid_${base_name}.json"
MMPOSE_JSON_17_FINAL="${OUTPUT_MMPOSE_FINAL}/results_smooth_${base_name}_17.json"
MMPOSE_JSON_133_FINAL="${OUTPUT_MMPOSE_FINAL}/results_smooth_${base_name}_133.json"

# Define scratch paths for processing. RTMW omits _smooth from the file name but we'll use it in network storage later
MMPOSE_JSON="${OUTPUT_MMPOSE}/results_smooth_${base_name}.json"
MMPOSE_JSON_HYBRID="${OUTPUT_MMPOSE}/results_${base_name}.json"
MMPOSE_JSON_17="${OUTPUT_MMPOSE}/results_${base_name}_17.json"
MMPOSE_JSON_133="${OUTPUT_MMPOSE}/results_${base_name}_133.json"

# Define downsampled file path based on original input file name
downsampled_file="${SCRATCH_JOB_DIR}/${ORIGINAL_BASENAME}.mp4"

# Flag to track whether we created a downsampled copy
DOWNSAMPLED_CREATED=0

# Check if downsampled file exists; if not, run ffmpeg to create it
if [ -f "${downsampled_file}" ]; then
    echo "Downsampled file already exists: ${downsampled_file}"
else
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Creating downsampled video for input file..."
    conda activate "${CONDA_ENV_RTMW}" #rtmw environment includes required ffmpeg install
    # Get input video vertical resolution using ffprobe
    input_height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height \
        -of csv=p=0 "${input_file}")

    if [ "${input_height:-0}" -le 360 ]; then
        echo "Input resolution ${input_height}px is <= 360. Copying file instead of downsampling."
        cp "${input_file}" "${downsampled_file}" || {
            echo "Error: Failed to copy input file to ${downsampled_file}" >&2
            exit 1
        }
    else
        echo "Input resolution ${input_height}px is > 360. Running ffmpeg downsampling."
        ffmpeg_downsample_cmd=(ffmpeg -i "${input_file}" -vf "scale=-2:min(360\,ih)" -c:v libx264 -preset fast -crf 23 -an -y "${downsampled_file}")
        print_cmd_array ffmpeg_downsample_cmd
        run_cmd ffmpeg_downsample_cmd
        if [ $? -ne 0 ]; then
            echo "Error: ffmpeg failed to create downsampled video." >&2
            exit 1
        fi
    fi
    DOWNSAMPLED_CREATED=1
fi

# Use the downsampled file as the new input for subsequent steps
input_file="${downsampled_file}"
echo "Using file: ${input_file}"

# Helper: copy files from scratch to network storage
copy_to_network() {
    local scratch_file="$1"
    local network_file="$2"
    local description="$3"
    
    if [ -f "${scratch_file}" ]; then
        echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Copying ${description} to network storage..."
        mkdir -p "$(dirname "${network_file}")" || return 1
        cp "${scratch_file}" "${network_file}" || {
            echo "Warning: Failed to copy ${description}: ${scratch_file} -> ${network_file}" >&2
            return 1
        }
        echo "Successfully copied: ${network_file}"
        return 0
    else
        echo "Warning: ${description} not found in scratch: ${scratch_file}" >&2
        return 1
    fi
}

# Helper: trap exits and move the created log files to meaningful names
if [ -z "${ENV_DS_LOGPATH:-}" ]; then
    # Just print a message on exit saying we didn't attempt to rename because no log path was passed in.
    cleanup_noop() {
        local rc=$?
        local now
        now="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        printf '%s cleanup: ENV_DS_LOGPATH not set; leaving sbatch-managed logs in place (exit code=%s)\n' "$now" "$rc" >&2
        return "$rc"
    }
    trap cleanup_noop EXIT HUP INT TERM PIPE
else
    # Use ENV_DS_LOGPATH as the directory where sbatch placed output_%a_log.txt and error_%a_log.txt
    LOG_DIR="${ENV_DS_LOGPATH}"
    mkdir -p "${LOG_DIR}" 2>/dev/null || true

    ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
    SB_OUT="${LOG_DIR}/output_${ARRAY_ID}_log.txt"
    SB_ERR="${LOG_DIR}/error_${ARRAY_ID}_log.txt"

    NEW_OUT="${LOG_DIR}/${base_name}_log_output.txt"
    NEW_ERR="${LOG_DIR}/${base_name}_log_error.txt"

    mv_with_retry() {
        local src="$1"; shift
        local dst="$1"; shift
        local tries=5
        local wait=1
        local i=0
        [ -e "${src}" ] || return 0
        while [ "${i}" -lt "${tries}" ]; do
            if mv -f "${src}" "${dst}"; then
                return 0
            fi
            i=$((i+1))
            sleep "${wait}"
            wait=$((wait * 2))
        done
        return 1
    }

    cleanup_and_rename_logs() {
        local rc=$?
        local now
        now="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        
        # Clean up scratch directory unless told otherwise
        if [ "${ENV_DS_CLEANUP:-0}" = "1" ]; then
            if [ -n "${SCRATCH_JOB_DIR:-}" ] && [[ "${SCRATCH_JOB_DIR}" == /scratch/deepscreens/* ]]; then
                echo "${now} cleanup: Removing scratch directory: ${SCRATCH_JOB_DIR}"
                rm -rf "${SCRATCH_JOB_DIR}" || {
                    echo "${now} cleanup: WARNING: failed to remove scratch directory: ${SCRATCH_JOB_DIR}" >&1
                }
            fi
        else
            echo "Scratch directory cleanup skipped."
        fi

        printf '%s cleanup: exit code=%s\n' "$now" "$rc" >&1
        printf '%s cleanup: attempting to move sbatch logs in %s\n' "$now" "${LOG_DIR}" >&1
        if mv_with_retry "${SB_OUT}" "${NEW_OUT}"; then
            printf '%s cleanup: moved %s -> %s\n' "$now" "${SB_OUT}" "${NEW_OUT}" >&1
        else
            printf '%s cleanup: WARNING: failed to move %s -> %s\n' "$now" "${SB_OUT}" "${NEW_OUT}" >&1
        fi
        if mv_with_retry "${SB_ERR}" "${NEW_ERR}"; then
            printf '%s cleanup: moved %s -> %s\n' "$now" "${SB_ERR}" "${NEW_ERR}" >&1
        else
            printf '%s cleanup: WARNING: failed to move %s -> %s\n' "$now" "${SB_ERR}" "${NEW_ERR}" >&1
        fi
        return "$rc"
    }

    trap cleanup_and_rename_logs EXIT HUP INT TERM PIPE
fi

mmpose_cmd=(python -u "${ROOT_MMPOSE}/demo/merged_2d_3d.py" \
    "${ROOT_MMPOSE}/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py" \
    "${SCRATCH_MODELS}/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth" \
    "${ROOT_MMPOSE}/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py" \
    "${SCRATCH_MODELS}/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth" \
    "${ROOT_MMPOSE}/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py" \
    "${SCRATCH_MODELS}/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth" \
    --input "${input_file}" \
    --output-root "${OUTPUT_MMPOSE}" \
    --save-predictions \
    --save-meta-info \
    --meta-info-root "${META_INFO_ROOT}")

if [ "${ENV_DS_VIDEO:-0}" = "1" ]; then
    mmpose_cmd+=( --save-video )
fi

rtmw_cmd=(python -u "${ROOT_DEEPSCREENS}/emma_deepscreens_rtmw.py" \
    "${ROOT_RTMW}/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py" \
    "${SCRATCH_MODELS}/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth" \
    "${ROOT_RTMW}/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py" \
    "${SCRATCH_MODELS}/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth" \
    --input "${input_file}" \
    --output-root "${OUTPUT_MMPOSE}" \
    --save-predictions \
    --save-meta-info \
    --meta-info-root "${META_INFO_ROOT}" \
    --kpt-thr "0.5" \
    #--use-oks-tracking \
    --tracking-thr "0.5")

if [ "${ENV_DS_VIDEO:-0}" = "1" ]; then
    rtmw_cmd+=( --save-video )
fi

hybrid_cmd=(python -u "${ROOT_DEEPSCREENS}/emma_deepscreens_rtmw_videopose.py" \
    #detection config
    "${ROOT_RTMW}/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py" \
    #detection weights
    "${SCRATCH_MODELS}/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth" \
    #2d pose config
    "${ROOT_RTMW}/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail13/rtmw-l_8xb320-270e_cocktail14-384x288.py" \
    #2d pose weights
    "${SCRATCH_MODELS}/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth" \
    #3d pose config
    "${ROOT_RTMW}/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py" \
    #3d pose weights
    "${SCRATCH_MODELS}/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth" \
    --input "${input_file}" \
    --output-root "${OUTPUT_MMPOSE}" \
    --save-predictions \
    --num-instances 1 \
    --kpt-thr "0.5" \
    --meta-info-root "${META_INFO_ROOT}" \
    --smooth-2d "${ENV_DS_SMOOTH_2D:-0}" \
    --smooth-3d "${ENV_DS_SMOOTH_3D:-0}" \
    --source-file-name "${ORIGINAL_BASENAME}.mp4" \
    #--use-oks-tracking \
    --tracking-thr "0.5")

if [ "${ENV_DS_VIDEO:-0}" = "1" ]; then
    hybrid_cmd+=( --save-video )
fi

convert_cmd=(python -u "${ROOT_DEEPSCREENS}/convert_to_h36m.py" \
    "${OUTPUT_MMPOSE}/results_${base_name}.json" \
    "${OUTPUT_MMPOSE}/results_${base_name}_17.json")

pyscenedetect_cmd=("/isi/mep-rip/working/shared_miniconda/pyscenedetect/bin/scenedetect" \
    --input "${downsampled_file}" \
    --output "${OUTPUT_PYSCENE}" \
    --drop-short-scenes \
    detect-adaptive \
    --min-scene-len 30 \
    save-images \
    --num-images 1 \
    --width 200 \
    --quality 60 \
    --filename '$VIDEO_NAME-frame-$FRAME_NUMBER' \
    split-video \
    --filename 'segment_$SCENE_NUMBER' \
    list-scenes )

divide_cmd=(python -u "${ROOT_DEEPSCREENS}/divideByCutDetector_one.py" \
    --cut "${OUTPUT_PYSCENE}/${base_name}-Scenes.csv" \
    --min-frames "${ENV_DS_MIN_FRAMES}" \
    --output "${OUTPUT_JSON_FINAL}")

if [ "${ENV_DS_MODEL:-}" = "hybrid" ]; then
    divide_cmd+=( --mmpose "${MMPOSE_JSON_HYBRID}" )
elif [ "${ENV_DS_MODEL:-}" = "mmpose" ]; then
    divide_cmd+=( --mmpose "${MMPOSE_JSON}" )
fi

divide_cmd_133=(python -u "${ROOT_DEEPSCREENS}/divideByCutDetector_one.py" \
    --cut "${OUTPUT_PYSCENE}/${base_name}-Scenes.csv" \
    --mmpose "${OUTPUT_MMPOSE}/results_${base_name}_133.json" \
    --min-frames "${ENV_DS_MIN_FRAMES}" \
    --output "${OUTPUT_JSON_FINAL}")

divide_cmd_17=(python -u "${ROOT_DEEPSCREENS}/divideByCutDetector_one.py" \
    --cut "${OUTPUT_PYSCENE}/${base_name}-Scenes.csv" \
    --mmpose "${OUTPUT_MMPOSE}/results_${base_name}_17.json" \
    --min-frames "${ENV_DS_MIN_FRAMES}" \
    --output "${OUTPUT_JSON_FINAL}")

# If ENV_DS_MODEL is explicitly set to "rtmw", append the --rtmw flag so the python script
# runs in RTMW mode (it will remap keypoints -> keypoints3d as requested).
if [ "${ENV_DS_MODEL:-}" = "rtmw" ]; then
    divide_cmd+=( --rtmw )
fi

#
# Dry-run (print-exact-commands) branch
#

# Helper: print a safe srun example line that sources conda.sh, activates the given env,
# and runs the provided command array. This is intended for copy/paste into a shell.
# Arguments:
#   $1 = conda_env (may be empty)
#   $2 = label (for display)
#   rest = command array elements
print_srun_example() {
    local conda_env="$1"; shift
    local label="$1"; shift
    local -a cmd_array=( "$@" )

    # Build the inner command (the command that will run on the node).
    # We'll construct it as: source '<CONDA_SH>'; conda activate '<env>'; exec 'arg1' 'arg2' ...
    # Use single-quoting for each argument to protect spaces/quotes. Outer wrapper uses double quotes.
    local inner_cmd_parts=()

    if [ -n "${CONDA_SH:-}" ] && [ -f "${CONDA_SH}" ]; then
        # Use single quotes for the path so it is safe inside double quotes
        inner_cmd_parts+=( "source '${CONDA_SH}';" )
    fi
    if [ -n "$conda_env" ]; then
        inner_cmd_parts+=( "conda activate '${conda_env}';" )
    fi

    # Start exec and append each argument as a single-quoted token
    inner_cmd_parts+=( "exec" )
    for a in "${cmd_array[@]}"; do
        # Escape any single quotes inside the argument by closing, escaping, and reopening
        esc=$(printf "%s" "$a" | sed "s/'/'\\\\''/g")
        inner_cmd_parts+=( " '$esc'" )
    done

    # Join parts into a single inner command string
    local inner_cmd=""
    for part in "${inner_cmd_parts[@]}"; do
        inner_cmd+="${part}"
    done

    # Build srun resource string to match SBATCH header
    local srun_resources="--gres=gpu:1 --partition=a100 --time=00:30:00 --account=mep"
    # You can override srun_resources if you want different resources for interactive runs.

    # Print a clearly labeled, copy/paste-ready srun example.
    # Wrap the inner command in double quotes so inner single quotes remain literal.
    printf '\nSRUN EXAMPLE (%s):\n' "$label"
    printf 'srun %s bash -lc "%s"\n\n' "$srun_resources" "$inner_cmd"
}


    # Header with relevant environment variables for context
    echo "================== wrapper command listing =================="
    echo "HOSTNAME: $(hostname)"
    echo "JOB_ID: ${SLURM_JOB_ID:-N/A}"
    echo "ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
    echo "NOTIFY_EMAIL: ${NOTIFY_EMAIL:-N/A}"
    echo "DRY_RUN_IN_PYTHON: ${DRY_RUN_IN_PYTHON:-0}"
    echo "ENV_DS_MODEL: ${ENV_DS_MODEL:-mmpose}"
    echo "ENV_DS_QUIET: ${ENV_DS_QUIET}"
    echo "ENV_DS_CLEANUP: ${ENV_DS_CLEANUP}"
    echo "ENV_DS_ZIPONLY: ${ENV_DS_ZIPONLY:-0}"
    echo "ENV_DS_LOGPATH: ${ENV_DS_LOGPATH}"
    echo "ENV_DS_MIN_FRAMES: ${ENV_DS_MIN_FRAMES}"
    echo "CUTFILES delimiter: ASCII RS (0x1E)"
    echo "Target input file (mapped): $input_file"
    echo "CONDA_SH: ${CONDA_SH:-N/A}"
    echo "CONDA_ENV_PYSCENE: ${CONDA_ENV_PYSCENE:-N/A}"
    if [ "${ENV_DS_MODEL:-mmpose}" = "rtmw" ]; then
        echo "CONDA_ENV_MODEL (selected): ${CONDA_ENV_RTMW:-N/A}"
        echo "CONVERT_CMD (added for RTMW):"
        print_cmd_array convert_cmd
        print_srun_example "${CONDA_ENV_MMPOSE}" "convert" "${convert_cmd[@]}"
    else
        echo "CONDA_ENV_MODEL (selected): ${CONDA_ENV_MMPOSE:-N/A}"
    fi
    if [ "${ENV_DS_MODEL:-mmpose}" = "none" ]; then
        echo "ENV_DS_MODEL Set to 'none' - skipping mmpose/rtmw and divide by cut"
    fi
    echo "------------------------------------------------------------------"

    # Print either mmpose or rtmw command depending on ENV_DS_MODEL
    if [ "${ENV_DS_MODEL:-mmpose}" = "rtmw" ]; then
        echo "RTMW COMMAND:"
        print_cmd_array rtmw_cmd
        print_srun_example "${CONDA_ENV_RTMW}" "rtmw" "${rtmw_cmd[@]}"
    elif [ "${ENV_DS_MODEL:-mmpose}" = "hybrid" ]; then
        echo "HYBRID COMMAND:"
        print_cmd_array hybrid_cmd
        print_srun_example "${CONDA_ENV_RTMW}" "hybrid" "${hybrid_cmd[@]}"
    else
        echo "MMPOSE COMMAND:"
        print_cmd_array mmpose_cmd
        print_srun_example "${CONDA_ENV_MMPOSE}" "mmpose" "${mmpose_cmd[@]}"
    fi
    echo

    echo "PYSCENEDETECT COMMAND:"
    print_cmd_array pyscenedetect_cmd
    print_srun_example "${CONDA_ENV_PYSCENE}" "pyscenedetect" "${pyscenedetect_cmd[@]}"
    echo

    echo "DIVIDE-BY-CUT COMMANDS ARE PROVISIONAL - SEE BELOW FOR FINAL VERSIONS"
    if [ "${ENV_DS_MODEL:-mmpose}" = "rtmw" ]; then
        echo "DIVIDE-BY-CUT 133 COMMAND:"
        print_cmd_array divide_cmd_133
        print_srun_example "${CONDA_ENV_PYSCENE}" "divide_133" "${divide_cmd_133[@]}"
        echo "DIVIDE-BY-CUT 17 COMMAND:"
        print_cmd_array divide_cmd_17
        print_srun_example "${CONDA_ENV_PYSCENE}" "divide_17" "${divide_cmd_17[@]}"
    else
        echo "DIVIDE-BY-CUT COMMAND:"
        print_cmd_array divide_cmd
        print_srun_example "${CONDA_ENV_PYSCENE}" "divide" "${divide_cmd[@]}"
    fi
    echo "================== end wrapper commands =================="
    echo ""
if [ "${DRY_RUN_IN_PYTHON:-0}" = "1" ]; then
    echo "Dry run selected. Ending script without executing commands."
    exit 0
fi

#
# Normal execution path (re-use same arrays)
#

echo "Processing file: $input_file"
echo "Base name: $base_name"

# Activate conda.sh (required to use conda activate)
if [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
else
    echo "Error: conda.sh not found at ${CONDA_SH}" >&2
    exit 1
fi

# --- PYSCENEDETECT step ---
PYSCENE_CSV_FINAL="${OUTPUT_PYSCENE_FINAL}/${base_name}-Scenes.csv"
echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Checking pyscenedetect output..."
if [ -f "${PYSCENE_CSV_FINAL}" ]; then
    echo "pyscenedetect: Scenes CSV already exists: ${PYSCENE_CSV_FINAL}"
    # Copy existing file to scratch for potential use by later steps
    cp "${PYSCENE_CSV_FINAL}" "${OUTPUT_PYSCENE}/${base_name}-Scenes.csv" || {
        echo "Warning: Failed to copy existing scene detection file to scratch" >&2
    }
    echo "Skipping pyscenedetect step."

    # Copy existing segment MP4s from network storage to scratch pyscenedetect dir
    echo "Copying existing segment MP4s to scratch..."
    seg_copied=0
    for seg_mp4 in "${OUTPUT_MMPOSE_FINAL}/${base_name}"/segment_*.mp4; do
        if [ -f "${seg_mp4}" ]; then
            cp "${seg_mp4}" "${OUTPUT_PYSCENE}/" || {
                echo "Warning: Failed to copy $(basename ${seg_mp4}) to scratch" >&2
            }
            ((seg_copied++))
        fi
    done
    echo "Copied ${seg_copied} segment MP4(s) to scratch."
else
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') pyscenedetect: Scenes CSV missing. Will run pyscenedetect."
    conda activate "${CONDA_ENV_PYSCENE}"
    if run_cmd pyscenedetect_cmd; then
        echo "pyscenedetect: Completed successfully."

        scenes_subdir="${OUTPUT_PYSCENE_FINAL}/${base_name}-scenes"
        mkdir -p "${scenes_subdir}" || {
            echo "Error: Failed to create scenes subdirectory: ${scenes_subdir}" >&2
            exit 1
        }

        # Copy output back to network storage
        copy_to_network "${OUTPUT_PYSCENE}/${base_name}-Scenes.csv" "${scenes_subdir}/${base_name}-Scenes.csv" "scene detection CSV"

        # Copy all JPG files to the subdirectory
        jpg_count=0
        for jpg_file in "${OUTPUT_PYSCENE}"/*.jpg; do
            if [ -f "${jpg_file}" ]; then
                jpg_basename=$(basename "${jpg_file}")
                copy_to_network "${jpg_file}" "${scenes_subdir}/${jpg_basename}" "scene image ${jpg_basename}"
                ((jpg_count++))
            fi
        done
        
        if [ ${jpg_count} -eq 0 ]; then
            echo "Warning: No JPG files found in ${OUTPUT_PYSCENE}" >&2
        else
            echo "Successfully copied ${jpg_count} JPG file(s) to ${scenes_subdir}"
        fi

        # Rename segment files from segment_[sceneNumber].mp4 to segment_[startFrame]_[endFrame].mp4
        # Parse the CSV to get frame ranges for each scene
        echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Renaming segment files using frame numbers from CSV..."
        csv_file="${OUTPUT_PYSCENE}/${base_name}-Scenes.csv"

        if [ ! -f "${csv_file}" ]; then
            echo "Error: Scene CSV file not found: ${csv_file}" >&2
            exit 1
        fi

        # Create a temporary directory to stage renamed files
        # This prevents glob pattern from matching already-renamed files
        tmp_rename_dir="${OUTPUT_PYSCENE}/.rename_tmp_$$"
        mkdir -p "${tmp_rename_dir}"

        # Read CSV and rename files (skip header lines)
        rename_count=0
        while IFS=',' read -r scene_num start_frame _st_ sts end_frame _rest; do
            # Skip header lines and empty lines
            if [[ "${scene_num}" =~ ^[0-9]+$ ]]; then
                # Find matching segment file
                found_file=""
                for file in "${OUTPUT_PYSCENE}"/segment_*.mp4; do
                    [ -f "${file}" ] || continue
                    filename=$(basename "${file}" .mp4)
                    file_scene_num="${filename#segment_}"
                    
                    # Only process original files (pure numbers, not already renamed)
                    if [[ "${file_scene_num}" =~ ^[0-9]+$ ]]; then
                        file_scene_num=$((10#${file_scene_num}))
                        if [ "${file_scene_num}" -eq "${scene_num}" ]; then
                            found_file="${file}"
                            break
                        fi
                    fi
                done
                
                if [ -n "${found_file}" ]; then
                    # Move to temp directory with new name
                    new_name="segment_${start_frame}_${end_frame}.mp4"
                    if mv "${found_file}" "${tmp_rename_dir}/${new_name}"; then
                        echo "Renamed: $(basename "${found_file}") -> ${new_name}"
                        ((rename_count++))
                    else
                        echo "Warning: Failed to rename ${found_file}" >&2
                    fi
                else
                    echo "Warning: No segment file found for scene ${scene_num}" >&2
                fi
            fi
        done < "${csv_file}"

        # Move all renamed files back to original directory
        mv "${tmp_rename_dir}"/*.mp4 "${OUTPUT_PYSCENE}/" 2>/dev/null
        rmdir "${tmp_rename_dir}"

        echo "Renamed ${rename_count} segment file(s)"



        # echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Renaming segment files using frame numbers from CSV..."
        # csv_file="${OUTPUT_PYSCENE}/${base_name}-Scenes.csv"
        
        # if [ ! -f "${csv_file}" ]; then
        #     echo "Error: Scene CSV file not found: ${csv_file}" >&2
        #     exit 1
        # fi
        
        # # Read CSV and rename files (skip header lines)
        # # CSV format: Scene Number, Start Frame, ..., End Frame, ...
        # # Data starts at line 3 (after header line and column names)
        # rename_count=0
        # while IFS=',' read -r scene_num start_frame _st_ sts end_frame _rest; do
        #     # Skip header lines and empty lines
        #     if [[ "${scene_num}" =~ ^[0-9]+$ ]]; then
        #         # Find the file with any zero-padding for this scene number
        #         found_file=""
        #         for file in "${OUTPUT_PYSCENE}"/segment_*.mp4; do
        #             if [ -f "${file}" ]; then
        #                 filename=$(basename "${file}" .mp4)
        #                 file_scene_num="${filename#segment_}"
        #                 # Remove leading zeros for comparison
        #                 file_scene_num=$((10#${file_scene_num}))
        #                 if [ "${file_scene_num}" -eq "${scene_num}" ]; then
        #                     found_file="${file}"
        #                     break
        #                 fi
        #             fi
        #         done
                
        #         if [ -n "${found_file}" ]; then
        #             new_name_scratch="${OUTPUT_PYSCENE}/segment_${start_frame}_${end_frame}.mp4"
        #             mv "${found_file}" "${new_name_scratch}" || {
        #                 echo "Warning: Failed to rename ${found_file} -> ${new_name_scratch}" >&2
        #             }
        #             echo "Renamed: $(basename "${found_file}") -> segment_${start_frame}_${end_frame}.mp4"
        #             ((rename_count++))
        #         else
        #             echo "Warning: No segment file found for scene ${scene_num}" >&2
        #         fi
        #     fi
        # done < "${csv_file}"

        # echo "Renamed ${rename_count} segment file(s)"

    else
        echo "pyscenedetect: FAILED." >&2
        exit 1
    fi
fi

# --- MMPOSE / RTMW step ---
# if [ "${ENV_DS_MODEL:-mmpose}" = "rtmw" ]; then
#     export PYTHONPATH="${ROOT_RTMW}/projects/rtmpose3d:${PYTHONPATH}"
#     echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') RTMW selected. Checking RTMW outputs..."

#     echo "=== RTMW FILE CHECK DEBUG ==="
#     echo "MMPOSE_JSON_17_FINAL: ${MMPOSE_JSON_17_FINAL}"
#     echo "File exists: $([ -f "${MMPOSE_JSON_17_FINAL}" ] && echo "YES" || echo "NO")"
#     if [ -f "${MMPOSE_JSON_17_FINAL}" ]; then
#         echo "File size: $(ls -lh "${MMPOSE_JSON_17_FINAL}")"
#         echo "File permissions: $(ls -l "${MMPOSE_JSON_17_FINAL}")"
#     fi
#     echo "MMPOSE_JSON_133_FINAL: ${MMPOSE_JSON_133_FINAL}"  
#     echo "133-file exists: $([ -f "${MMPOSE_JSON_133_FINAL}" ] && echo "YES" || echo "NO")"
#     echo "=== END FILE CHECK DEBUG ==="


#     # RTMW will produce the 133-keypoint json, then convert to 17-keypoint json
#     if [ -f "${MMPOSE_JSON_17_FINAL}" ]; then
#         echo "rtmw: Final mmpose 17-keypoint JSON already exists: ${MMPOSE_JSON_17_FINAL}"
#         # Copy existing files to scratch for potential use by later steps
#         if [ -f "${MMPOSE_JSON_133_FINAL}" ]; then
#             cp "${MMPOSE_JSON_133_FINAL}" "${MMPOSE_JSON_133}" || {
#                 echo "Warning: Failed to copy existing 133-keypoint file to scratch" >&2
#             }
#         fi
#         cp "${MMPOSE_JSON_17_FINAL}" "${MMPOSE_JSON_17}" || {
#             echo "Warning: Failed to copy existing 17-keypoint file to scratch" >&2
#         }
#         echo "Skipping rtmw and conversion steps."
#         RTMW_RAN=0
#     else
#         echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') rtmw: 17-keypoint output missing. Will run RTMW."
#         echo "=== RTMW DEBUG INFO ==="
#         echo "RTMW command about to execute:"
#         print_cmd_array rtmw_cmd
#         echo "Current working directory: $(pwd)"
#         echo "Input file: ${input_file}"
#         echo "Input file exists: $([ -f "${input_file}" ] && echo "YES" || echo "NO")"
#         echo "Input file size: $([ -f "${input_file}" ] && ls -lh "${input_file}" || echo "N/A")"
#         echo "Output directory: ${OUTPUT_MMPOSE}"
#         echo "Output directory exists: $([ -d "${OUTPUT_MMPOSE}" ] && echo "YES" || echo "NO")"
#         echo "Output directory writable: $([ -w "${OUTPUT_MMPOSE}" ] && echo "YES" || echo "NO")"
#         echo "Expected output file: ${OUTPUT_MMPOSE}/results_${base_name}.json"

#         conda activate "${CONDA_ENV_RTMW}"
#         echo "After conda activate:"
#         echo "  Conda env: ${CONDA_DEFAULT_ENV:-none}"
#         echo "  Python executable: $(which python)"
#         echo "  Python version: $(python --version 2>&1)"
#         echo "=== CONDA ENVIRONMENT VERIFICATION ==="
#         echo "CONDA_ENV_RTMW path: ${CONDA_ENV_RTMW}"
#         echo "Conda env directory exists: $([ -d "${CONDA_ENV_RTMW}" ] && echo "YES" || echo "NO")"
#         echo "Active conda env: ${CONDA_DEFAULT_ENV:-none}"
#         echo "Python path: $(which python)"
#         echo "PYTHONPATH: ${PYTHONPATH:-empty}"

#         # Test if the python script exists and is readable
#         echo "Python script accessibility:"
#         echo "  Script exists: $([ -f '/isi/mep-rip/working/deepscreens/rtmw/projects/rtmpose3d/demo/body3d_img2pose_demo_ds.py' ] && echo "YES" || echo "NO")"

#         echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Starting RTMW execution..."

#         # Check what files were actually created
#         echo "Files in output directory after RTMW:"
#         ls -la "${OUTPUT_MMPOSE}/" 2>/dev/null || echo "Output directory not accessible"

#         echo "Files matching pattern 'results_*' in output directory:"
#         find "${OUTPUT_MMPOSE}/" -name "results_*" -type f 2>/dev/null || echo "No matching files found"

#         if run_cmd rtmw_cmd; then
#             echo "rtmw: Completed successfully."
#             if [ -f "${OUTPUT_MMPOSE}/results_${base_name}.json" ]; then
#                 echo "Expected output file created successfully."
#             else
#                 echo "ERROR: RTMW completed but expected output file not found!" >&2
#                 echo "Expected: ${OUTPUT_MMPOSE}/results_${base_name}.json" >&2
#                 exit 1
#             fi
#             # Run convert only if it produces the expected files
#             echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') rtmw: Running convert_to_h36m to produce 17-keypoint JSON..."
#             echo "Convert command about to execute:"
#             print_cmd_array convert_cmd
#             echo "Current working directory: $(pwd)"
#             echo "Current conda environment: ${CONDA_DEFAULT_ENV:-none}"
#             echo "Python path: $(which python)"
#             echo "Input file exists: $([ -f "${OUTPUT_MMPOSE}/results_${base_name}.json" ] && echo "YES" || echo "NO")"
#             echo "Input file size: $([ -f "${OUTPUT_MMPOSE}/results_${base_name}.json" ] && ls -lh "${OUTPUT_MMPOSE}/results_${base_name}.json" || echo "N/A")"
#             echo "Output directory writable: $([ -w "${OUTPUT_MMPOSE}" ] && echo "YES" || echo "NO")"
#             if run_cmd convert_cmd; then
#             # if "${convert_cmd[@]}" 2>&1; then
#                 echo "convert_to_h36m: Completed successfully."
#                 # move original to _133.json to disambiguate
#                 if [ -f "${OUTPUT_MMPOSE}/results_${base_name}.json" ]; then
#                     mv "${OUTPUT_MMPOSE}/results_${base_name}.json" "${MMPOSE_JSON_133}" || {
#                         echo "Warning: failed to mv ${OUTPUT_MMPOSE}/results_${base_name}.json -> ${MMPOSE_JSON_133}"
#                     }
#                 fi
#                 # Copy outputs back to network storage
#                 copy_to_network "${MMPOSE_JSON_133}" "${MMPOSE_JSON_133_FINAL}" "rtmw 133-keypoint results"
#                 copy_to_network "${MMPOSE_JSON_17}" "${MMPOSE_JSON_17_FINAL}" "rtmw 17-keypoint results"
                
#                 # Check for optional visualization video
#                 RTMW_VIDEO_SCRATCH="${OUTPUT_MMPOSE}/${ORIGINAL_BASENAME}.mp4"
#                 RTMW_VIDEO_FINAL="${OUTPUT_MMPOSE_FINAL}/${ORIGINAL_BASENAME}.mp4"
#                 if [ -f "${RTMW_VIDEO_SCRATCH}" ]; then
#                     echo "RTMW visualization video found, copying to network storage..."
#                     copy_to_network "${RTMW_VIDEO_SCRATCH}" "${RTMW_VIDEO_FINAL}" "rtmw visualization video"
#                 else
#                     echo "No RTMW visualization video found (this is normal if --save-video was not used)."
#                 fi
                
#                 RTMW_RAN=1
#             else
#                 convert_exit_code=$?
#                 echo "convert_to_h36m: FAILED with exit code ${convert_exit_code}." >&2
#                 echo "Convert command that failed:" >&2
#                 print_cmd_array convert_cmd >&2
#                 echo "Files in output directory:" >&2
#                 ls -la "${OUTPUT_MMPOSE}/" >&2
#                 exit 1
#             fi
#         else
#             echo "rtmw: FAILED." >&2
#             exit 1
#         fi
#     fi
# elif [ "${ENV_DS_MODEL:-mmpose}" = "hybrid" ]; then
    # export PYTHONPATH="${ROOT_RTMW}:${PYTHONPATH}"
    # echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Hybrid selected. Checking Hybrid outputs..."

    # echo "=== HYBRID FILE CHECK DEBUG ==="
    # echo "MMPOSE_JSON_HYBRID_FINAL: ${MMPOSE_JSON_HYBRID_FINAL}"
    # echo "File exists: $([ -f "${MMPOSE_JSON_HYBRID_FINAL}" ] && echo "YES" || echo "NO")"
    # if [ -f "${MMPOSE_JSON_HYBRID_FINAL}" ]; then
    #     echo "File size: $(ls -lh "${MMPOSE_JSON_HYBRID_FINAL}")"
    #     echo "File permissions: $(ls -l "${MMPOSE_JSON_HYBRID_FINAL}")"
    # fi
    # echo "=== END FILE CHECK DEBUG ==="

    # # Hybrid model runs rtmw2d first and then uplifts using videopose 3d
    # if [ -f "${MMPOSE_JSON_HYBRID_FINAL}" ]; then
    #     echo "hybrid: Final JSON already exists: ${MMPOSE_JSON_HYBRID_FINAL}"
    #     # Copy existing files to scratch for potential use by later steps
    #     cp "${MMPOSE_JSON_HYBRID_FINAL}" "${MMPOSE_JSON_HYBRID}" || {
    #         echo "Warning: Failed to copy existing hybrid file to scratch" >&2
    #     }
    #     echo "Skipping hybrid pose analysis steps."
    #     HYBRID_RAN=0
    # else
    #     echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') hybrid: pose keypoint output missing. Will run hybrid models."
    #     echo "=== HYBRID DEBUG INFO ==="
    #     echo "HYBRID command about to execute:"
    #     print_cmd_array hybrid_cmd
    #     echo "Current working directory: $(pwd)"
    #     echo "Input file: ${input_file}"
    #     echo "Input file exists: $([ -f "${input_file}" ] && echo "YES" || echo "NO")"
    #     echo "Input file size: $([ -f "${input_file}" ] && ls -lh "${input_file}" || echo "N/A")"
    #     echo "Output directory: ${OUTPUT_MMPOSE}"
    #     echo "Output directory exists: $([ -d "${OUTPUT_MMPOSE}" ] && echo "YES" || echo "NO")"
    #     echo "Output directory writable: $([ -w "${OUTPUT_MMPOSE}" ] && echo "YES" || echo "NO")"
    #     echo "Expected output file: ${OUTPUT_MMPOSE}/results_hybrid_${base_name}.json"

    #     conda activate "${CONDA_ENV_RTMW}"
    #     echo "After conda activate:"
    #     echo "  Conda env: ${CONDA_DEFAULT_ENV:-none}"
    #     echo "  Python executable: $(which python)"
    #     echo "  Python version: $(python --version 2>&1)"
    #     echo "=== CONDA ENVIRONMENT VERIFICATION ==="
    #     echo "CONDA_ENV_RTMW path: ${CONDA_ENV_RTMW}"
    #     echo "Conda env directory exists: $([ -d "${CONDA_ENV_RTMW}" ] && echo "YES" || echo "NO")"
    #     echo "Active conda env: ${CONDA_DEFAULT_ENV:-none}"
    #     echo "Python path: $(which python)"
    #     echo "PYTHONPATH: ${PYTHONPATH:-empty}"

    #     echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Starting hybrid execution..."

    #     # Check what files were actually created
    #     echo "Files in output directory after hybrid:"
    #     ls -la "${OUTPUT_MMPOSE}/" 2>/dev/null || echo "Output directory not accessible"

    #     echo "Files matching pattern 'results_*' in output directory:"
    #     find "${OUTPUT_MMPOSE}/" -name "results_*" -type f 2>/dev/null || echo "No matching files found"

    #     if run_cmd hybrid_cmd; then
    #         echo "hybrid: Completed successfully."
    #         if [ -f "${MMPOSE_JSON_HYBRID}" ]; then
    #             echo "Expected output file created successfully."
    #             copy_to_network "${MMPOSE_JSON_HYBRID}" "${MMPOSE_JSON_HYBRID_FINAL}" "hybrid results"
                
    #             # Check for optional visualization video
    #             HYBRID_VIDEO_SCRATCH="${OUTPUT_MMPOSE}/${ORIGINAL_BASENAME}.mp4"
    #             HYBRID_VIDEO_FINAL="${OUTPUT_MMPOSE_FINAL}/${ORIGINAL_BASENAME}.mp4"
    #             if [ -f "${HYBRID_VIDEO_SCRATCH}" ]; then
    #                 echo "Hybrid visualization video found, copying to network storage..."
    #                 copy_to_network "${HYBRID_VIDEO_SCRATCH}" "${HYBRID_VIDEO_FINAL}" "hybrid visualization video"
    #             else
    #                 echo "No hybrid visualization video found (this is normal if --save-video was not used)."
    #             fi
                
    #             HYBRID_RAN=1
    #         else
    #             echo "ERROR: hybrid completed but expected output file not found!" >&2
    #             echo "Expected: ${MMPOSE_JSON_HYBRID}" >&2
    #             exit 1
    #         fi
    #     else
    #         echo "hybrid: FAILED." >&2
    #         exit 1
    #     fi
    # fi
# elif [ "${ENV_DS_MODEL:-mmpose}" = "mmpose" ]; then
#     echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') MMPOSE selected. Checking mmpose outputs..."
#     # mmpose step produces results_smooth_${base_name}.json — treat this as the required artifact
#     if [ -f "${MMPOSE_JSON_FINAL}" ]; then
#         echo "mmpose: Output JSON already exists: ${MMPOSE_JSON_FINAL}"
#         # Copy existing file to scratch for potential use by later steps
#         cp "${MMPOSE_JSON_FINAL}" "${OUTPUT_MMPOSE}/results_smooth_${base_name}.json" || {
#             echo "Warning: Failed to copy existing mmpose file to scratch" >&2
#         }
#         echo "Skipping mmpose step."
#         MMPOSE_RAN=0
#     else
#         echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') mmpose: Output JSON missing. Will run mmpose."
#         conda activate "${CONDA_ENV_MMPOSE}"
#         if run_cmd mmpose_cmd; then
#             echo "mmpose: Completed successfully."
#             # Copy output back to network storage
#             copy_to_network "${OUTPUT_MMPOSE}/results_smooth_${base_name}.json" "${OUTPUT_MMPOSE_FINAL}/results_smooth_${base_name}.json" "mmpose results"
#             MMPOSE_RAN=1
#         else
#             echo "mmpose: FAILED." >&2
#             exit 1
#         fi
#     fi
# fi





# --- PER-SEGMENT POSE ESTIMATION ---
# Process each segment video individually
echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Starting per-segment pose estimation..."

# Find all segment MP4 files in pyscenedetect output directory
segment_files=()
while IFS= read -r -d '' segment_file; do
    segment_files+=("$segment_file")
done < <(find "${OUTPUT_PYSCENE}" -name "segment_*.mp4" -type f -print0 | sort -z)

if [ ${#segment_files[@]} -eq 0 ]; then
    echo "Warning: No segment files found in ${OUTPUT_PYSCENE}" >&2
    echo "Skipping per-segment pose estimation."
else
    echo "Found ${#segment_files[@]} segment file(s) to process."

    segment_network_dir="${OUTPUT_MMPOSE_FINAL}/${base_name}"
    # Copy results to network storage
    mkdir -p "${segment_network_dir}" || {
        echo "  Error: Failed to create network segment directory" >&2
        exit 1
    }

    # Process each segment
    processed_count=0
    skipped_count=0
    
    for segment_file in "${segment_files[@]}"; do
        segment_basename=$(basename "${segment_file}" .mp4)
        echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Processing segment: ${segment_basename}"
        
        # Extract start and end frame numbers from filename: segment_START_END.mp4
        if [[ "${segment_basename}" =~ ^segment_([0-9]+)_([0-9]+)$ ]]; then
            start_frame="${BASH_REMATCH[1]}"
            end_frame="${BASH_REMATCH[2]}"
            frame_count=$((end_frame - start_frame + 1))
            
            echo "  Start frame: ${start_frame}, End frame: ${end_frame}, Frame count: ${frame_count}"
            
            # Check minimum frame threshold
            if [ "${frame_count}" -lt "${ENV_DS_MIN_FRAMES}" ]; then
                echo "  Skipping segment (${frame_count} frames < ${ENV_DS_MIN_FRAMES} minimum)"
                ((skipped_count++))
                continue
            fi

            # Determine which pose estimation model to run
            if [ "${ENV_DS_MODEL:-mmpose}" = "mmpose" ]; then
                conda activate "${CONDA_ENV_MMPOSE}"
            elif [ "${ENV_DS_MODEL:-mmpose}" = "rtmw" ]; then
                export PYTHONPATH="${ROOT_RTMW}/projects/rtmpose3d:${PYTHONPATH}"
                conda activate "${CONDA_ENV_RTMW}"
            elif [ "${ENV_DS_MODEL:-mmpose}" = "hybrid" ]; then
                export PYTHONPATH="${ROOT_RTMW}:${PYTHONPATH}"
                conda activate "${CONDA_ENV_RTMW}"
            fi

            # Run pose estimation on each segment
            if [ "${ENV_DS_MODEL:-mmpose}" = "mmpose" ]; then
                segment_json_final="${segment_network_dir}/results_smooth_${segment_basename}.json"
                
                if [ -f "${segment_json_final}" ]; then
                    echo "  Segment mmpose output already exists: ${segment_json_final}"
                    echo "  Skipping this segment."
                    ((skipped_count++))
                else
                    echo "  Running mmpose on segment..."
                    # Copy the existing mmpose_cmd array and update input parameters
                    segment_mmpose_cmd=("${mmpose_cmd[@]}")
                    
                    # Find and replace the --input parameter value
                    for i in "${!segment_mmpose_cmd[@]}"; do
                        if [ "${segment_mmpose_cmd[$i]}" = "--input" ]; then
                            segment_mmpose_cmd[$((i+1))]="${segment_file}"
                        fi
                    done

                    if run_cmd segment_mmpose_cmd; then
                        echo "  Segment mmpose completed successfully."

                       segment_json_scratch="${OUTPUT_MMPOSE}/results_smooth_${segment_basename}.json"
                        copy_to_network "${segment_json_scratch}" "${segment_json_final}" "segment mmpose results"
                        ((processed_count++))
                    else
                        echo "  Error: Segment mmpose failed for ${segment_basename}" >&2
                        exit 1
                    fi
                fi
            elif [ "${ENV_DS_MODEL:-mmpose}" = "rtmw" ]; then
                segment_json_17_final="${segment_network_dir}/results_${segment_basename}_17.json"
                segment_json_133_final="${segment_network_dir}/results_${segment_basename}_133.json"

                if [ -f "${segment_json_17_final}" ]; then
                    echo "  Segment rtmw output already exists: ${segment_json_17_final}"
                    echo "  Skipping this segment."
                    ((skipped_count++))
                else
                    echo "  Running rtmw on segment..."

                    # Copy the existing rtmw_cmd array and update input/output parameters
                    segment_rtmw_cmd=("${rtmw_cmd[@]}")
                    
                    # Find and replace the --input and --output-root parameter values
                    for i in "${!segment_rtmw_cmd[@]}"; do
                        if [ "${segment_rtmw_cmd[$i]}" = "--input" ]; then
                            segment_rtmw_cmd[$((i+1))]="${segment_file}"
                        fi
                    done

                    if run_cmd segment_rtmw_cmd; then
                        echo "  Segment rtmw completed successfully."
                        
                        # Run conversion to 17-keypoint format
                        segment_convert_cmd=(python -u "${ROOT_DEEPSCREENS}/convert_to_h36m.py" \
                            "${OUTPUT_MMPOSE}/results_${segment_basename}.json" \
                            "${OUTPUT_MMPOSE}/results_${segment_basename}_17.json")

                        if run_cmd segment_convert_cmd; then
                            echo "  Segment conversion completed successfully."
                            
                            # Rename original to _133.json
                            if [ -f "${OUTPUT_MMPOSE}/results_${segment_basename}.json" ]; then
                                mv "${OUTPUT_MMPOSE}/results_${segment_basename}.json" \
                                   "${OUTPUT_MMPOSE}/results_${segment_basename}_133.json"
                            fi
                            
                            # Copy results to network storage
                            copy_to_network "${OUTPUT_MMPOSE}/results_${segment_basename}_133.json" \
                                          "${segment_json_133_final}" "segment rtmw 133-keypoint results"
                            copy_to_network "${OUTPUT_MMPOSE}/results_${segment_basename}_17.json" \
                                          "${segment_json_17_final}" "segment rtmw 17-keypoint results"
                            ((processed_count++))
                        else
                            echo "  Error: Segment conversion failed for ${segment_basename}" >&2
                            exit 1
                        fi
                    else
                        echo "  Error: Segment rtmw failed for ${segment_basename}" >&2
                        exit 1
                    fi
                fi
                
            elif [ "${ENV_DS_MODEL:-mmpose}" = "hybrid" ]; then
            segment_json_final="${segment_network_dir}/results_${segment_basename}.json"

            if [ -f "${segment_json_final}" ]; then
                echo "  Segment hybrid output already exists: ${segment_json_final}"
                echo "  Skipping this segment."
                ((skipped_count++))
            else
                echo "  Running hybrid on segment..."

                segment_hybrid_cmd=("${hybrid_cmd[@]}")

                for i in "${!segment_hybrid_cmd[@]}"; do
                    if [ "${segment_hybrid_cmd[$i]}" = "--input" ]; then
                        segment_hybrid_cmd[$((i+1))]="${segment_file}"
                    fi
                done

                segment_hybrid_cmd+=(
                    --cleaned-json "${segment_network_dir}/${segment_basename}_cleaned.json"
                )
                segment_hybrid_cmd+=( --segment-start-frame )
                segment_hybrid_cmd+=( ${start_frame} )
                segment_hybrid_cmd+=( --segment-end-frame )
                segment_hybrid_cmd+=( ${end_frame} )

                echo "=== HYBRID DEBUG INFO ==="
                echo "HYBRID command about to execute:"
                print_cmd_array segment_hybrid_cmd
                echo "Cleaned JSON path: ${segment_network_dir}/${segment_basename}_cleaned.json"
                echo "Cleaned JSON exists: $([ -f "${segment_network_dir}/${segment_basename}_cleaned.json" ] && echo "YES" || echo "NO")"
                echo "Input segment file: ${segment_file}"
                echo "Input segment exists: $([ -f "${segment_file}" ] && echo "YES" || echo "NO")"
                echo "Output directory: ${OUTPUT_MMPOSE}"

                if run_cmd segment_hybrid_cmd; then
                    echo "  Segment hybrid completed successfully."

                    segment_json_scratch="${OUTPUT_MMPOSE}/results_${segment_basename}.json"
                    copy_to_network "${segment_json_scratch}" "${segment_json_final}" "segment hybrid results"

                    HYBRID_VIDEO_SCRATCH="${OUTPUT_MMPOSE}/${segment_basename}.mp4"
                    HYBRID_VIDEO_FINAL="${segment_network_dir}/${segment_basename}.mp4"
                    if [ -f "${HYBRID_VIDEO_SCRATCH}" ]; then
                        echo "Hybrid visualization video found, copying to network storage..."
                        copy_to_network "${HYBRID_VIDEO_SCRATCH}" "${HYBRID_VIDEO_FINAL}" "hybrid visualization video"
                    else
                        echo "No hybrid visualization video found (this is normal if --save-video was not used)."
                    fi

                    HYBRID_RAN=1
                    ((processed_count++))
                else
                    echo "  Error: Segment hybrid failed for ${segment_basename}" >&2
                    exit 1
                fi
            fi
        fi
            
        else
            echo "  Warning: Could not parse frame numbers from filename: ${segment_basename}" >&2
            ((skipped_count++))
        fi
    done
    
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Per-segment pose estimation complete."
    echo "  Processed: ${processed_count} segment(s)"
    echo "  Skipped: ${skipped_count} segment(s)"
fi













# --- CONSOLIDATE OUTPUTS step ---
echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Consolidating outputs to final directory..."

# Create final output directory
FINAL_OUTPUT_DIR="${OUTPUT_JSON_FINAL}/${base_name}"
FINAL_OUTPUT_DATA_DIR="${FINAL_OUTPUT_DIR}/data"
mkdir -p "${FINAL_OUTPUT_DIR}" || {
    echo "Error: Failed to create final output directory: ${FINAL_OUTPUT_DIR}" >&2
    exit 1
}
mkdir -p "${FINAL_OUTPUT_DATA_DIR}" || {
    echo "Error: Failed to create final output data directory: ${FINAL_OUTPUT_DATA_DIR}" >&2
    exit 1
}

# Track copy statistics
csv_copied=0
jpg_copied=0
mp4_copied=0
json_copied=0

# Copy pyscenedetect outputs
echo "Copying pyscenedetect outputs..."
scenes_subdir="${OUTPUT_PYSCENE_FINAL}/${base_name}-scenes"

# Copy scenes CSV file
if [ -f "${scenes_subdir}/${base_name}-Scenes.csv" ]; then
    cp "${scenes_subdir}/${base_name}-Scenes.csv" "${FINAL_OUTPUT_DATA_DIR}/segments.csv" || {
        echo "Warning: Failed to copy scenes CSV" >&2
    }
    ((csv_copied++))
    echo "  Copied: ${base_name}-scenes.csv"
fi

# Copy and rename scene JPG files from scenes subdirectory
if [ -d "${scenes_subdir}" ]; then
    # First, build a list of segment frame ranges from the CSV
    declare -A segment_ranges  # Maps frame numbers to "start_end" strings
    csv_file="${scenes_subdir}/${base_name}-Scenes.csv"
    
    if [ -f "${csv_file}" ]; then
        # Read CSV and build segment ranges (skip header lines)
        while IFS=',' read -r scene_num start_frame _st_ sts end_frame _rest; do
            # Skip header lines and empty lines
            if [[ "${start_frame}" =~ ^[0-9]+$ ]] && [[ "${end_frame}" =~ ^[0-9]+$ ]]; then
                # Store the range for this segment
                segment_ranges["${start_frame}_${end_frame}"]="${start_frame}_${end_frame}"
            fi
        done < "${csv_file}"
    fi
    
    # Now process each JPG file
    for jpg_file in "${scenes_subdir}"/*.jpg; do
        if [ -f "${jpg_file}" ]; then
            jpg_basename=$(basename "${jpg_file}")
            
            # Extract frame number from pattern: {base_name}-frame-{frame_number}.jpg
            if [[ "${jpg_basename}" =~ -frame-([0-9]+)\.jpg$ ]]; then
                frame_num="${BASH_REMATCH[1]}"
                
                # Find which segment this frame belongs to
                matched_segment=""
                for range in "${!segment_ranges[@]}"; do
                    IFS='_' read -r seg_start seg_end <<< "${range}"
                    
                    # Check if frame_num falls within this segment
                    if [ "${frame_num}" -ge "${seg_start}" ] && [ "${frame_num}" -le "${seg_end}" ]; then
                        matched_segment="${range}"
                        break
                    fi
                done
                
                if [ -n "${matched_segment}" ]; then
                    # Rename JPG to match segment pattern
                    new_jpg_name="segment_${matched_segment}.jpg"
                    cp "${jpg_file}" "${FINAL_OUTPUT_DATA_DIR}/${new_jpg_name}" || {
                        echo "Warning: Failed to copy ${jpg_basename}" >&2
                        continue
                    }
                    ((jpg_copied++))
                    echo "  Copied and renamed: ${jpg_basename} -> ${new_jpg_name}"
                else
                    echo "  Warning: Frame ${frame_num} does not match any segment range, skipping ${jpg_basename}" >&2
                fi
            else
                echo "  Warning: Could not parse frame number from ${jpg_basename}" >&2
            fi
        fi
    done
    echo "  Copied ${jpg_copied} JPG file(s)"
fi

segment_network_dir="${OUTPUT_MMPOSE_FINAL}/${base_name}"
# Copy mmpose/pose estimation outputs
echo "Copying pose estimation outputs..."
if [ -d "${segment_network_dir}" ]; then
    for json_file in "${segment_network_dir}"/*.json; do
        if [ -f "${json_file}" ]; then
            json_basename=$(basename "${json_file}")
            cp "${json_file}" "${FINAL_OUTPUT_DATA_DIR}/${json_basename}" || {
                echo "Warning: Failed to copy ${json_basename}" >&2
                continue
            }
            ((json_copied++))
        fi
    done
    echo "  Copied ${json_copied} JSON file(s)"
    
    # Copy any visualization MP4 files if they exist
    for vis_mp4 in "${segment_network_dir}"/*.mp4; do
        if [ -f "${vis_mp4}" ]; then
            vis_basename=$(basename "${vis_mp4}")
            cp "${vis_mp4}" "${FINAL_OUTPUT_DATA_DIR}/${vis_basename}" || {
                echo "Warning: Failed to copy ${vis_basename}" >&2
                continue
            }
            ((mp4_copied++))
        fi
    done
fi

# --- CREATE FILM METADATA JSON ---
echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Creating film metadata JSON..."

FILM_METADATA_FILE="${FINAL_OUTPUT_DIR}/film_metadata.json"

if [ -f "${META_INFO_ROOT}" ]; then
    if command -v jq &> /dev/null; then
        # Use jq to extract the film metadata using ORIGINAL_BASENAME
        jq --arg key "${ORIGINAL_BASENAME}.mp4" '.[$key] // {"status": "Metadata not available for this file."}' \
            "${META_INFO_ROOT}" > "${FILM_METADATA_FILE}" 2>/dev/null || {
            echo "Warning: Failed to create film metadata JSON with jq" >&2
        }
    else
        echo "Warning: jq not available, skipping film metadata extraction" >&2
    fi
else
    echo "Warning: Metadata corpus file not found at ${META_INFO_ROOT}" >&2
fi

if [ -f "${FILM_METADATA_FILE}" ]; then
    echo "  Film metadata JSON created: ${FILM_METADATA_FILE}"
else
    echo "  Warning: Film metadata JSON was not created" >&2
fi

(cd "${FINAL_OUTPUT_DATA_DIR}" && zip -9 -r "${FINAL_OUTPUT_DIR}/${base_name}.zip" .) || {
    echo "Error: Failed to create zip archive" >&2
    exit 1
}

(cd "${FINAL_OUTPUT_DIR}" && zip -9 -r "${OUTPUT_JSON_FINAL}/${base_name}.zip" film_metadata.json ${base_name}.zip) || {
    echo "Error: Failed to create zip archive" >&2
    exit 1
}

# Summary
echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Consolidation complete:"
echo "  CSV files: ${csv_copied}"
echo "  JPG files: ${jpg_copied}"
echo "  JSON files: ${json_copied}"
echo "  MP4 files: ${mp4_copied}"
echo "  Final directory: ${FINAL_OUTPUT_DIR}"

if [ $((csv_copied + jpg_copied + json_copied + mp4_copied)) -eq 0 ]; then
    echo "Warning: No files were copied to final directory" >&2
fi

# Cleanup FINAL_OUTPUT_DIR if requested and zips were created successfully
if [ "${ENV_DS_ZIPONLY:-0}" = "1" ]; then
    echo "Deleting original files after zipping (ENV_DS_ZIPONLY=1):"
    # Verify zip files exist before deleting source directory
    MAIN_ZIP="${FINAL_OUTPUT_DIR}/${base_name}.zip"
    FINAL_ZIP="${OUTPUT_JSON_FINAL}/${base_name}.zip"
    
    if [ -f "${MAIN_ZIP}" ] && [ -f "${FINAL_ZIP}" ]; then
        # Additional safety checks before deletion
        if [ -n "${FINAL_OUTPUT_DIR}" ] && \
            [[ "${FINAL_OUTPUT_DIR}" == "${OUTPUT_JSON_FINAL}/${base_name}" ]] && \
            [ -d "${FINAL_OUTPUT_DIR}" ] && \
            [ ! -L "${FINAL_OUTPUT_DIR}" ]; then
            
            echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Removing final output directory (ENV_DS_ZIPONLY=1): ${FINAL_OUTPUT_DIR}"
            rm -rf "${FINAL_OUTPUT_DIR}" || {
                echo "Warning: Failed to remove final output directory: ${FINAL_OUTPUT_DIR}" >&2
            }
        else
            echo "Warning: Safety check failed for FINAL_OUTPUT_DIR deletion. Directory not removed." >&2
            echo "  FINAL_OUTPUT_DIR: ${FINAL_OUTPUT_DIR}" >&2
            echo "  Expected path: ${OUTPUT_JSON_FINAL}/${base_name}" >&2
        fi
    else
        echo "Warning: Zip files not found, skipping FINAL_OUTPUT_DIR cleanup for safety." >&2
        [ ! -f "${MAIN_ZIP}" ] && echo "  Missing: ${MAIN_ZIP}" >&2
        [ ! -f "${FINAL_ZIP}" ] && echo "  Missing: ${FINAL_ZIP}" >&2
    fi
else
    echo "Final output directory retained (ENV_DS_ZIPONLY=${ENV_DS_ZIPONLY:-0})."
fi

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') All requested steps completed (or skipped where outputs already existed)."






# # --- DIVIDE-BY-CUT step ---
# echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Checking divide-by-cut step inputs/outputs..."
# # For divide step we expect mmpose results (either the normal JSON or 17-keypoint JSON) 
# # Determine which mmpose json to use (check scratch locations):

# echo "DEBUG: Checking for files:"
# echo "  MMPOSE_JSON_17: ${MMPOSE_JSON_17}"
# echo "  File exists: $([ -f "${MMPOSE_JSON_17}" ] && echo "YES" || echo "NO")"
# echo "  Second path: ${MMPOSE_JSON}"
# echo "  File exists: $([ -f "${MMPOSE_JSON}" ] && echo "YES" || echo "NO")"
# echo "  MMPOSE_JSON_133: ${MMPOSE_JSON_133}"
# echo "  File exists: $([ -f "${MMPOSE_JSON_133}" ] && echo "YES" || echo "NO")"
# echo "  MMPOSE_JSON_HYBRID: ${MMPOSE_JSON_HYBRID}"
# echo "  File exists: $([ -f "${MMPOSE_JSON_HYBRID}" ] && echo "YES" || echo "NO")"
# echo "  base_name: ${base_name}"
# echo "  OUTPUT_MMPOSE: ${OUTPUT_MMPOSE}"

# if [ -f "${MMPOSE_JSON_HYBRID}" ]; then
#     DIV_MMPose_JSON="${MMPOSE_JSON_HYBRID}"
#     echo "DEBUG: Set DIV_MMPose_JSON to: '${DIV_MMPose_JSON}'"
# elif [ -f "${MMPOSE_JSON_17}" ]; then
#     DIV_MMPose_JSON="${MMPOSE_JSON_17}"
#     echo "DEBUG: Set DIV_MMPose_JSON to: '${DIV_MMPose_JSON}'"
# elif [ -f "${MMPOSE_JSON}" ]; then
#     DIV_MMPose_JSON="${MMPOSE_JSON}"
#     echo "DEBUG: Set DIV_MMPose_JSON to: '${DIV_MMPose_JSON}'"
# elif [ -f "${MMPOSE_JSON_133}" ]; then
#     DIV_MMPose_JSON="${MMPOSE_JSON_133}"
#     echo "DEBUG: Set DIV_MMPose_JSON to: '${DIV_MMPose_JSON}'"
# else
#     DIV_MMPose_JSON=""
#     echo "DEBUG: No files found, DIV_MMPose_JSON is empty"
# fi

# if [ -n "${DIV_MMPose_JSON}" ]; then
#     # Remove the provisional --mmpose entry from divide_cmd by rebuilding the array
#     new_divide_cmd=()
#     skip_next=0
#     for i in "${!divide_cmd[@]}"; do
#         if [ "${skip_next}" -eq 1 ]; then
#             skip_next=0
#             continue
#         fi
#         if [ "${divide_cmd[$i]}" = "--mmpose" ]; then
#             # Skip this element and the next one (the placeholder value)
#             skip_next=1
#             continue
#         fi
#         new_divide_cmd+=( "${divide_cmd[$i]}" )
#     done
#     divide_cmd=( "${new_divide_cmd[@]}" )
#     # Add the actual --mmpose argument with the discovered JSON path
#     divide_cmd+=( --mmpose "${DIV_MMPose_JSON}" )
#    # Print final divide command so dry-run / logs reflect what will run
#    echo "FINAL DIVIDE COMMAND:"
#    print_cmd_array divide_cmd
#    print_srun_example "${CONDA_ENV_PYSCENE}" "divide" "${divide_cmd[@]}"
# fi

# echo "DEBUG: Final DIV_MMPose_JSON value: '${DIV_MMPose_JSON}'"
# echo "DEBUG: Is it empty? $([ -z "${DIV_MMPose_JSON}" ] && echo "YES" || echo "NO")"

# if [ -z "${DIV_MMPose_JSON}" ]; then
#     echo "divide-by-cut: No mmpose JSON available to run divide step. Expected one of:"
#     echo "  - ${MMPOSE_JSON_17}"
#     echo "  - ${MMPOSE_JSON}"
#     echo "  - ${MMPOSE_JSON_133}"
#     echo "  - ${MMPOSE_JSON_HYBRID}"
#     echo "Skipping divide step." >&2
# else
#     conda activate "${CONDA_ENV_PYSCENE}" #PYSCENE also has modules needed for divide
#     # Decide which divide command variant to use based on which mmpose json we have
#     if [ "${ENV_DS_MODEL:-mmpose}" = "rtmw" ]; then
#         # For rtmw we previously planned to run both 133 and 17 divides; only run those whose input exists
#         if [ -f "${MMPOSE_JSON_133}" ]; then
#             echo "divide-by-cut: Running divide for 133-keypoint JSON..."
#             if run_cmd divide_cmd_133; then
#                 echo "divide-by-cut (133): Completed successfully."
#             else
#                 echo "divide-by-cut (133): FAILED." >&2
#                 print_cmd_array divide_cmd_133
#                 exit 1
#             fi
#         else
#             echo "divide-by-cut: 133-keypoint JSON missing; skipping divide_133."
#         fi

#         if [ -f "${MMPOSE_JSON_17}" ]; then
#             echo "divide-by-cut: Running divide for 17-keypoint JSON..."
#             if run_cmd divide_cmd_17; then
#                 echo "divide-by-cut (17): Completed successfully."
#             else
#                 echo "divide-by-cut (17): FAILED." >&2
#                 print_cmd_array divide_cmd_17
#                 exit 1
#             fi
#         else
#             echo "divide-by-cut: 17-keypoint JSON missing; skipping divide_17."
#         fi
#     elif [ "${ENV_DS_MODEL:-mmpose}" = "hybrid" ]; then
#          # hybrid flow: run the single divide command if inputs exist
#         echo "divide-by-cut: Running divide (hybrid flow) with: ${DIV_MMPose_JSON}"
#         if run_cmd divide_cmd; then
#             echo "divide-by-cut: Completed successfully."
#         else
#             echo "divide-by-cut: FAILED." >&2
#             exit 1
#         fi   
#     else
#         # mmpose flow: run the single divide command if inputs exist
#         echo "divide-by-cut: Running divide (mmpose flow) with: ${DIV_MMPose_JSON}"
#         if run_cmd divide_cmd; then
#             echo "divide-by-cut: Completed successfully."
#         else
#             echo "divide-by-cut: FAILED." >&2
#             exit 1
#         fi
#     fi
# fi

# echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') All requested steps completed (or skipped where outputs already existed)."
