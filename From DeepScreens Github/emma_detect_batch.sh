#~/bin/bash -l

# example: bash detect_batch.sh /isi/mep-rip/working/deepscreens/cutdetector/videos_to_run mp4 john.p.bell@dartmouth.edu
# example optional flags:
#   bash detect_batch.sh --dry-run-sbatch /path mp4 user@ex
#   bash detect_batch.sh --dry-run-python /path mp4 user@ex
#
# Flags:
#   --dry-run-sbatch            : Do not submit to sbatch locally (current implemented behavior)
#   --dry-run-python            : Submit to sbatch but export DRY_RUN_IN_PYTHON=1 so the sbatch
#                                 wrapper (which you will modify) can detect it and do a dry-run
#                                 inside the sbatch job (e.g., print commands instead of executing).
#   --model <mmpose|rtmw|hybrid|none>  : Which model to use in python. Defaults to mmpose if omitted. Setting to none skips running pose detection
#   --verbose                   : Default behavior sends all stdout from python scripts to /dev/null. Including this logs stdout
#   --leave-scratch             : Leave scratch files after processing
#   --min-frames                : Minimum number of frames per segment (default: 60)
#   --show-video                : Default behavior does not create demo videos in mmpose/track
#   --smooth-2d                 : Default behavior does not apply 2D smoothing to keypoints
#   --smooth-3d                 : Default behavior does not apply 3D smoothing to keypoints
#   --zip-only                  : Default behavior does not delete output files after zipping them. Setting this flag deletes them

# Parse optional flags. Supports --model, --dry-run-sbatch, and --dry-run-python.
DRY_RUN_SBATCH=0
DRY_RUN_PYTHON=0
ENV_DS_MODEL="mmpose"   # default; allowed: mmpose, rtmw
ENV_DS_QUIET=1        # default
ENV_DS_VIDEO=0        # default
ENV_DS_CLEANUP=1      # default
ENV_DS_MIN_FRAMES=60  # default
ENV_DS_SMOOTH_2D=0    # default
ENV_DS_SMOOTH_3D=0    # default
ENV_DS_ZIPONLY=0      # default


# Build a new args array without recognized flags so existing positional handling is unchanged
new_args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run-sbatch)
            DRY_RUN_SBATCH=1
            shift
            ;;
        --dry-run-python)
            DRY_RUN_PYTHON=1
            shift
            ;;        
        --verbose)
            ENV_DS_QUIET=0
            shift
            ;;        
        --show-video)
            ENV_DS_VIDEO=1
            shift
            ;;
        --zip-only)
            ENV_DS_ZIPONLY=1
            shift
            ;;     
        --leave-scratch)
            ENV_DS_CLEANUP=0
            shift
            ;;
        --min-frames)
            if [ -z "${2:-}" ]; then
                echo "Error: --min-frames requires an argument (positive integer)"
                echo "Usage: $0 [--dry-run-sbatch] [--dry-run-python] [--verbose] [--show-video] [--min-frames <frames>] <directory_path> <file_extension> <your_email_address>"
                exit 1
            fi
            ENV_DS_MIN_FRAMES="$2"
            shift 2
            ;;
        --smooth-2d)
            ENV_DS_SMOOTH_2D=1
            shift
            ;;
        --smooth-3d)
            ENV_DS_SMOOTH_3D=1
            shift
            ;;
        --min-frames)
            if [ -z "${2:-}" ]; then
                echo "Error: --min-frames requires an argument (positive integer)"
                echo "Usage: $0 [--dry-run-sbatch] [--dry-run-python] [--verbose] [--show-video] [--min-frames <frames>] <directory_path> <file_extension> <your_email_address>"
                exit 1
            fi
            ENV_DS_MIN_FRAMES="$2"
            shift 2
            ;;
        --model)
            if [ -z "${2:-}" ]; then
                echo "Error: --model requires an argument (mmpose or rtmw or hybrid or none)"
                echo "Usage: $0 [--dry-run-sbatch] [--dry-run-python] [--verbose] [--show-video] [--model <mmpose|rtmw|hybrid|none>] <directory_path> <file_extension> <your_email_address>"
                exit 1
            fi
            case "$2" in
                mmpose|rtmw|none|hybrid)
                    ENV_DS_MODEL="$2"
                    shift 2
                    ;;
                *)
                    echo "Error: invalid value for --model: '$2'. Allowed: mmpose, rtmw, hybrid, none"
                    exit 1
                    ;;
            esac
            ;;
        --) # explicit end of flags
            shift
            while [ "$#" -gt 0 ]; do
                new_args+=("$1"); shift
            done
            break
            ;;
        -*)
            # unknown option: preserve it (so we don't break any future short flags)
            new_args+=("$1"); shift
            ;;
        *)
            new_args+=("$1"); shift
            ;;
    esac
done

# Restore positional parameters for the rest of the script
set -- "${new_args[@]}"

# Check if a directory path and file extension are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 [--dry-run-sbatch] [--dry-run-python] [--verbose] [--show-video] [--leave-scratch] [--model <mmpose|rtmw|hybrid|none>] [--min-frames <60>] [--zip-only] [<directory_path> <file_extension> <your_email_address>]"
    exit 1
fi

# Set the target directory and file extension
target_dir="$1"
file_extension="$2"
email_address="$3"

# tee_echo: print to stdout and (when DRY_RUN_SBATCH=1) append to ${log_dir}/dry-run-sbatch.log
# Supports an optional leading -e to enable backslash-escape interpretation (like echo -e).
# Usage:
#   tee_echo "simple message"
#   tee_echo -e "line1\nline2"
# tee_echo: print to stdout and (when DRY_RUN_SBATCH=1) append to ${log_dir}/dry-run-sbatch.log
# Supports an optional leading -e to enable backslash-escape interpretation (like echo -e).
# Preserves newlines passed as arguments (use printf with "%b" for -e mode).
# Usage:
#   tee_echo "simple message"
#   tee_echo -e "line1\nline2"
tee_echo() {
    local use_escape=0
    # Detect and consume a single -e flag if present as first arg
    if [ "${1:-}" = "-e" ]; then
        use_escape=1
        shift
    fi

    # Print to stdout. If -e requested, use echo -e (to preserve escape behavior);
    # otherwise use printf to preserve newlines/arguments safely.
    if [ "$use_escape" -eq 1 ]; then
        # echo -e is used intentionally for escape sequences and to honor backslashes in the input
        echo -e "$*"
    else
        # printf preserves arguments safely (each argument on one line if multiple args provided)
        # To preserve possible embedded newlines within arguments, print the single joined string if only one arg,
        # otherwise print each arg on its own line.
        if [ "$#" -eq 1 ]; then
            printf '%s\n' "$1"
        else
            for a in "$@"; do
                printf '%s\n' "$a"
            done
        fi
    fi

    # If in dry-run-sbatch mode, append the same message(s) (with timestamp) to the dry-run log.
    # if [ "${DRY_RUN_SBATCH:-0}" -eq 1 ]; then
        # Ensure log_dir exists (it should). Append each printed line with timestamp.
        if [ "$use_escape" -eq 1 ]; then
            # Expand escape sequences consistently for logging using printf '%b'
            printf '%s ' "$(date +%Y-%m-%dT%H:%M:%S%z)" >> "${log_dir}/sbatch.log"
            printf '%b\n' "$*" >> "${log_dir}/sbatch.log"
        else
            if [ "$#" -eq 1 ]; then
                printf '%s %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$1" >> "${log_dir}/sbatch.log"
            else
                for a in "$@"; do
                    printf '%s %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$a" >> "${log_dir}/sbatch.log"
                done
            fi
        fi
    # fi
}

script_dir=$(dirname "$(readlink -f "$0")")
log_dir="/isi/mep-rip/working/deepscreens/output/logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$log_dir"
tee_echo "The script_dir is $script_dir"

# Change to the target directory
cd "$target_dir" || { tee_echo "Error: Directory not found"; exit 1; }

file_list=()
while IFS= read -r -d '' file; do
    file_list+=("$file")
done < <(find -L "${target_dir}" -type f -iname "*.${file_extension}" -exec printf '%s\0' {} +)

tee_echo "${#file_list[@]}"

declare -a input_files=()

dispatch_sbatch(){
    # sbatch doesn't want to export an array, so serializing
    # Use a delimiter unlikely to appear in filenames (ASCII Record Separator)
    local delim
    delim=$'\036'   # use a non-printable unit separator to safely join filenames
    CUTFILES=$(printf "%s${delim}" "${input_files[@]}")

    tee_echo "Files to process (input_files):"
    tee_echo "${#input_files[@]}"

    if [ ${#input_files[@]} -gt 0 ]; then
        tee_echo "start each"

        # Build an export string robustly instead of trying fragile string edits.
        # Always export CUTFILES and NOTIFY_EMAIL; optionally add DRY_RUN_IN_PYTHON if requested.
        # We must escape any double quotes and backslashes in values so they are safely passed through --export.
        escape_for_export() {
            # Escape backslashes and double quotes
            local v="$1"
            v="${v//\\/\\\\}"
            v="${v//\"/\\\"}"
            printf '%s' "$v"
        }

        # Prepare exported values (escape special chars)
        esc_cutfiles="$(escape_for_export "$CUTFILES")"
        esc_email="$(escape_for_export "$email_address")"
        esc_model="$(escape_for_export "$ENV_DS_MODEL")"
        esc_quiet="$(escape_for_export "$ENV_DS_QUIET")"
        esc_logpath="$(escape_for_export "$log_dir")";
        esc_video="$(escape_for_export "$ENV_DS_VIDEO")"
        esc_cleanup="$(escape_for_export "$ENV_DS_CLEANUP")"
        esc_min_frames="$(escape_for_export "$ENV_DS_MIN_FRAMES")"
        esc_smooth_2d="$(escape_for_export "$ENV_DS_SMOOTH_2D")"
        esc_smooth_3d="$(escape_for_export "$ENV_DS_SMOOTH_3D")"
        esc_zip_only="$(escape_for_export "$ENV_DS_ZIPONLY")"

        EXPORT_VARS="CUTFILES=\"$esc_cutfiles\",NOTIFY_EMAIL=\"$esc_email\",ENV_DS_MODEL=\"${esc_model}\",ENV_DS_QUIET=\"${esc_quiet}\",ENV_DS_CLEANUP=\"${esc_cleanup}\",ENV_DS_LOGPATH=\"${esc_logpath}\",ENV_DS_VIDEO=\"${esc_video}\",ENV_DS_SMOOTH_2D=\"${esc_smooth_2d}\",ENV_DS_SMOOTH_3D=\"${esc_smooth_3d}\",ENV_DS_ZIPONLY=\"${esc_zip_only}\",ENV_DS_MIN_FRAMES=\"${esc_min_frames}\""
        if [ "$DRY_RUN_PYTHON" -eq 1 ]; then
            EXPORT_VARS+=",DRY_RUN_IN_PYTHON=\"1\""
        fi

        # Construct sbatch argument array to avoid eval and preserve quoting
        sbatch_args=(
            "--mail-user=${email_address}"
            "--mail-type=ALL"
            "--array=0-$(( ${#input_files[@]} - 1 ))"
            "--export=${EXPORT_VARS}"
            "--output=${log_dir}/output_%a_log.txt"
            "--error=${log_dir}/error_%a_log.txt"
            "$script_dir/emma_wrapper_batch_run.sbatch"
        )

        tee_echo "SBATCH command (preview):"
        # Print a readable preview of the sbatch invocation
        tee_echo "sbatch ${sbatch_args[*]}"

        if [ "$DRY_RUN_SBATCH" -eq 1 ]; then
            # existing behavior: do not submit locally
            tee_echo "[dry-run-sbatch] not submitting job to sbatch"
        elif [ "$DRY_RUN_PYTHON" -eq 1 ]; then
            # submit to sbatch, but notify user this is a python-wrapper dry-run pass-through
            tee_echo "[dry-run-python] submitting to sbatch with DRY_RUN_IN_PYTHON=1"
            # Call sbatch with the array form to preserve quoting
            sbatch "${sbatch_args[@]}"
        else
            # normal submit
            tee_echo "[submit] sending job to sbatch"
            sbatch "${sbatch_args[@]}"
        fi
    else
        tee_echo "No files found to process; skipping dispatch."
    fi
}

# Iterate through each file with the specified extension in the directory
for input_file in "${file_list[@]}"; do
    # There is a maximum length of the parameters you can hand to sbatch. This is a workaround
    if [ ${#input_files[@]} -gt 500 ]; then
        dispatch_sbatch
        input_files=()
    fi

    # If no file with the specified extension is found, skip the loop
    [ -e "$input_file" ] || continue

    tee_echo -e "${input_file}"
    input_files+=("$input_file")

    # Commented out below because wrapper script handles dupe checks
    # Get the file's base name without the extension
    # base_name="${input_file%.*}"
    
    # # Construct the output file name with the .txt extension
    # output_file="${base_name}.txt"
    
    # # Check if the output file already exists, if not, process the input file
    # if [ ! -e "$output_file" ]; then
    #     input_files+=("$input_file")
    # fi
done

dispatch_sbatch
