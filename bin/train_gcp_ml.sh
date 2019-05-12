#!/usr/bin/env bash

set -o errexit -o pipefail -o noclobber -o nounset

BASEDIR=$(cd $(dirname "$0"); pwd -P)
DISTDIR="${BASEDIR}/dist"

now=$(date +"%Y%m%d_%H%M%S")
export TOOL_NAME="$0"
export TRAINER_PACKAGE_PATH=`pwd`/
export MAIN_TRAINER_MODULE="hugin.tools.cli"
export _PYTHON_PREFIX=$(python -c 'import sys; print sys.prefix')
export _DEFAULT_SCALE_CONFIG="${BASEDIR}/etc/gcp/standard_cpu.yaml"
export _DEFAULT_REGION="europe-west4"
export _DEFAULT_JOB_NAME="${LOGNAME}_$now"

function show_help() {
    echo -e "Usage: $0"
    echo -e "\t-c ${_DEFAULT_SCALE_CONFIG} # ML Engine Config"
    echo -e "\t-r ${_DEFAULT_REGION} # ML Engine Region"
    echo -e "\t-j ${_DEFAULT_JOB_NAME} # Job Name"
    echo -e "\t-b gs://BUCKET_PATH # Specify bucket path"
}

! args=$(getopt c:r:j:b: $*)
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    show_help
    exit 2
fi

set -- $args

for i do
    case "$i" in
        -c)
            carg="$2"; shift;
            shift;;
        -r)
            rarg="$2"; shift;
            shift;;
        -j)
            jarg="$2"; shift;
            shift;;
        -b)
            barg="$2"; shift;
            shift;;
        --)
            shift; break;;
    esac
done

TOOL_ARGS="$@"


GCP_SCALE_CONFIG=${carg:-"${_DEFAULT_SCALE_CONFIG}"}
REGION=${rarg:-"${_DEFAULT_REGION}"}
JOB_NAME=${jarg:-"${_DEFAULT_JOB_NAME}"}
BUCKET_PATH=${barg:-""}

if [ "$JOB_NAME" != "${_DEFAULT_JOB_NAME}" ]; then
    JOB_NAME="${JOB_NAME}_${_DEFAULT_JOB_NAME}"
fi

if [ -z "${BUCKET_PATH}" ]; then
    echo "Bucket path not specified!"
    show_help
    exit 2
fi
BUCKET_PATH=${BUCKET_PATH%/}

export PACKAGE_STAGING_PATH="${BUCKET_PATH}"
export PACKAGES_PATH="${PACKAGE_STAGING_PATH}/packages"


JOB_DIR="${BUCKET_PATH}/jobs/${JOB_NAME}"
echo "Scale config: ${GCP_SCALE_CONFIG}"
echo "Region: ${REGION}"
echo "Job name: ${JOB_NAME}"
echo "Bucket Path: ${BUCKET_PATH}"
echo "Job Dir: ${JOB_DIR}"
echo "[----]"

cd "$BASEDIR"


if [ -e ${DISTDIR} ]; then
    echo "Removing binary package"
    rm -fr "${DISTDIR}"
fi
echo "Building binary package"
python setup.py sdist || exit 1

LATEST_VERSION=$(ls -tr "${DISTDIR}" | tail -1)
echo "Latest build version is ${LATEST_VERSION}"

if [ -z "${LATEST_VERSION}" ]; then
    echo "No package built :("
    exit 1
fi

BASE_PACKAGE_NAME=$(echo ${LATEST_VERSION} | sed 's|.tar.gz||g')
PACKAGE_NAME="${BASE_PACKAGE_NAME}-${now}.tar.gz"
PACKAGE_PATH="${PACKAGES_PATH}/${PACKAGE_NAME}"

echo "Uploading package to ${PACKAGE_PATH}"
gsutil cp "${DISTDIR}/${LATEST_VERSION}" "${PACKAGE_PATH}" || exit 1


gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --job-dir $JOB_DIR  \
    --packages "${PACKAGE_PATH}" \
    --module-name $MAIN_TRAINER_MODULE \
    --scale-tier custom \
    --config ${GCP_SCALE_CONFIG} \
    --region $REGION \
    --python-version 3.5 \
    --runtime-version 1.12 \
    -- \
    train \
    --switch-to-prefix \
    ${TOOL_ARGS}




