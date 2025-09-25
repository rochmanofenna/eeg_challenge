#!/bin/bash
# Download HBN EEG data from S3

RELEASE=${1:-R1}
OUTPUT_DIR="/workspace/hbn_data/${RELEASE}"

echo "Downloading HBN Release ${RELEASE}..."
echo "S3 URI: s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_${RELEASE}"

mkdir -p ${OUTPUT_DIR}

# Download with no-sign-request (public bucket)
# Limit to task-rest EEG files only for faster download
aws s3 sync \
    s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_${RELEASE}/ \
    ${OUTPUT_DIR}/ \
    --no-sign-request \
    --exclude "*" \
    --include "*/eeg/*task-rest*" \
    --include "participants.tsv" \
    --include "dataset_description.json"

echo "Downloaded to ${OUTPUT_DIR}"
echo "Size: $(du -sh ${OUTPUT_DIR} | cut -f1)"
