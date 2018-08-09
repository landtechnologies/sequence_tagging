#! /bin/bash

mkdir -p "${MODEL_PATH}"

aws s3 cp "s3://${MODEL_S3_BUCKET}/${MODEL_S3_PATH}" "${MODEL_PATH}" --recursive