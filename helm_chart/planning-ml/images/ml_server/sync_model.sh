#! /bin/bash

while :; do
  aws s3 sync "s3://${MODEL_S3_BUCKET}/${MODEL_S3_PATH}" "${MODEL_PATH}"
  sleep $SLEEP_DURATION
done
