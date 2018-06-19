#! /bin/bash

docker build --tag tensorflow-serving tensorflow-serving
docker run -it -p 9000:9000 --name landtech_policy_tagging -v $(pwd)/exported/:/models/ tensorflow-serving --port=9000 --model_name=landtech_policy_tagging --model_base_path=/models