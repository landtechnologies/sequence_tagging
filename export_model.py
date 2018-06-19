from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import tensorflow as tf
import sys

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.utils import build_tensor_info

# Inspiration taken from here: https://medium.com/epigramai/tensorflow-serving-101-pt-1-a79726f7c103

export_loc = sys.argv[1] # e.g. ./models/simple_model/1

def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    a_tensor = model.sess.graph.get_tensor_by_name(model.input_name + ':0')
    sum_tensor = model.sess.graph.get_tensor_by_name(model.output_name + ':0')

    model_input = build_tensor_info(a_tensor)
    model_output = build_tensor_info(sum_tensor)

    # Create a signature definition for tfserving
    signature_definition = signature_def_utils.build_signature_def(
        inputs={model.input_name: model_input},
        outputs={model.output_name: model_output},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    builder = saved_model_builder.SavedModelBuilder(export_loc)

    builder.add_meta_graph_and_variables(
        model.sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        },
        legacy_init_op=tf.tables_initializer()
    )

    # Save the model so we can serve it with a model server :)
    builder.save()


if __name__ == "__main__":
    main()
