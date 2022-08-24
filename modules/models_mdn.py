import tensorflow_probability as tfp
import tensorflow as tf
import sonnet as snt
from gn4pions.modules.models import MultiOutBlockModel, MultiOutWeightedRegressModel
from graph_nets import utils_tf, modules

def convert_to_tensor(X):
    return tf.concat([tfp.distributions.Distribution.mean(X), tfp.distributions.Distribution.stddev(X)],1)

class MultiOutBlockModel_MDN(MultiOutBlockModel):

    def __init__(self,
               global_output_size=1,
               num_outputs=1,
               model_config=None,
               name="MultiOutBlockModel",
               num_components=3):
        super().__init__(global_output_size=global_output_size,
                         num_outputs=num_outputs,
                         model_config=model_config)

        self.mdn = tfp.layers.MixtureNormal(num_components, event_shape=[1], validate_args=True,
                                 convert_to_tensor_fn=convert_to_tensor)


    def __call__(self, input_op):
        latent = self._core[0](input_op)
        latent_all = [input_op]
        for i in range(1, self._num_blocks):
            if self._concat_input:
                core_input = utils_tf.concat([latent, latent_all[-1]], axis=1)
            else:
                core_input = latent

            latent_all.append(latent)
            latent = self._core[i](core_input)

        latent_all.append(latent)
        stacked_latent = utils_tf.concat(latent_all, axis=1)
        output = []

        for i in range(self._num_outputs):
            output.append(self._output_transform[i](stacked_latent))

        input_MDN = utils_tf.concat(output, axis=1)

        output_MDN = self.mdn(input_MDN.globals)
        return output_MDN



class MultiOutWeightedRegressModel_MDN(MultiOutWeightedRegressModel):

    def __init__(self,
               global_output_size=1,
               num_outputs=1,
               model_config=None,
               name="MultiOutWeightedRegressModel_MDN",
               num_components=3):
        super(MultiOutWeightedRegressModel_MDN, self).__init__(global_output_size,num_outputs,
                                                               model_config=model_config,
                                                               name="MultiOutWeightedRegressModel_MDN")

        global_regress_fn = lambda: snt.Linear(9, name="regress_linear")
        self._regress_transform =  modules.GraphIndependent(
                None, None, global_regress_fn, name="reression_output")

        self.mdn = tfp.layers.MixtureNormal(num_components, event_shape=[1], validate_args=True,
                                 convert_to_tensor_fn=convert_to_tensor)

    def __call__(self, input_op):

        latent = self._core[0](input_op)

        latent_all = [input_op]

        for i in range(1, self._num_blocks):
            if self._concat_input:
                core_input = utils_tf.concat([latent, latent_all[-1]], axis=1)
            else:
                core_input = latent

            latent_all.append(latent)
            latent = self._core[i](core_input)

        latent_all.append(latent)


        stacked_latent = utils_tf.concat(latent_all, axis=1)

        independent_output = []
        
        for i in range(self._num_outputs):

            independent_output.append(self._output_transform[i](stacked_latent))
        class_output = independent_output[-1]

        stacked_independent = utils_tf.concat(independent_output, axis=1)

        regress_output = self._regress_transform(stacked_independent)

        #input_MDN = utils_tf.concat(regress_output, axis=1)
        print(regress_output.globals)
        regress_output_MDN = self.mdn(regress_output.globals)
        return regress_output_MDN, class_output
