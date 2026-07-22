from src.model_small_qwen3.5 import Qwen35MoESmall


class Adapter:

    def forward(self, ids, sequence):
        self.model = Qwen35MoESmall()

        logits, kvs, states, convs = self.model(

            ids,

            kvs=sequence.kvs,

            states=sequence.states,

            convs=sequence.convs

        )

        sequence.kvs = kvs
        sequence.states = states
        sequence.convs = convs

        return logits