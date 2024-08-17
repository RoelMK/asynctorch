# This is a modified version of the Leaky class from snntorch.
# It makes the reset happen at the end of the timestep, rather than the beginning.
# source: https://snntorch.readthedocs.io/en/latest/_modules/snntorch/_neurons/leaky.html#Leaky
import torch
import snntorch as snn

class Leaky(snn.Leaky):
    def __init__(self, *args, init_hidden=True, reset_mechanism='zero', state_quant=False, **kwargs):
        if not init_hidden:
            raise NotImplementedError("The modified Leaky class is only implemented for init_hidden=True.")
        if reset_mechanism != 'zero':
            raise NotImplementedError("The modified Leaky class is only implemented for reset_mechanism='zero'.")
        if state_quant:
            raise NotImplementedError("The modified Leaky class is only implemented for state_quant=False.")
        super().__init__(*args, init_hidden=init_hidden, reset_mechanism=reset_mechanism, state_quant=state_quant, **kwargs)

    def forward(self, input_, mem=None):
        if not mem == None:
            self.mem = mem

        if self.init_hidden and not mem == None:
            raise TypeError(
                "`mem` should not be passed as an argument while `init_hidden=True`"
            )

        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)
            self.reset = torch.zeros_like(input_, device=self.mem.device)

        self.mem = self.state_function(input_)
        if self.inhibition:
            spk = self.fire_inhibition(
                self.mem.size(0), self.mem
            )  # batch_size
        else:
            spk = self.fire(self.mem)
        self.reset = self.mem_reset(self.mem) # check for reset
        self.mem = (1 - self.reset) * self.mem # apply reset
        if self.output:
            return spk, self.mem
        else:
            return spk
