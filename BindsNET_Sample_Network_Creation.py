# This code snippet demonstrates the basic creation of a spiking neural network
# using the BindsNET library, as described in Section 4.2.

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

# Create a network object
net = Network()

# Define an input layer with 100 neurons. This layer will receive external stimuli
# and convert them into spike trains for the SNN.
input_layer = Input(n=100)
net.add_layer(input_layer, name="Input")

# Define a layer of Leaky Integrate-and-Fire (LIF) neurons with 10 neurons.
# LIF neurons are a common model for biological neurons, accumulating input
# and firing a spike when their membrane potential crosses a threshold.
output_layer = LIFNodes(n=10)
net.add_layer(output_layer, name="Output")

# Create a connection between the input and output layers.
# The 'update_rule="stdp"' specifies Spike-Timing-Dependent Plasticity as the
# learning rule, which modifies synaptic weights based on the relative timing
# of pre- and post-synaptic spikes.
# 'nu=[1e-2, 1e-2]' are the learning rates for potentiation and depression.
connection = Connection(input_layer, output_layer, update_rule="stdp", nu=[1e-2, 1e-2])
net.add_connection(connection, source="Input", target="Output")

# Print the network structure to verify its components.
print(net)
