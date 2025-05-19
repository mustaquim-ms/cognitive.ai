from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
# Create a network object
net = Network()
# Define an input layer with 100 neurons
input_layer = Input(n=100)
net.add_layer(input_layer, name="Input")
# Define a layer of Leaky Integrate-and-Fire (LIF) neurons with 10 neurons
output_layer = LIFNodes(n=10)
net.add_layer(output_layer, name="Output")
# Create a connection between the input and output layers with STDP learning rule
connection = Connection(input_layer, output_layer, update_rule="stdp", nu=[1e-2, 1e-2])
net.add_connection(connection, source="Input", target="Output")

print(net)
