#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Class to create a Dense layer and a LIF layer at the same time.

# Non custom imports
import numpy as np
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.proc.dense.process import Dense
from lava.magma.core.model.sub.model import AbstractSubProcessModel

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements

from lava.magma.core.run_configs import RunConfig, Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import *
import matplotlib.pyplot as plt
import os

class DenseLayer(AbstractProcess):
    """Class to create a Dense layer and a LIF layer at the same time. 
    This class specifically serves as a data structure from which SubLayerDenseModel will inherit the variables.
    """ 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        #Shape of the weights of the dense layer
        shape = kwargs.get("shape", (1, 1))
        
        #Inverse of decay time-constant for current decay.
        du = kwargs.pop("du", 0)
        
        #Inverse of decay time-constant for voltage decay.
        dv = kwargs.pop("dv", 0)
  
        
        #bias current of the LIF
        bias = kwargs.pop("bias", 0)
        
        # Voltage threshold of the LIF
        vth_hi = kwargs.pop("vth_hi", 10)
        
        #weights between the dense layer and the LIF
        weights = kwargs.pop("weights", 0)
        
        
        #Define all the necessary ports and variables that lava needs according to the previous variables
        self.s_in = InPort(shape=(shape[1],))
        self.s_out = OutPort(shape=(shape[0],))
        self.u = Var(shape=(shape[0],), init=0)
        self.v = Var(shape=(shape[0],), init=0)
        self.bias = Var(shape=(shape[0],), init=bias)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.vth_hi = Var(shape=(1,), init=vth_hi)
        self.weights = Var(shape=shape)
        



        
@implements(proc=DenseLayer, protocol=LoihiProtocol)
class SubDenseLayerModel(AbstractSubProcessModel):
    """Class to create a Dense layer and a LIF layer at the same time. 

    This layer implements the actual Dense-LIF relationship.
    """ 
    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        # Instantiate child processes
        #input shape is a 2D vec (shape of weight mat)
        
        
        #Takes variables from parent layer
        shape = proc.init_args.get("shape",(1,1))
        weights = proc.init_args.get("weights",(1,1))
        bias = proc.init_args.get("bias",(1,1))
        vth_hi = proc.init_args.get("vth_hi",(1,1))
        du = proc.init_args.get("du",(1,1))
        

        dv = proc.init_args.get("dv",(1,1))
        #Creates synaptic dense layer according to the weights and shapes specified. 
        #The synapses can be excitatory or inhibitory.
        self.dense = Dense(shape=shape, weights=weights,sign_mode=1)
        
        #Creates the LIF neurons
        self.lif = LIF(shape=(shape[0],),bias=bias,vth=vth_hi,du=du,dv=dv)
        #Note that since the LIF model in Lava is a current-based model (CuBa).
        #Therefore, the synapses elicit an increment in current and not voltage.
        #The reset of this model is set to 0 by default.
        
        # connect Parent in port to child Dense in port.
        proc.in_ports.s_in.connect(self.dense.in_ports.s_in)
        
        # connect Dense Proc out port to LIF Proc in port. Note that this is only internal to this layer.
        self.dense.out_ports.a_out.connect(self.lif.in_ports.a_in)
        
        # connect child LIF out port to parent out port
        self.lif.out_ports.s_out.connect(proc.out_ports.s_out)
        
        
        #Exposes the SubDenseLayerModel variables to the DenseLayer
        proc.vars.u.alias(self.lif.vars.u)
        proc.vars.v.alias(self.lif.vars.v)
        proc.vars.bias.alias(self.lif.vars.bias)
        proc.vars.du.alias(self.lif.vars.du)
        proc.vars.dv.alias(self.lif.vars.dv)
        proc.vars.vth_hi.alias(self.lif.vars.vth)
        proc.vars.weights.alias(self.dense.vars.weights)
        #proc.vars.spikes.alias(self.lif.vars.spikes)
        
def plot_layer(layer_idx,monitors,lim, model_length):
    """Function that plots a layer's voltage, current and spiking activity.
    
    layer_idx: Index of the layer composing the chain (int)
    
    monitors: Nested list with dimensions (model_length, layer_width). Each slot within the list corresponds to a list containing the monitor of the spikes, 
    voltage and current of each layer.  
    
    lim: right boundary for the plotting function"""
    
    #Check if layer_idx is lower than model_length
    assert layer_idx <= model_length
    
    #Plot Voltage of the layer
    plt.figure(figsize=(10,10))
    plt.subplot(311)
    plt.plot(monitors[layer_idx][1].get_data()[list(monitors[layer_idx][1].get_data().keys())[0]]["v"])
    plt.title("Voltage of layer "+str(layer_idx))
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.xlim(0,lim)

    
    plt.subplot(312)
    plt.plot(monitors[layer_idx][2].get_data()[list(monitors[layer_idx][2].get_data().keys())[0]]["u"])
    plt.title("Current of layer "+str(layer_idx))
    plt.xlabel("Time")
    plt.ylabel("Current")
    plt.xlim(0,lim)

    
    plt.subplot(313)
    i,j=np.where(monitors[layer_idx][0].get_data()[list(monitors[layer_idx][0].get_data().keys())[0]]["s_out"]!=0)
    plt.scatter(i[np.where(j==0)],j[np.where(j==0)],c=['r'])
    plt.scatter(i[np.where(j==1)],j[np.where(j==1)],c=['g'])
    plt.scatter(i[np.where(j==2)],j[np.where(j==2)],c=['b'])
    plt.scatter(i[np.where(j==3)],j[np.where(j==3)],c=['y'])
    plt.scatter(i[np.where(j==4)],j[np.where(j==4)],c=['orange'])
    plt.title("Spiking activity of layer "+str(layer_idx))
    plt.xlabel("Time")
    plt.ylabel("Neuron index within the layer")
    plt.xlim(0,lim)
    
    plt.tight_layout()

    #plt.plot(monv0.get_data()[list(monv0.get_data().keys())[0]]["v"])
    
def chain_raster_plot(monitors,model_length,num_steps):
    """Function creating a raster plot for the whole chain. It also prints the spike-timing of one neuron of each of the layers.
    
    
    monitors: Nested list with dimensions (model_length, layer_width). Each slot within the list corresponds to a list containing the monitor of the spikes, 
    voltage and current of each layer.   """
    
    colours=['r','g','b','black','orange']
    for layer_idx in range(len(monitors)):
        i,j=np.where(monitors[layer_idx][0].get_data()[list(monitors[layer_idx][0].get_data().keys())[0]]["s_out"]!=0)
        plt.scatter(i[np.where(j==0)],model_length*layer_idx+j[np.where(j==0)],c=colours[layer_idx])
        plt.scatter(i[np.where(j==1)],model_length*layer_idx+j[np.where(j==1)],c=colours[layer_idx])
        plt.scatter(i[np.where(j==2)],model_length*layer_idx+j[np.where(j==2)],c=colours[layer_idx])
        plt.scatter(i[np.where(j==3)],model_length*layer_idx+j[np.where(j==3)],c=colours[layer_idx])
        plt.scatter(i[np.where(j==4)],model_length*layer_idx+j[np.where(j==4)],c=colours[layer_idx])
        plt.xlim(0,num_steps)
        
        print("Spike-timings of layer",layer_idx," ",i[np.where(j==0)])
    

