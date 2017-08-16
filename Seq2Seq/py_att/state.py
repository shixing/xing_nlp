import state_pb2
import numpy as np
import struct 

class StateWrapper:
    
    def __init__(self):
        self._state = state_pb2.State()

    def append_proto(self,repeat, array):
        for a in array:
            repeat.append(a)

    def create(self, inputs, outputs, weights, state):
        l = int(np.sum(weights))
        input = [x[0] for x in inputs[:l]]
        output = [x[0] for x in outputs[:l]]
        self._state.length = l
        self.append_proto(self._state.input, input)
        self.append_proto(self._state.output, output)

        for step in state[:l]:
            step_pb = self._state.steps.add()
            for layer in step:
                layer_pb = step_pb.layers.add()
                self.append_proto(layer_pb.fg.values, layer['fg'].reshape(-1).tolist())
                self.append_proto(layer_pb.ig.values, layer['ig'].reshape(-1).tolist())
                self.append_proto(layer_pb.og.values, layer['og'].reshape(-1).tolist())
                self.append_proto(layer_pb.i.values, layer['i'].reshape(-1).tolist())
                self.append_proto(layer_pb.h.values, layer['h'].reshape(-1).tolist())
                self.append_proto(layer_pb.c.values, layer['c'].reshape(-1).tolist())
                
        
    def save_to_stream(self,f):
        nbyte = self._state.ByteSize()
        f.write(struct.pack('i',nbyte))
        f.write(self._state.SerializeToString())
        f.flush()

    def load_from_stream(self,f):
        try:
            nbyte = struct.unpack('i',f.read(struct.calcsize('i')))[0]
            self._state.ParseFromString(f.read(nbyte))
            return nbyte
        except:
            return 0

    def pretty_print(self):
        names = ['fg','ig','og','i','h','c']
        for i,w in enumerate(weights):
            if w == 0:
                break
            print("Step:{} input:{} output:{}".format(i,inputs[0],outputs[0]))
            state = states[i]
            for l in xrange(len(state)):
                print("Layer:{}".format(l))
                state_layer  = state[l]
                for name in names:
                    print("{}".format(name))
                    print(state_layer[name])


def load_states(fn):
    f = open(fn,'rb')
    states = []
    while True:
        state = StateWrapper()
        nbyte = state.load_from_stream(f)
        if nbyte == 0:
            break
        states.append(state)
    f.close()
    return states

def state_ite(fn):
    f = open(fn,'rb')
    while True:
        state = StateWrapper()
        nbyte = state.load_from_stream(f)
        if nbyte == 0:
            break
        yield state
    f.close()

