import numpy as np

def lookup(w,i):
    return w[i]

def d_lookup(djdy,w,i):
    djdw = np.zeros_like(w)
    djdw[i] = djdy
    return djdw

def linear(w,b,x):
    return np.dot(x,w) + b

def d_linear(djdy,w,b,x):
    djdb = djdy
    djdw = np.dot(x.reshape(-1,1),djdy.reshape(1,-1)).T
    djdx = np.dot(djdy,w.T)
    return djdx,djdw,djdb

def tanh(x):
    return np.tanh(x)

def d_tanh(djdy,y):
    djdx = djdy * (1 - y)*y
    return djdx

def softmax(x):
    ex = np.exp(x)
    ex = ex / np.sum(ex)
    return ex

def cross_entropy(x,i):
    return -np.log(x[i])

def d_cross_entropy_softmax(softmax_y,i):
    djdx = softmax_y
    djdx[i] -= 1
    return djdx


def forward(input_embed,linear_w,linear_b,output_embed, output_embed_b, pre_word, current_word):

    print "Forward"
    print

    # forward
    h1 = lookup(input_embed, pre_word)
    print "h1"
    print h1

    h2 = linear(linear_w,linear_b,h1)
    print "h2"
    print h2

    h3 = tanh(h2)
    print "h3"
    print h3

    h4 = linear(output_embed.T,output_embed_b,h3)
    print "h4"
    print h4

    h5 = softmax(h4)
    print "h5"
    print h5

    ce = cross_entropy(h5,current_word)
    print "ce"
    print ce
    
    return h1,h2,h3,h4,h5,ce

def backward(input_embed,linear_w,linear_b,output_embed, output_embed_b, pre_word, current_word,h1,h2,h3,h4,h5):

    print "Backward"
    print

    # backward
    # dj/d_softmax_y
    djdh4 = d_cross_entropy_softmax(h4,current_word)
    
    print "djdh4"
    print djdh4

    djdh3, djd_output_embed, djd_output_embed_b = d_linear(djdh4,output_embed.T, output_embed_b, h3)
    print "djdh3, djd_output_embed, djd_output_embed_b"
    print djdh3 
    print djd_output_embed 
    print djd_output_embed_b

    
    djdh2 = d_tanh(djdh3,h2)
    print "djdh2"
    print djdh2


    djdh1, djd_linear_w, djd_linear_b = d_linear(djdh2, linear_w, linear_b, h1)
    print "djdh1, djd_linear_w, djd_linear_b"
    print djdh1
    print djd_linear_w
    print djd_linear_b


    djd_input_embed = d_lookup(djdh1,input_embed,pre_word)
    print "djd_input_embed"
    print djd_input_embed


    return djdh4,djdh3,djdh2,djdh1,djd_input_embed,djd_linear_w,djd_linear_b,djd_output_embed, djd_output_embed_b

def update_weight(input_embed,linear_w,linear_b,output_embed, output_embed_b,djd_input_embed,djd_linear_w,djd_linear_b,djd_output_embed, djd_output_embed_b,eta):
    input_embed += - eta * djd_input_embed
    linear_w += -eta * djd_linear_w
    output_embed += eta * djd_output_embed
    output_embed_b +=-eta * djd_output_embed_b
    return input_embed,linear_w,linear_b,output_embed, output_embed_b




def main():
    # Define the matrix
    input_embed = np.array([[0.4,1],[0.2,0.4],[-0.3,2]])
    linear_w = np.array([[1.2,0.2],[-0.4,0.4]])
    linear_b = np.array([0,0.5])
    output_embed = np.array([[-1,1],[0.4,0.5],[-0.3,0.2]])
    output_embed_b = np.array([0,0.5,0])
    pre_word = 0 # a
    current_word = 1 # b
    eta = 0.1

    #forward
    for i in xrange(10):

        print "======================"
        print "Iteration ", i
        
        h1,h2,h3,h4,h5,ce = forward(input_embed,linear_w,linear_b,output_embed, output_embed_b, pre_word, current_word)
        print

        # backward
        djdh4,djdh3,djdh2,djdh1,djd_input_embed,djd_linear_w,djd_linear_b,djd_output_embed, djd_output_embed_b = backward(input_embed,linear_w,linear_b,output_embed, output_embed_b, pre_word, current_word,h1,h2,h3,h4,h5)
        print 

        # update the parameters
        input_embed,linear_w,linear_b,output_embed, output_embed_b = update_weight(input_embed,linear_w,linear_b,output_embed, output_embed_b,djd_input_embed,djd_linear_w,djd_linear_b,djd_output_embed, djd_output_embed_b,eta)

if __name__ == "__main__":
    main()
