# Initial thoughts:
Which model to go for ? 

- reinforcement learning using neural network -- since this is a process of learning on the go...

- supervised learning models -- maybe neural network -- can be tried. 

- machine learning models -- since i don't have a defined output for a row, i dont think so i can handle this with machine learning models.

## 1 | Re-inforcement learning
since we don't have a training dataset, the output is totally based on the current state of the input.
input can be -- 1) the word(blank as well as already guessed few letters) 2 ) already wrongly guessed words 3) ...

the data characteristics:
- 2,50,000 words separated by new line
- words have only alphabetic letters
- the dataset is not filled with the common english words like "1, am, was, the , is , that ...etc"
- each word is having different length

## 1.1 | Models to go for

- The feed forward NN not handling the problem well. created the following model.
The model ended up guessing only one word repeatedly. Didn't spend much time to figure out why, since i already had a better option to handle the sequential data.

```
Model: "sequential_15"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_45 (Dense)            (None, 1, 6, 27)          756       
                                                                 
 flatten_16 (Flatten)        (None, 162)               0         
                                                                 
 dense_46 (Dense)            (None, 27)                4401      
                                                                 
 dense_47 (Dense)            (None, 26)                728       
                                                                 
=================================================================
Total params: 5885 (22.99 KB)
Trainable params: 5885 (22.99 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

- Going for Recurrent Neural networks -- which can handle the sequential data.
Do i actually need a RNN here ? (NOt really, but efficient)

- Reason 1: This models deal with different timestamp values effectively.
- Reason 2: It keeps the memory of each letter picked and easy to find the next letters.

# 2 | LSTM:
```

# Create the model.
def create_model(state_shape, action_shape):
    """
    Build the sequential model.

    :param state_shape: The shape of the state. (shape of the input)
    :param action_shape: The shape of the action. (shape of the output)
    """
    model = Sequential()


Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 reshape_1 (Reshape)         (None, 40, 27)            0         
                                                                 
 masking_1 (Masking)         (None, 40, 27)            0         
                                                                 
 lstm_2 (LSTM)               (None, 40, 27)            5940      
                                                                 
 lstm_3 (LSTM)               (None, 27)                5940      
                                                                 
 dense_2 (Dense)             (None, 26)                728       
                                                                 
=================================================================
Total params: 12,608
Trainable params: 12,608
Non-trainable params: 0
_________________________________________________________________
```

**1 | Metric(s) used:**

CategoricalCrossentropy -- Commonly used for multi-class classification tasks with more than two mutually exclusive classes. 
At the output layer, its basically deciding what letter to guess next...*basically choosing which class(letter) should i assign to the current state,*

**2| NOTE: padding all inputs:**

One downside of using the keras-rl dqn layer is, it adds one extra dimension to the input... so before giving it to the LSTM layer i need to Reshape it. Here getting the input shape dynamically is needed, so that i could able to reshape it to 3D. 

Because of that i am padding all the inputs to max 40 timestamps. assuming no word will have more than 40 letters.

Note: The maximum number of letter given in the training file is 29.




# 3 | Initial thinking in the Environment:

The OpenAi gym has a **Env** class, we can create a class inheriting it and train our model using keras-rl. They has a structured way of creating the Environment, So i chose this method for reinforcement training. (https://gymnasium.farama.org/api/env/)


1 | rewards
- method 1: 1 - if the guess is right. -1 -- wrong.
- method 2: n (number of letters present in the word for the particular guess), if the guess is right. -1 -- if the guess is wrong.

*Using:* method 2

2 | Action space:
- 26, the agent can take any number form 0 - 25.

3 | Input or way of encoding the word:

word: Apple
- method 1 - (n * 27) -- one-hot encoded n letters of the word.
```
array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       ])
```
- method 2: (n + 1 * 27) -- n, one hot encoded letters  + one rows indicating already guessed words.
```
array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       ])
```
- method 3: (n + 2 * 27) -- n, one hot encoded letters  + one rows indicating already guessed words + one rows indicating wrongly guessed letters.
```
array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
         0.],
       ])
``` 

I don't think so, wrongly guessed letters row will add value to the model.

*Using:* method 2

4 | Agents
Multiple agents are available, (https://keras-rl.readthedocs.io/en/latest/agents/overview/)

ours is an Descrete action space, so we can use -- DQN or CEM or SARSA.I didn't get much time to try out all. 

*Using DQN only.*


Reference:

- 1 | https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b

- 2 | linear layer at the end reason -- https://stackoverflow.com/questions/45493987/why-do-keras-rl-examples-always-choose-linear-activation-in-the-output-layer
- 3 | Keras lstm layer -- https://keras.io/api/layers/recurrent_layers/lstm/
- 4 | cntk rl -- https://cntk.azurewebsites.net/pythondocs/CNTK_203_Reinforcement_Learning_Basics.html
- 5 | keras rl -- https://keras.io/examples/rl/
- 6 | https://keras.io/examples/rl/deep_q_network_breakout/
- 7 | https://www.analyticsvidhya.com/blog/2021/02/introduction-to-reinforcement-learning-for-beginners/
- 8 | https://www.analyticsvidhya.com/blog/2022/03/a-brief-overview-of-recurrent-neural-networks-rnn/
- 9 | https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm/
