### Training details


- lr: 1e-4 to 1e-5, reduced step-by-step or reduced once convergence occurs.


- adam optimizer.


- start to change attenuation factor once the accuracy reaches 85% under normal softmax.


- When the loss converges, reduce the factor by 0.1 until 0.7.
