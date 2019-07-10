# Gradient-Descent
## Intro
The aim of supervised machine learning algorithms is to best estimate a target function that maps input data onto output variables. In other words, input data is used to produce predictions in the output based on some function. The idea of gradient descent involves minimising the error associated with these predicted and expected outputs, known as the cost function. Therefore, to have an accurate model, it is imperative to have this cost function minimised so that the error value is as low as possible. For a given function, this means finding the minimum value. 
## Plain Vanilla Gradient Descent
Given a function and a starting point somewhere along this function, plain vanilla gradient descent essentially finds the path to the minimum point by taking steps and calculating the slope at each step taken. By doing so, not only can the direction of descent be determined but also the sizes of steps taken, since the gradient becomes much smaller the closer it gets the minimum point. The algorithm stops when the calculated gradient approaches zero or after reaching a user specified limit for number of steps taken. 
## Variations
Certain modifications can be made to optimise the plain vanilla gradient descent algorithm. Two such examples include Momentum and Nesterov Accelerated Gradient (NAG). Momentum considers the exponential moving average of past gradients as well as the current gradient being calculated at each iteration. This method has the potential to avoid getting stuck at saddle points since the momentum should carry you over the point. However, it is also susceptible to overshooting the minimum as a result. NAG builds on this further by using projected gradients in its calculations, which involves essentially ‘looking ahead’ to determine whether the gradient is flattening or reversing in direction. This prevents potential oscillations from occurring when computing the gradient path. 

## Results
To test the ability of gradient descent to find the global minimum, a Booth function was used which has the following equation: f(x,y)=(x+2y-7)^2+(2x+y-5)^2
![*Figure 1: Surface plot of the Booth function along with the path taken by the plain vanilla gradient descent algorithm, learning rate = 0.01, accuracy = 1e-10*][fig1]
![fig1](https://github.com/booki96/Gradient-Descent/blob/master/vanilla.png) 

![fig2](https://github.com/booki96/Gradient-Descent/blob/master/lr%3D.01.png)
![fig3](https://github.com/booki96/Gradient-Descent/blob/master/lr%3D.1.png)
![fig4](https://github.com/booki96/Gradient-Descent/blob/master/momentum.png)
![fig5](https://github.com/booki96/Gradient-Descent/blob/master/nag.png)
