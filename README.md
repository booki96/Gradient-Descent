# Gradient-Descent
## Intro
The aim of supervised machine learning algorithms is to best estimate a target function that maps input data onto output variables. In other words, input data is used to produce predictions in the output based on some function. The idea of gradient descent involves minimising the error associated with these predicted and expected outputs, known as the cost function. Therefore, to have an accurate model, it is imperative to have this cost function minimised so that the error value is as low as possible. For a given function, this means finding the minimum value. 
## Plain Vanilla Gradient Descent
Given a function and a starting point somewhere along this function, plain vanilla gradient descent essentially finds the path to the minimum point by taking steps and calculating the slope at each step taken. By doing so, not only can the direction of descent be determined but also the sizes of steps taken, since the gradient becomes much smaller the closer it gets the minimum point. The algorithm stops when the calculated gradient approaches zero or after reaching a user specified limit for number of steps taken. 
## Variations
Certain modifications can be made to optimise the plain vanilla gradient descent algorithm. Two such examples include Momentum and Nesterov Accelerated Gradient (NAG). Momentum considers the exponential moving average of past gradients as well as the current gradient being calculated at each iteration. This method has the potential to avoid getting stuck at saddle points since the momentum should carry you over the point. However, it is also susceptible to overshooting the minimum as a result. NAG builds on this further by using projected gradients in its calculations, which involves essentially ‘looking ahead’ to determine whether the gradient is flattening or reversing in direction. This prevents potential oscillations from occurring when computing the gradient path. 

## Results
To test the ability of gradient descent to find the global minimum, a Booth function was used which has the following equation: f(x,y)=(x+2y-7)^2+(2x+y-5)^2

![fig1](https://github.com/booki96/Gradient-Descent/blob/master/vanilla.png) 
|:--:| 
| *Fig1: Surface plot of the Booth function along with the path taken by the plain vanilla gradient descent algorithm, learning rate = 0.01, accuracy = 1e-10* |

The surface has been plotted on 10x10 mesh with a starting point chosen as [-8,-8], and the path converges at [1,3]. This is where minimum loss is obtained and, in this case, where the global minimum is located.
The step sizes were adjusted in order to compare convergence rates. The figures below show that increasing the learning rate (and hence the step size) provides faster convergence. However, the algorithm becomes more prone to overshooting minimum points and failing to converge since it can ‘jump’ past the critical point. On the other hand, decreasing step size provides greater chance of convergence at the expense of computational time.  

![fig2](https://github.com/booki96/Gradient-Descent/blob/master/lr%3D.01.png)
|:--:| 
| *Plain vanilla convergence, learning rate = 0.01, degree of accuracy = 1e-10* |

![fig3](https://github.com/booki96/Gradient-Descent/blob/master/lr%3D.1.png)
|:--:| 
| *Plain vanilla convergence, learning rate = 0.11, degree of accuracy = 1e-10* |

Implementing Momentum into the gradient descent algorithm in figure 4 achieved convergence with fewer iterations. However, the path taken seems to be much more oscillatory in nature than the plain vanilla. NAG, shown in figure 5, also converges with fewer iterations than plain vanilla and, in addition, appears to have a smoother descent than Momentum. 

![fig4](https://github.com/booki96/Gradient-Descent/blob/master/momentum.png)
|:--:| 
| *Surface plot of the Booth function along with the path taken by the Momentum algorithm, learning rate = 0.01, accuracy = 1e-10* |

![fig5](https://github.com/booki96/Gradient-Descent/blob/master/nag.png)
|:--:| 
| *Surface plot of the Booth function along with the path taken by Nesterov Accelerated Gradient (NAG), learning rate = 0.01, accuracy = 1e-10* |

| Algorithm   |      lr=0.01      |  lr=0.1 |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |
