#############################################################################################################################################################
## Objective: Visualize simulated K-Means clustering animation.                                                                                             #                                                                                                                       #
## Please install "animation" package: install.packages("animation")                                                                                        #                                                                                          #
#############################################################################################################################################################

library(animation)

###  default objective function with grad.desc() f(x,y)=x2+2y2

grad.desc()


### objective function  f(x,y)=sin(12*x^2-14*y^4+3)cos(2*x+1-exp(y))

ani.options(nmax = 70)
par(mar = c(4, 4, 2, 0.1))
f2 = function(x, y) sin(1/2 * x^2 - 1/4 * y^2 + 3) * cos(2 * x + 1 - 
  exp(y))

grad.desc(f2, c(-2, -2, 2, 2), c(-1, 0.5), gamma = 0.3, tol = 1e-04)


####Maximum number of iterations reached!####
#### Try with smaller gamma (< 0.3) ######