# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:13:44 2020

@author: harshitm
"""

#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib.animation import FuncAnimation


def readData(fileName):
  return np.loadtxt(fileName)

#normalize the input data
def normalize(arrayX):
  meanX = np.mean(arrayX)
  varX = np.var(arrayX)

  i = 0
  while i<arrayX.size :
    arrayX[i] = (arrayX[i]-meanX)/(np.sqrt(varX))
    i = i+1

def getCostFnValue(arrayX,arrayY,theta0,theta1):
  j = 0.0
  i = 0
  while i<arrayX.size :
    j = j + (0.5)*np.square(arrayY[i]-theta1*arrayX[i]-theta0)
    i = i+1
  return j


def gradientDescent(arrayX,arrayY,lrate):

  # Batch Gradient descent
  count = 0   #iterations
  theta0 = np.array([0.0])
  theta1 = np.array([0.0]) #initialization
  J_theta_val = getCostFnValue(arrayX,arrayY,theta0[count],theta1[count]) 
  J_theta = np.array([J_theta_val])

  #convergence Criteria
  #1 difference in all individual theta < 1e-6
  DELTA_THETA = 1e-6 
  converged = False

  while converged == False:   
    #summation_over_all_exapmples((yi-theta_t*xi)*xij
    updateTerm1 = 0.0
    updateTerm0 = 0.0

    i = 0
    while i<arrayY.size :
      updateTerm1 = updateTerm1 + (arrayY[i] - (theta1[count]*arrayX[i]) - (theta0[count]*1))*arrayX[i]
      updateTerm0 = updateTerm0 + (arrayY[i] - (theta1[count]*arrayX[i]) - (theta0[count]*1))*1             #x0=1  for all i   
      i = i+1              
    # parameter update
    theta1 = np.append(theta1,[theta1[count] + lrate*updateTerm1])
    theta0 = np.append(theta0,[theta0[count] + lrate*updateTerm0])
    # update J_Theta
    J_theta_val = getCostFnValue(arrayX,arrayY,theta0[count],theta1[count]) 
    J_theta = np.append(J_theta,J_theta_val)
    count = count+1
    converged = np.abs(theta1[count] - theta1[count-1]) < DELTA_THETA and np.abs(theta0[count] - theta0[count-1]) < DELTA_THETA    

  print("Learning rate: {} Total Iterations: {}".format(lrate,count))
  print("Final values: theta1: {} theta0: {}",theta1[theta1.size-1],theta0[theta0.size-1])
  return (theta0, theta1, J_theta)   


def plotData(arrayX,arrayY,theta0f,theta1f):
  plt.figure(figsize=(15, 8))
  plt.plot(arrayX,arrayY,'ro')
  minX = np.amin(arrayX)
  vminX = theta1f*minX + theta0f;
  maxX = np.amax(arrayX)
  vmaxX = theta1f*maxX + theta0f;
  plt.plot([minX,maxX],[vminX,vmaxX],'b')
  plt.savefig('liner_regression_dataplot.jpg')
  plt.show()


def plotErrorFn(arrayX,arrayY,theta0,theta1,J_theta):

  fig = plt.figure()
  fig = plt.figure(figsize=(30, 10))
  ax = fig.add_subplot(1,2,1,projection='3d')
  theta1f = theta1[theta1.size-1]   # final value
  theta0f = theta0[theta0.size-1]
  A = np.linspace(theta0f-1,theta0f+0.5,50)
  B = np.linspace(theta1f-0.5,theta1f+0.5,50)
  A, B = np.meshgrid(A, B)
  i = 0
  J = np.square((1/np.sqrt(2))*(arrayY[i]-B*arrayX[i]-A))
  i = i+1
  while i<arrayX.size:
    R = np.square((1/np.sqrt(2))*(arrayY[i]-B*arrayX[i]-A))
    S = np.add(R,J)
    J = S
    i = i+1

  # Plot the surface.
  surf = ax.plot_surface(A, B, J,cmap=cm.coolwarm, alpha=0.7)
  ax.scatter(theta0,theta1,J_theta, color='black', marker = '<', zorder=2)
  # Customize the z axis.
  ax.set_zlim(0, 60)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
  ax.set_xlabel("theta0")
  ax.set_ylabel("theta1")
  ax.set_zlabel("Cost Fn (J_theta)")
  fig.colorbar(surf, shrink=0.5, aspect=5)


  ax.set_xlim(theta0f-1,theta0f+0.5)
  ax.set_ylim(theta1f-0.5,theta1f+0.5)
  plt.savefig('liner_regression_surface_plot.jpg')
  plt.show()

  #Plot the Contours
  fig2, ax2 = plt.subplots(1,1, figsize=(12, 7))
  cp = ax2.contourf(A, B, J)
  ax2.scatter(theta0,theta1,J_theta, color='yellow', marker = '>', zorder=2)
  fig2.colorbar(cp)
  ax2.set_title("Contour Plot")
  ax2.set_xlabel("theta0")
  ax2.set_ylabel("theta1")
  fig2.savefig('linear_regression_contour_plot.jpg')
  plt.show()
  
def main():
  arrayX = readData('./linearX.csv')
  arrayY = readData('./linearY.csv')
  normalize(arrayX)
  lrate = 0.001
  theta0, theta1, J_theta = gradientDescent(arrayX,arrayY,lrate)
  plotData(arrayX,arrayY,theta0[theta0.size-1],theta1[theta1.size-1])
  plotErrorFn(arrayX,arrayY,theta0,theta1,J_theta)

    
if __name__ == "__main__":
    main()
