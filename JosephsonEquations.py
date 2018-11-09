import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt, cm, colors
from scipy.constants import pi,hbar,e,h
import imageio

flux_0 = h/2./e

def potential(t_list,Ic, current_DC, x, voltage_function=None,current_function = None, voltage_DC = None):

	if current_function == None:
		# compute voltage evolution
		phase_list = np.array([quad(voltage_function,0.,t)[0] for t in t_list])
		current_list = Ic*np.sin(2.*pi*phase_list)
		voltage_list = np.vectorize(voltage_function)(t_list)

	if voltage_function == None:
		# compute current evolution
		current_DC = np.linspace(current_DC[x], current_DC[x], len(t_list))
		#current_DC = np.vectorize(current_function)(t_list)
		phase_list = np.array([quad(voltage_DC,0.,t)[0] for t in t_list])
		current_list = Ic*np.sin(2.*pi*phase_list) + current_DC
		#phase_list = np.arcsin((current_list - current_function)/Ic)
		voltage_list = np.concatenate([(phase_list[1:]-phase_list[:-1])/(t_list[1]-t_list[0]),[0]])

	potential_list = np.array((-Ic*np.cos(phase_list) - current_list*phase_list))
	#print(phase_list)
	#potential_list = np.array([np.trapz(voltage_list[:i]*current_list[:i],t_list[:i]) for i in range(len(t_list))])
	return phase_list,potential_list,current_list,voltage_list


N = 201
def voltage_DC(t):
	return 1.
def voltage(t):
	return 2.
def current(t):
	return 2. 



t_list = np.linspace(0,2.,N)
Ic = 0.5
current_DC = np.linspace(2,7,10)
images = []

for i, cur in enumerate(current_DC):
	cur = int(cur)
	phase,U,I,V = potential(t_list,Ic,current_DC, cur, voltage_function=None,current_function = current, voltage_DC = voltage_DC)
	particle = np.argmin(U[:150])
	print(phase[particle])
	phase_particle = phase[particle:]
	potential_particle = U[particle:]
	plt.plot(phase, U, color='blue')
	#plt.plot(phase_particle,potential_particle)
	plt.scatter([phase[particle]], [U[particle]], color='red',s=100)
	plt.savefig("cur{}.png".format(cur))
	plt.ylim((-12, 0))
	images.append(imageio.imread("cur{}.png".format(cur)))

	if (i==len(current_DC)-1):
		plt.xlabel('Phase')
		plt.ylabel('Potential')
		plt.plot(phase, U, color='blue')
		#plt.show()
		for j in range (particle,len(phase)):
			plt.figure()
			plt.plot(phase, U, color='blue')
			plt.scatter([phase[j]], [U[j]], color='red',s=100)
			plt.savefig("particle{}.png".format(j))
			images.append(imageio.imread("particle{}.png".format(j)))
			#plt.show()

imageio.mimsave('WashboardPotential_1.gif', images)
# plt.plot(phase,U)
plt.close()