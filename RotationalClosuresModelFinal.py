import numpy as np
from scipy.integrate import odeint  
import matplotlib.pyplot as plt 
import math
import seaborn as sb
import multiprocessing as mp 
from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap


####### To install libraries ###############
# pip install [library name] in terminal
# e.g., pip install seaborn 
# list of libraries: matplotlib, numpy, seaborn, joblib, scipy 

class Model:
	
	# constructor for Model object
	def __init__(self, model_type, n, frac_nomove, mgmt_strat = 'periodic'): 
		self.model_type = model_type
		self.n = n 
		self.frac_nomove = frac_nomove
		self.mgmt_strat = mgmt_strat 

		self.f = 0
		self.closure_length = 0
		self.m = 0 
		self.poaching = 0


	# initialize model (van de Leemput and Blackwood-Mumby)
	def initialize_patch_model(self, P0, C0L, C0H, M0L, M0H, M0iL = None, M0iH = None):


		if self.model_type == 'RB':
			self.RB_initialize_patch_model(P0, C0L, C0H, M0L, M0H, M0iL, M0iH)
			return 

		frac_dispersed = (1-self.frac_nomove)*(1/(self.n)) # fraction of fish that disperse to other patches symmetrically

		kP = np.empty((self.n,self.n)) 
		for i in range(self.n):
			for j in range(self.n):
				kP[i][j] = frac_dispersed
				if i == j:
					kP[i][j] = -frac_dispersed*(self.n - 1)
		setattr(self,'kP', kP)

		
		setattr(self, 'P', np.zeros(self.n))
		setattr(self, 'C' , np.empty(self.n))
		setattr(self,'M', np.empty(self.n))
		setattr(self,'dPs', np.empty(self.n))
		setattr(self,'dCs', np.empty(self.n))
		setattr(self,'dMs', np.empty(self.n))
		setattr(self, 'X1', [P0]*self.n + [C0L]*self.n + [M0H]*self.n)
		setattr(self, 'X2', [P0]*self.n + [C0H]*self.n + [M0L]*self.n)  
		

	# initialize Rassweiler-Briggs model
	def RB_initialize_patch_model(self, P0, C0L, C0H, M0vL, M0vH, M0iL, M0iH):

		frac_dispersed = (1-self.frac_nomove)*(1/(self.n)) # fraction of fish that disperse to other patches symmetrically
		# transition matrix for dispersal: element [i,j] of kP describes influx of P from j to i
		kP = np.empty((self.n,self.n))
		for i in range(self.n):
			for j in range(self.n):
				kP[i][j] = frac_dispersed
				if i == j:
					kP[i][j] = -frac_dispersed*(self.n - 1)
		setattr(self,'kP', kP)

		setattr(self, 'P', np.empty(self.n))
		setattr(self, 'C' , np.empty(self.n))
		setattr(self,'Mi', np.empty(self.n))
		setattr(self,'Mv', np.empty(self.n))
		setattr(self,'dPs', np.empty(self.n))
		setattr(self,'dCs', np.empty(self.n))
		setattr(self,'dMis', np.empty(self.n))
		setattr(self,'dMvs', np.empty(self.n))

		setattr(self, 'X1', [P0]*self.n + [C0L]*self.n + [M0vH]*self.n + [M0iH]*self.n)
		setattr(self, 'X2', [P0]*self.n + [C0H]*self.n + [M0vL]*self.n + [M0iL]*self.n)


	# returns the model run for a certain set of parameters 
	def run_model(self, IC_set, t):
		sol = odeint(patch_system, IC_set, t, args = (self, ))
		return sol 

	# make a time series of coral in each patch over time 
	# to show algae or parrotfish, modify line 110 by adding (algae) or subtracting (parrotfish) 'self.n' to 'self.n + i'
	def time_series(self, IC_set, t, save, show, show_legend = False):

		fig, ax = plt.subplots()
		sol = odeint(patch_system, IC_set, t, args = (self, ))
		patch_num = [x for x in range(1, self.n+1)]
		recov_time = self.get_coral_recovery_time(t)
		if recov_time == -1:
			print("ERROR")
			recov_time = 1
		avg = 0

		'''
		avg = 0
		if self.closure_length*self.n / len(t) < 0.5:
		  for year in range(len(t) - len(t) % (self.n*self.closure_length) - self.closure_length,
		  len(t) - len(t)  % (self.n*self.closure_length)):
		    avg += sol[year][self.n]
		  avg = avg / (self.n*self.closure_length + 1)
		else:
		  for patch in range(self.n):
		    avg += sol[len(t) - 1][self.n + patch]
		  avg = avg / self.n
		'''
		
		scaling_array = np.asarray([1 / recov_time]*len(t))

		time_scaled = np.multiply(scaling_array, t)
		for i in range(self.n):
			if show_legend:
			    ax.plot(time_scaled, sol[:, self.n+i],  label = 'patch % d'% (int(i) + 1))
			else:
			    ax.plot(time_scaled, sol[:, self.n+i],  label = avg)
		
		ax.set_xlabel('Time (scaled to coral recovery time)')
		
		ax.set_ylabel('Coral cover (total fraction of area)')
		ax.set_title('Coral cover in each patch over time')
		plt.legend(loc=0)
		ax.set_xlim([0, len(t)/recov_time])
		ax.set_ylim([0, 1])
		# txt="parameters" + "\nfishing when open: " + str(self.f/(1-self.m/self.n)) + "\npercent closure: " + str(self.m/self.n) +"\nclosure time: " + str(self.closure_length)
		# plt.figtext(0.7, .31, txt, wrap=True, fontsize=8)
		if save:
			plt.savefig('TS_' + str(self.model_type) + '_f=' + str(self.f) + '_' + str(self.m) +'out_of_'+ str(self.n) + '_dispersal=' + str(self.frac_nomove) + '_' + str(self.closure_length) + '.jpg')
		if show:
		  plt.show()
		plt.close()

	# heatmap of coral 
	def coral_recovery_map(self, t, fishing_intensity, IC_set = None, filename = None):

		fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, figsize=(int(2/3*self.n), int(0.7*self.n)), gridspec_kw={'height_ratios': [1, self.n]})


		P0, C0L, C0H, M0L, M0H, M0vH, M0vL, M0iH, M0iL = 0.1, 0.04, 0.4, 0.04, 0.4, 0.04, 0.4, 0.04, 0.4


		# slow version  
		# IC_set = self.X2
		MAX_TIME = len(t) # last year in the model run 

		coral_array =  np.zeros(shape=(int(2/3*self.n), int(0.7*self.n))) # array of long-term coral averages
		# CL_array = np.empty(int(0.75*self.n+1)) # array of closure lenghts 
		# m_array = np.empty(int(self.n / 2))  # array of number of closures 
		closure_lengths = np.empty(int(0.7*self.n))
		ms = np.empty(int(2/3*self.n))
		
		def fill_heatmap_cell(self, t, fishing_intensity, IC_set, closure_length, m):
		  
			# closure_length = int(closure_length / 10) + 1
			period = self.n*closure_length
			# set management parameters for this run 
			self.set_mgmt_params(closure_length, fishing_intensity, m, self.poaching)

			# solve ODE system 
			sol = odeint(patch_system, IC_set, t, args = (self, ))
			# average over coral cover of last two full rotations for a single patch (assumes symmetry, may fix that)
			avg = 0



			""" AVERAGING MATH -- NEED TO CHECK AGAIN """ 
			'''
			for year in range(MAX_TIME - MAX_TIME % self.closure_length - self.closure_length, MAX_TIME - MAX_TIME % self.closure_length):
			  for j in range(self.n):
			   avg += sol[year][self.n + j]
			try:
			  avg = avg / self.closure_length
			except:
			  print("division by zero !")
			avg = avg / self.n
			'''
			for year in range(MAX_TIME - period, MAX_TIME):
			  for j in range(self.n):
			   avg += sol[year][self.n + j]
			avg = avg / period
			avg = avg / self.n

			print("made it ")
			# original heatmap average below
			# average over time but not patches
			'''
			if True:
				for year in range(MAX_TIME - MAX_TIME % period - 1*period, 
					MAX_TIME - MAX_TIME % period):
					avg += sol[year][self.n]
				avg = avg / ((1*period) + 1)
			'''
		
			
			return avg
	    


		solutions_pool = mp.Pool(processes = 40) # defaults to num_cpu


		final_coral_covers = Parallel(n_jobs = 40)(delayed(fill_heatmap_cell)(self, t, fishing_intensity, IC_set, closure_length, m) for m in range(int(self.n*2/3)) for closure_length in range(1, int(0.7*self.n + 1)))
		
		print(final_coral_covers)
		print("FINAL CORAL COVERS ABOVE")
		coral_array = np.transpose((np.asarray(final_coral_covers)).reshape((int(2/3*self.n), int(0.7*self.n))))
		print(coral_array)
		print("transposed array above")
		recov_time = self.get_coral_recovery_time(t)
		if recov_time < 0:
			print("oops")
			print(recov_time)
			quit()
		ax2.set_title('Coral cover under periodic closures', fontsize = 30)
		f = lambda y: str(float(self.n*y / recov_time))[0:3]
		new_labels = [f(y) for y in range(1, int(0.7*self.n+1))]

		g = lambda x: str(x / self.n)[0:4]
		
		new_x_labels = [g(x) for x in range(len(ms))]
		# new_x_labels = [0.1, 0.2, 0.303, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
		# print(coral_array)
		
		if self.model_type == 'RB':
			high_coral_eq = 0.5
		elif self.model_type == 'BM':
			high_coral_eq = 0.54
		else: 
			high_coral_eq = 0.7
			
		ax2 = sb.heatmap(coral_array, vmin = 0.0, vmax = high_coral_eq + 0.1, cmap = "viridis", xticklabels = new_x_labels, yticklabels = new_labels, square = True, cbar = False, cbar_kws={"shrink": 0.5}) #YlGnBu for original color scheme
		ax2.invert_yaxis()
		ax2.set_aspect('equal')
		ax2.set_xticks(ax2.get_xticks()[::2]) # display every other tick 
		try:
		  ax2.tick_params(axis='x', fontsize=30)
		  ax2.tick_params(axis='y', fontsize=30)
		except:
		  for a in [ax1, ax2]:
 		    for label in (a.get_xticklabels() + a.get_yticklabels()):
 		      label.set_fontsize(20)
 		      # label.set_fontweight('bold')
		ax2.set_yticks(ax2.get_yticks()[::2]) # display every other tick
		ax2.set_ylabel('Cycle period (in terms of coral recovery time)', fontsize = 30) # x-axis label with fontsize 15
		ax2.set_xlabel('Fraction of seascape closed', fontsize = 30) # y-axis label with fontsize 15
		# ax.yticks(ax.get_yticks(), ax.get_yticks() * 3)
		# ax.set_yticks(rotation=0)
		# plt.show()45


		# ms, closure_lengths = np.meshgrid(ms, closure_lengths)

		# ax = plt.axes(projection='3d')
		# ax.plot_surface(ms, closure_lengths, coral_array,
		#                 cmap='viridis', edgecolor='none')
		crt = self.get_coral_recovery_time(t)
		'''
		if self.model_type == 'vdL':
		  """ Plot isoclines """ 
  		
		  ps  = np.linspace(0,self.n,100)
		  
		  isocline2 = lambda x: 0.1*crt/(x+0.0000001)
		  y  = np.asarray([isocline2(val) for val in ps])
		  ax2.plot(ps, y, 'r--')
  
		  isocline3 = lambda x: 0.4*crt/(x+0.0000001)
		  y  = np.asarray([isocline3(val) for val in ps])
		  ax2.plot(ps, y, 'r--')
		  
		  isocline4 = lambda x: 0.9*crt/(x+0.0000001)
		  y  = np.asarray([isocline4(val) for val in ps])
		  ax2.plot(ps, y, 'y--')
  	
		  ax2.axvline(x=0.375, color = 'w', linestyle = '--')
		'''
		mako = ListedColormap(sb.color_palette('viridis').as_hex())
		
		# plot closure duration == recovery time line if coral is low 
		if IC_set == self.X1:
				print(IC_set)
				ps = np.linspace(0, self.n, 100)
				recovery_time_isocline = lambda x: crt / (x + 0.0000000001)
				y = np.asarray([recovery_time_isocline(x) for x in ps])
				ax2.plot(ps, y, 'w--', linewidth = 3)
		
		""" Make MPA colorbar for comparisons """
		# hacky but works 
		# need to simulate MPA for second colorbar to juxtapose with heatmap...
		# creating another model object within this one seems unnecessarily funky 
		
		z = Model(self.model_type, self.n, 1, mgmt_strat = 'MPA')
		mpa_corals = np.empty(self.n)
		# set initial conditions 
		z.initialize_patch_model(P0, C0L, C0H, M0L, M0H, M0iL, M0iH)
		z.load_parameters() # do this inside initializer
		for i in range(self.n):
			z.set_mgmt_params(500, fishing_intensity, i, self.poaching)
			MPAsol = z.run_model(IC_set, t)
			# print(MPAsol)
			extent = [0, int(2/3*self.n), 0, high_coral_eq + 0.1]
			coral_f = 0
			for j in range(self.n):
				coral_f += MPAsol[len(t) - 1][j+self.n]
			coral_f = coral_f / self.n # patch average
			mpa_corals[i] = coral_f
			# print(mpa_corals)
		
		
		ax1.axis('off')
		ax1.imshow(mpa_corals[np.newaxis][:], cmap=mako, aspect=.60, extent=extent, label='MPA coral cover', vmin=0, vmax=2/3)
		ps  = np.linspace(0,self.n,10)
		
		# ax1.plot(ps, mpa_corals)
		ax1.set_title('Coral cover under MPA')
		try:
		  ax1.title(fontsize = 17)
		except:
		  pass
			# ax1.set_yticks([])
		ax1.set_xlim(extent[0], extent[1])
		# ax1.xticks(fontsize=20)

		position = ax1.get_position()
		position = ax2.get_position()
		# ax1.set_position([0.125, 0.17, 0.75, 1.1])
		# ax1.set_position([position.x0, position.y0 + position.y1, position.x1, 1.1])


		if filename == None:
			plt.show()
		else:
			plt.savefig(str(filename) + '.jpg')
			plt.close()


		# name = 'heatmap_viridis_coral_vs_fishing_' + str(self.model_type) + str(fishing_intensity) + '_' + str(self.n) + '.jpg'
		# plt.savefig(name)
		# plt.close()

	def bistable_zone(self, t, filename = None):
		""" plot final coral cover for different values of fishing effort for two sets of initial conditions """ 
		final_coral_high = np.empty(20)
		final_coral_low = np.empty(20)


		fishing_range = np.linspace(0, 1.2, 20)

		for i, f in enumerate(fishing_range):

			# set management parameters 
			self.set_mgmt_params(0, f, 0, self.poaching)

			# make high start solution
			high_sol = odeint(patch_system, self.X2, t, args = (self, ))

			# make low start solution 
			low_sol = odeint(patch_system, self.X1, t, args = (self, ))

			# note: this only works without periodic oscillations, which this plot assumes are not present 
			yrs = len(t)
			final_coral_high[int(i)] = high_sol[yrs - 1][self.n]

			final_coral_low[int(i)] = low_sol[yrs - 1][self.n]

		plt.plot(fishing_range, final_coral_low, label = 'coral starts low', color = 'blue', linewidth = 3)
		plt.plot(fishing_range, final_coral_high, label = 'coral starts high' , color = 'green', linewidth = 3)
		plt.xlabel('fishing effort', fontsize = 'medium')
		plt.ylabel('long-term coral cover', fontsize = 'medium')
		

		if filename == None:
			plt.show()
		else:
			plt.savefig(str(filename) + '.jpg')
			plt.close()

	# plots coral versus closure length instead of coral versus fraction closed 
	def coral_vs_CL(self, t, fishing_intensity, IC_set = None):

		durations = [i for i in range(1,100)]

		for m in range(0, self.n):
			coral_covers = []
			for duration in durations:

				self.set_mgmt_params(duration, fishing_intensity, m, self.poaching)

				sol = odeint(patch_system, IC_set, t, args = (self, ))

				avg = 0
				# average over last cycle for one patch
				for year in range(len(t) - duration*self.n, len(t)):
					avg += sol[year][self.n]
				avg = avg / (duration*self.n)

				# avg = avg / self.n

				coral_covers.append(avg)

			plt.title('coral cover versus closure duration')
			plt.xlabel('closure duration in years')
			plt.ylabel('coral cover as fraction of seascape')
			plt.ylim([0,1])

			plt.plot(durations, coral_covers, label = 'coral when {} patches are closed'.format(m))
		plt.legend(loc = 0)
		plt.show()



	def find_unstable_equilibrium(self, t, lowC = 0.1, highC = 0.7, recursion_depth = 0):
		""" binary search for unstable equilibrium """ 


		midpoint = (lowC + highC) / 2
		print(midpoint)


		IC_mid = [0.1]*self.n + [midpoint]*self.n + [0.04]*self.n # verify this 

		# run model starting from midpoint -- could reduce runtime by making t smaller 
		mid_sol = odeint(patch_system, IC_mid, t, args = (self, ))

		if recursion_depth > 10:
			print("Close enough....")
			return midpoint

		# if coral cover grows from the midpoint, the equilibrium is above it
		if mid_sol[len(t) - 1][1] - midpoint > 0:
			print("going up...")
			new_recursion_depth = recursion_depth + 1
			return self.find_unstable_equilibrium(t, lowC = midpoint, recursion_depth = new_recursion_depth)
		# if coral cover declines from the midpoint, the equilibrium is below it 
		elif mid_sol[len(t) - 1][1] - midpoint < 0: 
			print("going down...")
			new_recursion_depth = recursion_depth + 1
			return self.find_unstable_equilibrium(t, highC = midpoint, recursion_depth = new_recursion_depth)
		else: # unstable equilibrium found!
			return midpoint 


	# plot coral versus fraction closed for different periods 
	def scenario_plot(self, t, fishing_intensity, IC_set, filename = None, show_legend = False):
		P0, C0L, C0H, M0L, M0H, M0vH, M0vL, M0iH, M0iL = 0.1, 0.04, 0.4, 0.04, 0.4, 0.04, 0.4, 0.04, 0.4

		crt = self.get_coral_recovery_time(t)
		if crt == -1:
			print("coral recovery time too high")
			quit()
		print("crt: ", crt)
		
		multipliers = np.asarray([0.25, 0.5, 1, 2, 4])
		crts = np.asarray([crt]*5)
		# do we want the cycle period or the individual closure duration to match these?
		periods = np.multiply(crts, multipliers)# [0.1*crt, 0.5*crt, 1*crt, 2*crt, 4*crt]  # parametrize in terms of coral growth time? 
			
		
		# there is a cooler way to do colors than this 
		# color_sequence = {periods[0]: '#1f77b4', periods[1]: '#aec7e8', periods[2]: '#ff7f0e', periods[3]:'#ffbb78', periods[4]:'#2ca02c'}
		color_sequence = {periods[0]: '#fa0000', periods[1]: '#fa0092', periods[2]: '#cc00fa' , periods[3]:'#8500fa', periods[4]:'#1400fa'}
		print("HERE WE GO")
		print(self.closure_length)
		MAX_TIME = len(t)

		# this loops over each closure fraction 
		def loop_over_m_vals(self, t, period, fishing_intensity, IC_set, m):
			# print("SCENARIO PLOT PERIOD IS ", period)
			# set management parameters for this run  -- Divided by n in original??
			self.set_mgmt_params(period / self.n, fishing_intensity, m, self.poaching)
			self.set_mgmt_params(period / self.n, fishing_intensity, m, self.poaching)
			sol = odeint(patch_system, IC_set, t, args = (self, ))
			avg = 0
			# print(sol[MAX_TIME - 1])
			
			


			""" multiple averaging methods below, tradeoffs between speed and accuracy -- this is the most likely source of bugs... """
			'''
			# fast averaging, only looks at a single patch and averages over last period
			if self.frac_nomove == 0.99:
			  if period / MAX_TIME < 0.5:
  				for year in range(MAX_TIME - MAX_TIME % period - 2*period, 
  					MAX_TIME - MAX_TIME % period):
  					avg += sol[year][self.n]
  				avg = avg / ((2*period) + 1)
			  else:
  			  # average over patches but not time
  				for patch in range(self.n):
  					avg += sol[MAX_TIME - MAX_TIME%period - 1][self.n + patch]
  				avg = avg / self.n
			else:
			  for year in range(MAX_TIME - period, MAX_TIME):
  			   avg += sol[year][self.n]
			  avg = avg / period
			  if avg < 0.2:
			    print("what's happenin here?")
			    print(sol[:])
			'''
			# added april 14 -- JUNE FIFTH NOTE, THIS IS USED FOR CURRENT HEATMAPS 
			'''
			if self.closure_length < 1:
			  print("branch entered")
			  self.closure_length = 1 
			for year in range(int(MAX_TIME - (self.closure_length) - (MAX_TIME%((self.closure_length)))), int(MAX_TIME - (MAX_TIME%(int(self.closure_length))))):
			  for j in range(self.n):
			   avg += sol[year][self.n + j]
			avg = avg / float(self.n)
			avg = avg / float(self.closure_length)
			print("The average is ", avg)
			'''
			# JUNE FIFTH VERSION, EDITED BELOW 
			for year in range(MAX_TIME - 2*int(period)  - MAX_TIME % int(period), MAX_TIME - MAX_TIME % int(period)):
			  for j in range(self.n):
			   avg += sol[year][self.n + j]
			avg = avg / (2*period)
			avg = avg / self.n
      
			'''
			for year in range(MAX_TIME - period, MAX_TIME):
			  avg += sol[year][self.n]
			avg = avg / period
			'''
			'''
			if avg < 0.2:
			  print("what's happenin here?")
			  print(sol[:])
			'''
			# trying this average bc it works for heatmap: (april 20)
			'''
			if period / MAX_TIME < 0.5:
				for year in range(MAX_TIME - MAX_TIME % period - 2*period, 
					MAX_TIME - MAX_TIME % period):
					avg += sol[year][self.n]
				avg = avg / ((2*period) + 1)
			else:
			  # average over patches but not time
				for patch in range(self.n):
					avg += sol[MAX_TIME - MAX_TIME%period - 1][self.n + patch]
				avg = avg / self.n
			'''
			'''
			for patchnum in range(self.n):
				avg += sol[MAX_TIME-MAX_TIME % period][self.n + patchnum]
			avg = avg / self.n
			'''
			# print("Average coral cover: ")
			# print(avg)
		  
			
			# slow averaging: averages over all patches over last complete cycle
			'''
			for patchnum in range(self.n):
			  
			  # avg += sol[int(MAX_TIME - MAX_TIME % period) - 1][self.n + patchnum] # -- alternatively, skip time averaging and just look at all patches at end of last cycle 
			  # avg += sol[int(MAX_TIME - 1)][self.n + patchnum]
			 
			  for year in range(MAX_TIME - MAX_TIME % period - period, 
          MAX_TIME - MAX_TIME % period):
				
				  avg += sol[year][self.n + patchnum]
			avg = avg / (period+1)
		
			avg = avg / self.n
			'''
			
			'''
			# June 7 20:53 trying this average 
			if period / MAX_TIME < 0.5:
				for year in range(MAX_TIME - MAX_TIME % period - period, 
					MAX_TIME - MAX_TIME % period):
					avg += sol[year][self.n]
				avg = avg / ((period) + 1)
			else:
			  # average over patches but not time
				for patch in range(self.n):
					avg += sol[MAX_TIME - 1][self.n + patch]
				avg = avg / self.n
			'''

			# avg = avg / period
			# avg = avg / self.n
			
			'''
			# average over time but not patches 
			if period / MAX_TIME < 0.5:
				for year in range(MAX_TIME - MAX_TIME % period - period, 
					MAX_TIME - MAX_TIME % period):
					avg += sol[year][self.n]
				avg = avg / ((period) + 1)
			else:
			  # average over patches but not time
				for patch in range(self.n):
					avg += sol[MAX_TIME - 1][self.n + patch]
				avg = avg / self.n
			'''
			'''
			# April 14 averaging- RB still looking weird
			# no need to average over patches assuming symmetric behavior 
			for year in range( MAX_TIME-MAX_TIME%period -2*period, MAX_TIME-MAX_TIME%period):
			  # for patch in range(self.n):
			  avg += sol[year][self.n]
			# avg = avg / self.n
			avg = avg / (2*period)
	    '''
	    
			print("Average ",  m, "is ", avg, "when period is", period)
			final_coral[m] = avg
			ms[m] = m

			return [m, avg]
	   
   
		print("made it here")
		for i, period in enumerate(periods):
			final_coral = np.empty(self.n)
			print("CLOSURE LENGTH IN YEARS: ", float(period) / float(self.n))
			print("PERIOD IS ", period) 
			start_year = int(len(t) - float(period) / float(self.n) - len(t)%((float(period)/float(self.n))))
			print("START YEAR IS ", start_year)
			end_year = int(len(t) - len(t)%((float(period)/float(self.n))))
			print("END YEAR IS ", end_year)
			ms = np.empty(self.n)
			solutions_pool = mp.Pool(processes = 40) # defaults to num_cpu
			try:
			  res = Parallel(n_jobs = 40)(delayed(loop_over_m_vals)(self, t, period, fishing_intensity, IC_set, m) for m in range(self.n))
			except ZeroDivisionError:
			  print("divide by  zero ")
			for j in range(self.n):
			  ms[j] = res[j][0]
			  final_coral[j] = res[j][1]
			  if final_coral[j] < 0.2:
			    print("period of ", period)
			    print("final_coral less than 0.2 when m is: ", ms[j])
			print(final_coral)
			print(ms, "is ms")
    	# plot result for this period
    	
			width_sequence = {periods[0]: 1.5, periods[1]: 2, periods[2]: 2.5, periods[3]: 3, periods[4]: 3.5}


			if show_legend:
			  plt.plot(ms / self.n, final_coral, label = 'period = %s' % str(multipliers[i]), color = color_sequence[period], linewidth=width_sequence[period])
			else:
			   plt.plot(ms / self.n, final_coral, label = None, color = color_sequence[period], linewidth=width_sequence[period])# no label
  
    # plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.xlabel('Fraction closed')
		plt.ylabel('Final coral Cover')
		plt.rc('axes', labelsize = 12)
		plt.rc('axes', titlesize = 12)
		# plt.title('Final coral state across closure scenarios - ' + str(self.model_type)) -- no titles 
# plt.show()



		# make MPA line 
		z = Model(self.model_type, self.n, self.frac_nomove, mgmt_strat = 'MPA')
		mpa_corals = np.empty(self.n)
		# set initial conditions 
		z.initialize_patch_model(P0, C0L, C0H, M0L, M0H, M0iL, M0iH)
		z.load_parameters() # do this inside initializer
		
		# loop over frac closed 
		for i in range(self.n): 
			z.set_mgmt_params(500, fishing_intensity, i, self.poaching)
			# z.time_series(z.X2, t, save = True, show = False)
			MPAsol = z.run_model(IC_set, t) 
			
			# loop over patches 
			total = 0 
			for j in range(self.n):
				coral_f = 0
				coral_f += MPAsol[len(t) - 1][j+self.n]
				total += coral_f
			total = total / self.n
			mpa_corals[i] = total

		arr = np.linspace(0, 1, self.n)
		if show_legend:
		  plt.plot(arr, mpa_corals, label = 'MPA', color = 'black')
		else:
		  plt.plot(arr, mpa_corals, label = None, color = 'black')
		plt.xlim([0, 0.66])
		if IC_set == self.X1:
			plt.legend(loc=1)
		else:
			plt.legend(loc=3)
	
		# plt.show()
		
		if filename == None:
			plt.savefig('newest_scenario_plot' + str(self.model_type) + '_' + str(fishing_intensity) + '_' + str(IC_set[self.n]) + '_' + str(self.poaching) + '_' + str(self.frac_nomove)+'.jpg')
		else:
			plt.savefig(str(filename) + '.jpg')

		plt.close()





	# calculate coral recovery time based on model type 
	def get_coral_recovery_time(self, t):

		P0, C0L, C0H, M0L, M0H, M0vH, M0vL, M0iH, M0iL = 0.1, 0.04, 0.4, 0.04, 0.4, 0.04, 0.4, 0.04, 0.4


		z = Model(self.model_type, self.n, 1, mgmt_strat = 'MPA')
		# set initial conditions 
		z.initialize_patch_model(P0, C0L, C0H, M0L, M0H, M0iL, M0iH)
		z.load_parameters() # do this inside initializer

		z.set_mgmt_params(500, 0, 0, 0)

		MPAsol = z.run_model(z.X1, t)
		coralsol = MPAsol[:, self.n]
		list_of_labels = ['fish']*self.n + ['coral']*self.n + ['algae']*self.n
		# plt.plot(t, coralsol, label = 'coral')


		# plt.show()
		# np.take_along_axis(a, ai, axis=1)
		# plt.legend(loc=0)

		if self.model_type == 'RB':
			high_coral_eq = 0.5
		elif self.model_type == 'BM':
			high_coral_eq = 0.54
		else: 
			high_coral_eq = 0.7

		coral_recovery_time = -1
		for i, state in enumerate(coralsol):
			if state > high_coral_eq:
				coral_recovery_time = i
				break

		return coral_recovery_time + 10



	def load_parameters(self):
		if self.model_type == 'vdL':
			params = {
			"r": 0.3,
			"i_C" : 0.05,
			"i_M" : 0.05,
			"ext_C" : 0.0001,
			"ext_P" : 0.0001,
			"gamma" : 0.8,
			"d" : 0.1,
			"g" : 1,
			"s" : 1,
			"sigma" : .5, #strength of coral-herbivore feedback
			"eta" : 2, #strength of algae-herbivore feedback
			"alpha" : 0.5, #strength of algae-coral feedback 
			"P0" : 0.1,
			"C_HI" : .4,
			"M_LO" : .04,
			"C_LO" : .04,
			"M_HI" : .4
			}

		elif self.model_type == 'vdL_MC':
			params = {
			"r": 0.3,
			"i_C" : 0.05,
			"i_M" : 0.05,
			"ext_C" : 0.0001,
			"ext_P" : 0.0001,
			"gamma" : 0.8,
			"d" : 0.1,
			"g" : 1,
			"s" : 1,
			"sigma" : 0, #strength of coral-herbivore feedback
			"eta" : 0, #strength of algae-herbivore feedback
			"alpha" : 0.5 #strength of algae-coral feedback 
			}

		elif self.model_type == 'vdL_MP':
			params = {
			"r": 0.3,
			"i_C" : 0.05,
			"i_M" : 0.05,
			"ext_C" : 0.0001,
			"ext_P" : 0.0001,
			"gamma" : 0.8,
			"d" : 0.1,
			"g" : 1,
			"s" : 1,
			"sigma" : 0, #strength of coral-herbivore feedback
			"eta" : 2, #strength of algae-herbivore feedback
			"alpha" : 0 #strength of algae-coral feedback 
			}

		elif self.model_type == 'vdL_PC':
			params = {
			"r": 0.3,
			"i_C" : 0.05,
			"i_M" : 0.05,
			"ext_C" : 0.0001,
			"ext_P" : 0.0001,
			"gamma" : 0.8,
			"d" : 0.1,
			"g" : 1,
			"s" : 1,
			"sigma" : .5, #strength of coral-herbivore feedback
			"eta" : 0, #strength of algae-herbivore feedback
			"alpha" : 0, #strength of algae-coral feedback 
			}

		elif self.model_type == 'BM':
			params = {
			"gamma" : 0.8,
			"beta" : 1,
			"alpha" : 1,
			"s" : 0.49,
			"r" : 1,
			"d" : 0.44,
			"a" : 0.1,
			"i_C" : 0.05,
			"i_M" : 0.05
			}

		elif self.model_type == 'RB':
			params = {
			"phiC" : 0.001, #open recruitment of coral
			"phiM" : 0.0001, #open recruitment of macroalgae

			"rM" : 0.5, #production of vulnerable macroalgae from invulnerable stage"

			#growth rates
			"gTC" : 0.1, #combined rate of growth and local recruitment of corals over free space 
			"gTV" : 0.2, #growth rate of vulnerable macroalgae over free space 
			"gTI" : 0.4, #growth rate of invulnerable macroalgae over free space 
			"rH" : 0.49,

			"gamma" : 0.4, #growth of macroalgae over coral vs free space
			"omega" : 2, #maturation rate of macroalgae from vulnerable to invulnerable class "

			#death rates
			"dC" : 0.05, #death rate of coral 
			"dI" : 0.4, #death rate of invulnerable macroalgae
			"dV" : 0.4, #death rate of vulnerable macroalgae per unit biomass of herbivores "

			"K" : 20, 
			"Graze" : 0.58
			}

		for name, val in params.items():
		
			setattr(self, name, val)

	# management parameter setter 
	def set_mgmt_params(self, closure_length, f, m, poaching):
		self.closure_length = closure_length
		self.f = f
		self.m = m
		self.poaching = poaching 




def patch_system(X, t, system_model):
	P_influx = [0]*system_model.n

	for i in range(system_model.n):

		# add influx at end AFTER the parrotfish have been initialized so it isn't populated with random values
		for j in range(system_model.n):
			P_influx[i] += (system_model.kP[i][j]) * X[j]  # X[j] first n entries are the parrotfish pops

		# this could be structured more nicely
		if system_model.model_type == 'RB':
			
			results = rass_briggs(X, t, i, system_model, P_influx)
			system_model.dPs[i] = results[0]
			system_model.dCs[i] = results[1]
			system_model.dMvs[i] = results[2]
			system_model.dMis[i] = results[3]

		elif system_model.model_type == 'BM':
			
			results = blackwood(X, t, i, system_model, P_influx)
			system_model.dPs[i] = results[0]
			system_model.dCs[i] = results[1]
			system_model.dMs[i] = results[2]

		elif system_model.model_type == 'vdL_PC':
			
			results = leemput(X, t, i, system_model, P_influx)
			system_model.dPs[i] = results[0]
			system_model.dCs[i] = results[1]
			system_model.dMs[i] = results[2]

		elif system_model.model_type == 'vdL_MP':
			results = leemput(X, t, i, system_model, P_influx)
			system_model.dPs[i] = results[0]
			system_model.dCs[i] = results[1]
			system_model.dMs[i] = results[2]

		elif system_model.model_type == 'vdL_MC':
			
			results = leemput(X, t, i, system_model, P_influx)
			system_model.dPs[i] = results[0]
			system_model.dCs[i] = results[1]
			system_model.dMs[i] = results[2]

		elif system_model.model_type == 'vdL': # all feedbacks active 

			results = leemput(X, t, i, system_model, P_influx)
			# print(leemput(X, t, i, system_model, P_influx))
			system_model.dPs[i] = results[0]
			system_model.dCs[i] = results[1]
			system_model.dMs[i] = results[2]
			
		else:
			print("Bad input, defaulting to Blackwood-Mumby!")
			results = blackwood(X, t, i, system_model, P_influx)
			system_model.dPs[i] = results[0]
			system_model.dCs[i] = results[1]
			system_model.dMs[i] = results[2]

		
	if system_model.model_type == 'RB':
		return np.concatenate((system_model.dPs, system_model.dCs, system_model.dMvs, system_model.dMis), axis = 0)
	else:
		return np.concatenate((system_model.dPs, system_model.dCs, system_model.dMs), axis = 0)

def rass_briggs(X, t, i, system_model, P_influx):


	P, C, Mv, Mi = X.reshape(4, system_model.n)
	T = 1 - C[i] - Mv[i] - Mi[i]
	
	
	dP = P_influx[i]+system_model.rH*P[i]*(1-P[i]/system_model.K) - system_model.f*P[i] *(square_signal(t, system_model.closure_length, i, system_model.m, system_model.n, system_model.poaching, system_model.mgmt_strat))
	

	dC = (system_model.phiC*T) + system_model.gTC*T*C[i] - system_model.gamma*system_model.gTI*Mi[i]*C[i] - system_model.dC*C[i]

	dMv = system_model.phiM*T + system_model.rM*T*Mi[i] + system_model.gTV*T*Mv[i] - system_model.dV*Mv[i] - P[i]*Mv[i]*system_model.Graze - system_model.omega * Mv[i]
	dMi = system_model.omega*Mv[i] + system_model.gTI*T*Mi[i] + system_model.gamma*system_model.gTI*Mi[i]*C[i] - system_model.dI*Mi[i]
	# print(P, C, Mv, Mi)
	# print(system_model.f)#/(1-system_model.m/system_model.n))# *P[i] *(square_signal(t, system_model.closure_length, i, system_model.m, system_model.n, system_model.poaching)))
	return [dP, dC, dMv, dMi]

	
	# dP = P_influx[i] + system_model.rH*P[i]*(1-P[i]/system_model.K) - system_model.f/(1-system_model.m/system_model.n)*P[i] *(square_signal(t, system_model.closure_length, i, sysem_model.m, system_model.n, system_model.poaching))
# 	dC = (system_model.phiC*T) + gTC*T*C - gamma*gTI*Mi*C - dC*C
# 	dMv = phiM*T + rM*T*Mi + gTV*T*Mv - dV*Mv - P*Mv*Graze - omega * Mv

# 	dMi = omega*Mv + gTI*T*Mi + gamma*gTI*Mi*C - dI*Mi
	# return [dP, dC, dMv, dMi]


	# check input
	# return None 

def K(sigma, C):
		return (1-sigma)+sigma*C
def BMK(C):
	return 1 - 0.5*C


# turns fishing on and off 
def square_signal(t, closure_length, region, m, n, poaching, mgmt_strat = 'periodic'):

	if mgmt_strat == 'periodic':

		if closure_length != 0: 
			start = int((t % (n*closure_length))/closure_length)
		else:
			start = 0

		if start+m-1 >= n:
			end = (start + m - 1)%n

		else:
			end = (start + m - 1)
		# print("START: ", start)
		# print("END: ", end)
		# print("REGION NUM ", region)


		""" CHECK POACHING IMPLEMENTATION: """
		# if region is closed (between start region and end region, inclusive), then we only have poaching
		if region >= start and region <= end:
			return poaching
		elif start + m - 1 >= n and (region >= start or region <= end):
			return poaching
		else:
			# if region is open:
			return (1-(m/n)*poaching)/(1 - (m/n))

	elif mgmt_strat == 'MPA':
		if m == 0:
			return 1 # if we close nothing, signal does not modify fishing intensity
		if m == n:
			return poaching # if we close everything, only poaching remains  
		if region <= m:
			return poaching  # closed region 
		else: 
			return (1 - (m / n) * poaching) / (1 - (m/n)) # open region 

	
#this signal function is not quite working yet 
def sigmoid_signal(t, period, p):
	if period == 0:
		return 0
	else:
		return 1.0 / (1 + math.exp(-(t % period - p * period)))

# fishing density dependence 
def fishing(parrotfish, f):
	steepness = 25
	shift = 0.2
	return f/(1+math.exp(-steepness*(parrotfish-shift)))


def blackwood(X, t, i, system_model, P_influx):
	
	P, C, M = X.reshape(3, system_model.n)

	# dP = s*P[i]*(1 - (P[i] / (beta*system_model.K(C[i])))) - system_model.fishing(P[i], f)*P[i]*system_model.square_signal(t, closure_length, i, m, n, poaching)
	dP = P_influx[i] + system_model.s*P[i]*(1 - (P[i] / (system_model.beta*BMK(C[i])))) - system_model.f*P[i]*square_signal(t, system_model.closure_length, i, system_model.m, system_model.n, system_model.poaching, system_model.mgmt_strat)
	dC = system_model.r*(1-M[i]-C[i])*C[i]-system_model.d*C[i] - system_model.a*M[i]*C[i] + 0.0005*system_model.i_C*(1-M[i]-C[i])
	# need to define g(P) before this model is used 
	dM = system_model.a*M[i]*C[i] - system_model.alpha*P[i]/system_model.beta*M[i] *(1/(1-C[i])) + system_model.gamma*M[i]*(1-M[i]-C[i]) + 0.0075*system_model.i_M*(1-M[i]-C[i])

	# return np.concatenate((dPs, dCs, dMs), axis=0)
	return [dP, dC, dM]
	
	'''
	P,C,M = X.reshape(3, system_model.n) # will reshaping work since we are passing arrays of length n? 
	# dC = P_influx[i]+ system_model.s*P[i]*(1 - (P[i] / K(system_model.sigma,C[i]))) - fishing(P[i], system_model.f)*P[i] *(square_signal(t, system_model.closure_length, i, system_model.m, system_model.n, system_model.poaching))
	dP = P_influx[i]+ system_model.s*P[i]*(1 - (P[i] / K(system_model.sigma,C[i]))) - system_model.f/(1-system_model.m/system_model.n)*P[i] *(square_signal(t, system_model.closure_length, i, system_model.m, system_model.n, system_model.poaching))
	# need separtate K function for blackwood
	# print(square_signal(t, system_model.closure_length, i, system_model.m, system_model.n, system_model.poaching))
	dC = (system_model.r*C[i])*(1-M[i]-C[i]) - system_model.a*M[i]*C[i] - system_model.d*C[i] # recruitment needed?
	
	dM = (system_model.gamma*M[i])*(1-M[i]-C[i])-system_model.g(P[i])*(1/(1-C[i]))*M[i]


	return [dP, dC, dM]
	'''

# will need to pass [self.P, self.C, self.M] to this for it to work 
def leemput(X, t, i, system_model, P_influx): 
  # check input 
  P,C,M = X.reshape(3, system_model.n) 
  # dC = P_influx[i]+ system_model.s*P[i]*(1 - (P[i] / K(system_model.sigma,C[i]))) - fishing(P[i], system_model.f)*P[i] *(square_signal(t, system_model.closure_length, i, system_model.m, system_model.n, system_model.poaching))
  
  dP = P_influx[i]+ system_model.s*P[i]*(1 - (P[i] / K(system_model.sigma,C[i]))) - system_model.f*P[i]*(square_signal(t, system_model.closure_length, i, system_model.m, system_model.n, system_model.poaching, system_model.mgmt_strat))
  # print(P_influx)
  # print(square_signal(t, system_model.closure_length, i, system_model.m, system_model.n, system_model.poaching))
  dC = (system_model.i_C + system_model.r*C[i])*(1-M[i]-C[i])*(1-system_model.alpha*M[i]) - system_model.d*C[i]
  dM = (system_model.i_M+system_model.gamma*M[i])*(1-M[i]-C[i]) - system_model.g*M[i]*P[i]/(system_model.g*system_model.eta*M[i]+1)
  
  return [dP, dC, dM]
  

def main():
  
  print("Running Model...")
  
  # time 
  yrs = 1000 #total amount of time
  t = np.linspace(0, yrs, yrs) #timestep array -- same number of timesteps as years 
  
  RB_yrs = 2000 #total amount of time -- changed from 3000 for runtime...
  RB_t = np.linspace(0, RB_yrs, RB_yrs) #timestep array -- same number of timesteps as years 
  
  
  # initial conditions 
  P0, C0L, C0H, M0L, M0H, M0vL, M0vH, M0iL, M0iH = 0.1, 0.04, 0.4 + 0.2, 0.04, 0.4, 0.04, 0.2, 0.04, 0.2
  
  P0_RB = 20 
  
  # midpoint levels of fishing from plots of bistable zones for each model
  vdl_fishing_midpoint = (0.1593 + 0.475) / 2
  BM_fishing_midpoint = ( 0.2372 + 0.394 ) / 2  

  RB_fishing_midpoint = (0.2367 + 0.3945) / 2
  
  
  blackwood_mumby = Model('BM', 12, 0.95, mgmt_strat = 'periodic')
  van_de_leemput = Model('vdL', 12, 0.95, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 12, 0.98, mgmt_strat = 'periodic')
  
  
 
  blackwood_mumby.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0_RB, C0L, 0.5, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  
  
  # get coral recovery times for each model
  BM_crt = blackwood_mumby.get_coral_recovery_time(t)
  RB_crt = rass_briggs.get_coral_recovery_time(RB_t)
  vdL_crt = van_de_leemput.get_coral_recovery_time(t)
  
  
  rass_briggs.set_mgmt_params(2*RB_crt/12, RB_fishing_midpoint, 3, 0)
  rass_briggs.time_series(rass_briggs.X2, RB_t, save = True, show = False)
  rass_briggs.set_mgmt_params(2*RB_crt/12, RB_fishing_midpoint, 4, 0)
  rass_briggs.time_series(rass_briggs.X2, RB_t, save = True, show = False) 
  rass_briggs.set_mgmt_params(2*RB_crt/12, RB_fishing_midpoint, 5, 0)
  rass_briggs.time_series(rass_briggs.X2, RB_t, save = True, show = False) 
  rass_briggs.set_mgmt_params(2*RB_crt/12, RB_fishing_midpoint, 6, 0)
  rass_briggs.time_series(rass_briggs.X2, RB_t, save = True, show = False) 
  

  
  
  # rass_briggs.time_series(rass_briggs.X2, t, save = False, show = True) 
  # rass_briggs.time_series(rass_briggs.X2, t, save = False, show = True) 
  # TIME SERIES
  '''
  blackwood_mumby.set_mgmt_params(.25/12*BM_crt, BM_fishing_midpoint,1, 0) # set closure duration, fishing level, number of closures, and poaching, in that order 
  blackwood_mumby.time_series(blackwood_mumby.X2, t, save = True, show = False) 
  blackwood_mumby.set_mgmt_params(BM_crt/12, BM_fishing_midpoint,2, 0) # set closure duration, fishing level, number of closures, and poaching, in that order 
  blackwood_mumby.time_series(blackwood_mumby.X2, t, save = True, show = False) 
  blackwood_mumby.set_mgmt_params(BM_crt/6, BM_fishing_midpoint,3, 0) # set closure duration, fishing level, number of closures, and poaching, in that order 
  blackwood_mumby.time_series(blackwood_mumby.X2, t, save = True, show = False) 
  print("PROGRESS CHECK")
  
  blackwood_mumby.set_mgmt_params(.1*BM_crt, BM_fishing_midpoint,4, 0) # set closure duration, fishing level, number of closures, and poaching, in that order 
  blackwood_mumby.time_series(blackwood_mumby.X2, t, save = True, show = False) 
  blackwood_mumby.set_mgmt_params(.1*BM_crt, BM_fishing_midpoint,5, 0) # set closure duration, fishing level, number of closures, and poaching, in that order 
  blackwood_mumby.time_series(blackwood_mumby.X2, t, save = True, show = False) 
  blackwood_mumby.set_mgmt_params(.1*BM_crt, BM_fishing_midpoint,6, 0) # set closure duration, fishing level, number of closures, and poaching, in that order 
  blackwood_mumby.time_series(blackwood_mumby.X2, t, save = True, show = False) 
  blackwood_mumby.set_mgmt_params(.1*BM_crt, BM_fishing_midpoint,7, 0) # set closure duration, fishing level, number of closures, and poaching, in that order 
  blackwood_mumby.time_series(blackwood_mumby.X2, t, save = True, show = False) 
  
  blackwood_mumby.set_mgmt_params(.2*BM_crt, BM_fishing_midpoint,2, 0) # set closure duration, fishing level, number of closures, and poaching, in that order 
  blackwood_mumby.time_series(blackwood_mumby.X2, t, save = True, show = False) 
  blackwood_mumby.set_mgmt_params(.2*BM_crt, BM_fishing_midpoint,4, 0) # set closure duration, fishing level, number of closures, and poaching, in that order 
  blackwood_mumby.time_series(blackwood_mumby.X2, t, save = True, show = False) 
  blackwood_mumby.set_mgmt_params(.2*BM_crt, BM_fishing_midpoint,3, 0) # set closure duration, fishing level, number of closures, and poaching, in that order 
  blackwood_mumby.time_series(blackwood_mumby.X2, t, save = True, show = False) 
  '''
  
  # rass_briggs.set_mgmt_params(1*RB_crt, RB_fishing_midpoint, 0, 0)
  # rass_briggs.time_series(rass_briggs.X2, t, save = False, show = True) 
  
  van_de_leemput.set_mgmt_params(vdL_crt, vdl_fishing_midpoint, 1, 0)
  # van_de_leemput.time_series(van_de_leemput.X1, t, save = True, show = False, show_legend = True) 
  
  van_de_leemput.set_mgmt_params(2.5*vdL_crt, vdl_fishing_midpoint, 1, 0)
  # van_de_leemput.time_series(van_de_leemput.X1, t, save = True, show = False) 
  
  # blackwood_mumby.bistable_zone(t, filename = '12patchJune23_BM_hysteresis')
  # van_de_leemput.bistable_zone(t, filename = '12patchJune23_vdL_hysteresis')
  # rass_briggs.bistable_zone(RB_t, filename = '12patchJune23_RB_hysteresis')
  
  blackwood_mumby = Model('BM', 30, 0.95, mgmt_strat = 'periodic') 
  van_de_leemput = Model('vdL', 30, 0.95, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 30, 0.95, mgmt_strat = 'periodic')
  
  blackwood_mumby.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0_RB, C0L, C0H, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  
  print("Generating recovery heatmaps...")
  
  # blackwood_mumby.coral_recovery_map(t, BM_fishing_midpoint, blackwood_mumby.X1, filename = '12patch_June23_0.95_TruncBM_HEATMAP_LOW')
  # blackwood_mumby.coral_recovery_map(t, BM_fishing_midpoint, blackwood_mumby.X2, filename = 'June14_0.95_TruncBM_HEATMAP_HIGH')
  # van_de_leemput.coral_recovery_map(t, vdl_fishing_midpoint, van_de_leemput.X1, filename = '12patchJune23_0.95_non_isoclineTruncVDL_HEATMAP_LOW')
  # rass_briggs.coral_recovery_map(RB_t, RB_fishing_midpoint, rass_briggs.X1, filename = '12patchJune23_0.95_fixedTruncRB_HEATMAP_LOW')
  # van_de_leemput.coral_recovery_map(t, vdl_fishing_midpoint, van_de_leemput.X2, filename = '12patchJune23_0.95_non_isoclineTruncVDL_HEATMAP_HIGH')
  # rass_briggs.coral_recovery_map(RB_t, RB_fishing_midpoint, rass_briggs.X2, filename = '12patchJune23_0.95_fixedTruncRB_HEATMAP_HIGH')
  
  # SCENARIO PLOTS 
  
  print("Generating scenario plots")
  # 99 percent do not disperse
  blackwood_mumby = Model('BM', 12, 0.98, mgmt_strat = 'periodic') 
  van_de_leemput = Model('vdL', 12, 0.98, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 12, 0.98, mgmt_strat = 'periodic')
  
  blackwood_mumby.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0_RB, C0L, C0H, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  
  ICs = van_de_leemput.X1
  van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patch_June23_vdL_12patch_ScenarioPlot_2PercentDispersal_StartingLow')
  ICs = van_de_leemput.X2
  van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patch_June23_vdL_12patch_ScenarioPlot_2PercentDispersal_StartingHigh', show_legend = True)
  
  ICs = rass_briggs.X2
  rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patch_June23_RB_12patch_ScenarioPlot_2PercentDispersal_StartingHigh')
  ICs = rass_briggs.X1
  rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patch_June23_RB_12patch_ScenarioPlot_2PercentDispersal_StartingLow')
  ICs = blackwood_mumby.X1
  blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patch_June23_BM_12patch_ScenarioPlot_2PercentDispersal_StartingLow')
  ICs = blackwood_mumby.X2
  blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patch_June23_BM_12patch_ScenarioPlot_2PercentDispersal_StartingHigh')
   # 95 percent do not disperse, 30 patches, runnin three times longer than scenario plots 
  
  # 90 percent do not disperse
  blackwood_mumby = Model('BM', 12, 0.9, mgmt_strat = 'periodic') 
  van_de_leemput = Model('vdL', 12, 0.9, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 12, 0.9, mgmt_strat = 'periodic')
  
  blackwood_mumby.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0, C0L, 0.6, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  
  ICs = van_de_leemput.X1
  van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = 'June23_vdL_12patch_ScenarioPlot_10PercentDispersal_StartingLow')
  ICs = van_de_leemput.X2
  van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = 'June23_vdL_12patch_ScenarioPlot_10PercentDispersal_StartingHigh', show_legend = True)
  
  ICs = rass_briggs.X2
  rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = 'June23_fixedRB_12patch_ScenarioPlot_10PercentDispersal_StartingHigh')
  ICs = rass_briggs.X1
  rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = 'June23_fixedRB_12patch_ScenarioPlot_10PercentDispersal_StartingLow')
  ICs = blackwood_mumby.X1
  blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = 'June23_BM_12patch_ScenarioPlot_10PercentDispersal_StartingLow')
  ICs = blackwood_mumby.X2
  blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = 'June23_BM_12patch_ScenarioPlot_10PercentDispersal_StartingHigh')
   # 95 percent do not disperse, 30 patches, runnin three times longer than scenario plots 
  quit()
 
  blackwood_mumby = Model('BM', 12, 0.95, mgmt_strat = 'periodic')
  van_de_leemput = Model('vdL', 12, 0.95, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 12, 0.95, mgmt_strat = 'periodic')
  
  
 
  blackwood_mumby.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0_RB, C0L, 0.6, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()

  
  van_de_leemput.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0)
  blackwood_mumby.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0)
  rass_briggs.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0)
  
   # JUMP 
  
  ICs = van_de_leemput.X1
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = 'June23_vdL_12patch_ScenarioPlot_FivePercentDispersal_StartingLow')
  ICs = van_de_leemput.X2
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = 'June23_vdL_12patch_ScenarioPlot_FivePercentDispersal_StartingHigh', show_legend = True)
 
  ICs = blackwood_mumby.X2
  # blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = 'June23_BM_12patch_ScenarioPlot_FivePercentDispersal_StartingHigh')
  ICs = rass_briggs.X1
  rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = 'June23_fixedRB_12patch_ScenarioPlot_FivePercentDispersal_StartingLow')
  ICs = rass_briggs.X2
  rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = 'June23_fixedRB_12patch_ScenarioPlot_FivePercentDispersal_StartingHigh')
  ICs = blackwood_mumby.X1
  # blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = 'June23_BM_12patch_ScenarioPlot_FivePercentDispersal_StartingLow')
  
  
  # 95 percent dispersal with 20% poaching
  blackwood_mumby = Model('BM', 12, 0.95, mgmt_strat = 'periodic') 
  van_de_leemput = Model('vdL', 12, 0.95, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 12, 0.95, mgmt_strat = 'periodic')
  
 
  blackwood_mumby.initialize_patch_model(P0, C0L, C0H, M0L, M0H) # adding 0.2 for bug ? april 18
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0_RB, C0L, 0.6, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  
  van_de_leemput.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0.2)
  blackwood_mumby.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0.2)
  rass_briggs.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0.2)
  
  
  ICs = van_de_leemput.X1
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = 'June23_vdL_12patch_PoachingScenarioPlot_FivePercentDispersal_StartingLow')
  ICs = van_de_leemput.X2
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = 'June23_vdL_12patch_PoachingScenarioPlot_FivePercentDispersal_StartingHigh', show_legend = True)
 
  ICs = blackwood_mumby.X2
  # blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = 'June23_BM_12patch_PoachingScenarioPlot_FivePercentDispersal_StartingHigh')
  ICs = rass_briggs.X1
  rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = 'June23_fixedRB_12patch_PoachingScenarioPlot_FivePercentDispersal_StartingLow')
  ICs = rass_briggs.X2
  rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = 'June23_fixedRB_12patch_PoachingScenarioPlot_FivePercentDispersal_StartingHigh')
  ICs = blackwood_mumby.X1
  # blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = 'June23_BM_12patch_PoachingScenarioPlot_FivePercentDispersal_StartingLow')
  
  van_de_leemput.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0)
  blackwood_mumby.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0)
  rass_briggs.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0)
  
  
  
 
  quit()
  # 75 percent do not disperse
  blackwood_mumby = Model('BM', 12, 0.75, mgmt_strat = 'periodic') 
  van_de_leemput = Model('vdL', 12, 0.75, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 12, 0.75, mgmt_strat = 'periodic')
  
 
  blackwood_mumby.initialize_patch_model(P0 + 0.2, C0L, C0H+ 0.2, M0L, M0H) # adding 0.2 for bug ? april 18
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0_RB, C0L, 0.6, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  
  ICs = van_de_leemput.X1
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_Apr25_vdL_12patch_ScenarioPlot_25Dispersal_StartingLow')
  ICs = van_de_leemput.X2
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_Apr25_vdL_12patch_ScenarioPlot_25Dispersal_StartingHigh', show_legend = True)
  
  ICs = rass_briggs.X2
  #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_fixedRB_12patch_ScenarioPlot_25Dispersal_StartingHigh')
  ICs = rass_briggs.X1
  #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_fixedRB_12patch_ScenarioPlot_25Dispersal_StartingLow')
  ICs = blackwood_mumby.X1
  # blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_Apr25_BM_12patch_ScenarioPlot_25Dispersal_StartingLow')
  ICs = blackwood_mumby.X2
  # blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_Apr25_BM_12patch_ScenarioPlot_25Dispersal_StartingHigh')
  
   
  # 98 percent do not disperse
  blackwood_mumby = Model('BM', 12, 0.9, mgmt_strat = 'periodic') 
  van_de_leemput = Model('vdL', 12, 0.9, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 12, 0.9, mgmt_strat = 'periodic')
  
  blackwood_mumby.initialize_patch_model(P0 + 0.2, C0L, C0H + 0.2, M0L, M0H)
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0_RB, C0L, 0.6, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  
  ICs = van_de_leemput.X1
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_Apr25_vdL_12patch_ScenarioPlot_98Dispersal_StartingLow')
  ICs = van_de_leemput.X2
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_Apr25_vdL_12patch_ScenarioPlot_98Dispersal_StartingHigh', show_legend = True)
  
  ICs = rass_briggs.X2
  #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_fixedRB_12patch_ScenarioPlot_TenPercentDispersal_StartingHigh')
  ICs = rass_briggs.X1
  #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_fixedRB_12patch_ScenarioPlot_TenPercentDispersal_StartingLow')
  ICs = blackwood_mumby.X1
  # blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_Apr25_BM_12patch_ScenarioPlot_98Dispersal_StartingLow')
  ICs = blackwood_mumby.X2
  # blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_Apr25_BM_12patch_ScenarioPlot_98Dispersal_StartingHigh')
  

  
  # JUMP

  # create Model objects
  
  
  blackwood_mumby = Model('BM', 10, 1, mgmt_strat = 'periodic') 
  van_de_leemput = Model('vdL', 10, 1, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 10, 1, mgmt_strat = 'periodic')
  
  
  
  blackwood_mumby.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0, C0L, C0H, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  

  
  ICs = rass_briggs.X2
  #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_fixedRB_ScenarioPlot_ZeroDispersal_StartingHigh')
  ICs = rass_briggs.X1
  #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_fixedRB_ScenarioPlot_ZeroDispersal_StartingLow')
  ICs = blackwood_mumby.X1
  blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_BM_ScenarioPlot_ZeroDispersal_StartingLow')
  ICs = blackwood_mumby.X2
  blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_BM_ScenarioPlot_ZeroDispersal_StartingHigh')
  ICs = van_de_leemput.X1
  van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_vdL_ScenarioPlot_ZeroDispersal_StartingLow')
  ICs = van_de_leemput.X2
  van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_vdL_ScenarioPlot_ZeroDispersal_StartingHigh')
  
  
  # 95 percent do not disperse
  blackwood_mumby = Model('BM', 10, 0.93, mgmt_strat = 'periodic') 
  van_de_leemput = Model('vdL', 10, 0.96, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 10, 0.99, mgmt_strat = 'periodic')
  
  blackwood_mumby.initialize_patch_model(P0 + 0.2, C0L, C0H + 0.2, M0L, M0H)
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0, C0L, C0H + 0.2, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  
  
   
  # blackwood_mumby.coral_recovery_map(RB_t, BM_fishing_midpoint, blackwood_mumby.X1, filename = '12patchJune23_BM_HEATMAP_LO5')
  # blackwood_mumby.coral_recovery_map(RB_t, BM_fishing_midpoint, blackwood_mumby.X2, filename = '12patchJune23_BM_HEATMAP_HI5')
  #  van_de_leemput.coral_recovery_map(RB_t, vdl_fishing_midpoint, van_de_leemput.X1, filename = '12patchJune23_VDL_HEATMAP_LO5')
  # rass_briggs.coral_recovery_map(RB_t, RB_fishing_midpoint, rass_briggs.X1, filename = '12patchJune23_RB_HEATMAP_LO5')
  # van_de_leemput.coral_recovery_map(RB_t, vdl_fishing_midpoint, van_de_leemput.X2, filename = '12patchJune23_VDL_HEATMAP_HI5')
  # rass_briggs.coral_recovery_map(RB_t, RB_fishing_midpoint, rass_briggs.X2, filename = '12patchJune23_RB_HEATMAP_HI5')
  # blackwood_mumby.coral_recovery_map(RB_t, BM_fishing_midpoint, blackwood_mumby.X1, filename = '12patchJune23_BM_HEATMAP_LO5')
  # blackwood_mumby.coral_recovery_map(RB_t, BM_fishing_midpoint, blackwood_mumby.X2, filename = '12patchJune23_BM_HEATMAP_HI5')
  
  


  # coral recovery times for each model

  vdL_crt = van_de_leemput.get_coral_recovery_time(t)

  RB_crt = rass_briggs.get_coral_recovery_time(t)

  BM_crt = blackwood_mumby.get_coral_recovery_time(t)



  # make some time series -- X1 is starting low, X2 is starting high 
  # blackwood_mumby.set_mgmt_params(BM_crt, BM_fishing_midpoint, 2, 0) # set closure duration, fishing level, number of closures, and poaching, in that order 
 #  blackwood_mumby.time_series(blackwood_mumby.X2, t, save = False, show = True) 
  
  # rass_briggs.set_mgmt_params(1*RB_crt, RB_fishing_midpoint, 0, 0)
  # rass_briggs.time_series(rass_briggs.X2, t, save = False, show = True) 
  
  # van_de_leemput.set_mgmt_params(0.1*vdL_crt, vdl_fishing_midpoint, 2, 0)
  # van_de_leemput.time_series(van_de_leemput.X2, t, save = False, show = True) 

  
  
  
  # scenario plots 
  
  ICs = blackwood_mumby.X1
  # blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_NewBM_ScenarioPlot_1PercentDispersal_StartingLow')
  ICs = blackwood_mumby.X2
  blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_NewBM_ScenarioPlot_7PercentDispersal_StartingHigh')
  # ICs = van_de_leemput.X1
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_NewvdL_ScenarioPlot_1PercentDispersal_StartingLow')
  # ICs = van_de_leemput.X2
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_NewvdL_ScenarioPlot_1PercentDispersal_StartingHigh')
  ICs = rass_briggs.X1
  #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_fixedRB_ScenarioPlot_1PercentDispersal_StartingLow')
  ICs = rass_briggs.X2
  #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_fixedRB_ScenarioPlot_FivePercentDispersal_StartingHigh')
  
  
  
  
  # scenario plots with poaching 
  van_de_leemput.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0.2)
  blackwood_mumby.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0.2)
  rass_briggs.set_mgmt_params(closure_length = 35, f = 0, m = 1, poaching =  0.2)
  
  ICs = blackwood_mumby.X1
  blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_5BM_ScenarioPlot_0.1Dispersal_StartingLow_Poaching')
  ICs = blackwood_mumby.X2
  blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_5BM_ScenarioPlot_0.1Dispersal_StartingHigh_Poaching')
  ICs = van_de_leemput.X1
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_5vdL_ScenarioPlot_0.1Dispersal_StartingLow_Poaching')
  ICs = van_de_leemput.X2
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_5vdL_ScenarioPlot_0.1Dispersal_StartingHigh_Poaching')
  ICs = rass_briggs.X1
  #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_fixedRB_ScenarioPlot_0.1Dispersal_StartingLow_Poaching')
  ICs = rass_briggs.X2
  #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_fixedRB_ScenarioPlot_0.1Dispersal_StartingHigh_Poaching')
  
  
  # scenario plots for dispersal -- need to initialize new objects due to code structure 
  blackwood_mumby = Model('BM', 30, 0.8, mgmt_strat = 'periodic') 
  van_de_leemput = Model('vdL', 30, 0.8, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 30, 0.8, mgmt_strat = 'periodic')
  
  blackwood_mumby.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0, C0L, C0H, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  
  
  
  # scenario plots for 20 percent of fish moving 
  
  ICs = blackwood_mumby.X1
  # blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_5BM_ScenarioPlot_0.2Dispersal_StartingLow')
  ICs = blackwood_mumby.X2
  blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_5BM_ScenarioPlot_0.2Dispersal_StartingHigh')
  # ICs = van_de_leemput.X1
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_4vdL_ScenarioPlot_0.2Dispersal_StartingLow')
  # ICs = van_de_leemput.X2
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_5vdL_ScenarioPlot_0.2Dispersal_StartingHigh')
  
  ICs = rass_briggs.X1
  # #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_4RB_ScenarioPlot_0.2Dispersal_StartingLow')
  
  ICs = rass_briggs.X2
 #  #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_5RB_ScenarioPlot_0.2Dispersal_StartingHigh')
  
  
  # for 50 percent of fish moving 
  
  blackwood_mumby = Model('BM', 30, 0.5, mgmt_strat = 'periodic') 
  van_de_leemput = Model('vdL', 30, 0.5, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 30, 0.5, mgmt_strat = 'periodic')
  
  blackwood_mumby.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0, C0L, C0H, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  
  
  ICs = blackwood_mumby.X1
  # blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_4BM_ScenarioPlot_FiftyPercentDispersal_StartingLow')
  ICs = blackwood_mumby.X2
  blackwood_mumby.scenario_plot(t, BM_fishing_midpoint, ICs, filename = '12patchJune23_5BM_ScenarioPlot_FiftyPercentDispersal_StartingHigh')
  ICs = van_de_leemput.X1
  # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_4vdL_ScenarioPlot_FiftyPercentDispersal_StartingLow')
  ICs = van_de_leemput.X2
 # van_de_leemput.scenario_plot(t, vdl_fishing_midpoint, ICs, filename = '12patchJune23_5vdL_ScenarioPlot_FiftyPercentDispersal_StartingHigh')
  ICs = rass_briggs.X1
  # #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_4RB_ScenarioPlot_FiftyPercentDispersal_StartingLow')
  ICs = rass_briggs.X2
  # #rass_briggs.scenario_plot(RB_t, RB_fishing_midpoint, ICs, filename = '12patchJune23_5RB_ScenarioPlot_FiftyPercentDispersal_StartingHigh')
  

  # 95 percent do not disperse
  blackwood_mumby = Model('BM', 30, 0.95, mgmt_strat = 'periodic') 
  van_de_leemput = Model('vdL', 30, 0.95, mgmt_strat = 'periodic')
  rass_briggs = Model('RB', 30, 0.95, mgmt_strat = 'periodic')
  
  blackwood_mumby.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  van_de_leemput.initialize_patch_model(P0, C0L, C0H, M0L, M0H)
  rass_briggs.initialize_patch_model(P0, C0L, C0H, M0vL, M0vH, M0iL, M0iH)
  
  van_de_leemput.load_parameters()
  rass_briggs.load_parameters()
  blackwood_mumby.load_parameters()
  
 
  quit()
  


if __name__ == '__main__':
	main()





