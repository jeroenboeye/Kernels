# -*- coding: utf-8 -*-
"""
Created on Thu Mar 07 15:28:49 2013

@author: Jboeye
"""
import numpy as np
import math
import copy as cp
import random
import sys
#from scipy.stats import lognorm

class Individual:
    '''Individual class  =  lowest level object'''
    def __init__(self, 
                 dispersal_kernel, 
                 patch_coord, 
                 dynamic, 
                 mutation_rate, 
                 nr_kernel_steps):
        self.coord = patch_coord
        self.dynamics = dynamic
        self.mutation_rate = mutation_rate
        self.nr_kernel_steps = nr_kernel_steps 
        self.dispersal_kernel = dispersal_kernel
        
    def set_density(self, density):
        '''Sets the density a disperser experiences to a certain value'''
        self.experienced_disperser_density  =  density
    
    def reproduce(self, avr_n_offspring, time_ratio):    
        '''Returns a number of offspring depending on the average nr given'''     
        offspring = []
        #mutation takes place after dispersal but before reproduction
        #this means individuals disperse by a kernel given by their parent
        self.dispersal_kernel = self.mutating(self.dispersal_kernel, time_ratio)
        nr_of_offspring = np.random.poisson(avr_n_offspring)
        for n in xrange(nr_of_offspring):
            offspring.append(Individual(self.dispersal_kernel[:], 
                                        self.coord[:], 
                                        self.dynamics, 
                                        self.mutation_rate, 
                                        self.nr_kernel_steps))   
        return offspring
        
    def dispersal_survival(self, step_mortality, distance):
        '''Returns a survival boolean after randomly determining whether an 
        individual survives given a certain chance'''
        survival = True
        mortality = 1-math.exp(-step_mortality*distance)
        if random.random() < mortality:
            survival = False
        return survival
        
    def mutating(self, old_kernel, time_ratio):
        '''All kernel steps have a chance to mutate here'''
        self.time_ratio = time_ratio
        new_kernel = old_kernel[:]
        for n in xrange(self.nr_kernel_steps):
            if random.random() < self.mutation_rate: # mutation or not     
                mut_sd   =  2 * math.exp(-5*self.time_ratio)
                mutation = np.random.normal(0, 0.1+mut_sd) 
                new_kernel[n]  +=  mutation              
        return new_kernel
        
    def relocate(self, two_pi, max_x):  
        '''Relocation of individual according to kernel'''
        total_kernel_prob = 0
        #sum all the positive kernel values (negative  =  0)
        for value in self.dispersal_kernel: 
            if value > 0:
                total_kernel_prob += value     
        #if not all kernel values are negative we calculate a distance
        if total_kernel_prob > 0:        
            prop_kernel = self.dispersal_kernel[:]
            for dist in xrange(self.nr_kernel_steps):
                #change negative values into zeros    
                if prop_kernel[dist]  <  0: 
                    prop_kernel[dist] = 0        
            for dist in xrange(self.nr_kernel_steps-1):
                #make the kernel cummulative  
                prop_kernel[dist+1] = prop_kernel[dist]+prop_kernel[dist+1]           
            samplenr = random.uniform(0, total_kernel_prob)
            #sample distance using random nr 
            if prop_kernel[0] > samplenr:
                distance = 0
            else:
                for dist in xrange(self.nr_kernel_steps-1):
                    if prop_kernel[dist] <=  samplenr < prop_kernel[dist+1]:
                        distance = dist+1
                        break   
        else: 
            distance = 0
        self.distance = distance    
        if distance  >  0:
            angle = random.uniform(0, two_pi)  
            self.x = int(round(distance*math.cos(angle)))
            self.y = int(round(distance*math.sin(angle)))
            self.coord[0] = (self.x+self.coord[0])%max_x
            self.coord[1] = (self.y+self.coord[1])%max_x
        
    def set_new_coords(self, x, y):
        '''Set the new coordintates'''
        self.coord = [x, y]
        
    def get_dispersal_kernel(self):
        '''Returns the dispersal kernel'''
        return np.array(self.dispersal_kernel)
    
class Patch:
    '''Patch class, contains local population of individuals'''
    def __init__(self, 
                 patch_coord, 
                 max_x, 
                 initial_pop, 
                 carrying_capacity, 
                 dynamic, 
                 growthrate, 
                 a, 
                 mutation_rate, 
                 nr_kernel_steps, 
                 two_pi):
        self.initial_pop = initial_pop
        self.carrying_capacity = carrying_capacity
        self.dynamics = dynamic
        self.growthrate = growthrate
        self.two_pi = two_pi
        self.max_x = max_x
        #there are several patches (each with their own "kernels)
        self.patch_coord = patch_coord  
        self.population = []
        self.disperser_pop = []
        self.a = a
        self.mutation_rate = mutation_rate
        self.nr_kernel_steps = nr_kernel_steps
        self.rounding_neg_to_zero = np.vectorize(lambda x:0 if x  <  0 else x)
        self.scaling = np.vectorize(lambda x, y: 100*x/float(y))
        self.matrix_size = (self.nr_kernel_steps*2)-1
        Patch.initialize_individuals(self)
            
    def initialize_individuals(self):
        '''Initialization of individuals'''
        for n in xrange(self.initial_pop):
            dispersal_kernel = []
            for d in xrange(self.nr_kernel_steps):
                dispersal_kernel.append(random.random())
            self.population.append(Individual(dispersal_kernel[:], 
                                              self.patch_coord[:], 
                                              self.dynamics, 
                                              self.mutation_rate, 
                                              self.nr_kernel_steps))
        
    def resident_reproduction(self, time_ratio, beta): 
        '''Reproduction of residents'''
        local_density = Patch.get_local_density(self)+self.dynamics.sample_random_immigrants()
        survival = 1/float((1+self.a*local_density)**beta)
        avr_n_offspring = survival*self.growthrate
        old_population = self.population[:]
        del self.population[:]
        for i in old_population:
            self.population.extend(i.reproduce(avr_n_offspring, time_ratio))
        del old_population[:]
        
    def disperser_reproduction(self, time_ratio, beta):
        '''Reproduction of dispersers'''
        old_population = self.disperser_pop[:]
        del self.disperser_pop[:]
        for disp in old_population: 
            local_density = disp.experienced_disperser_density
            survival = 1/float((1+self.a*local_density)**beta)
            avr_n_offspring = survival*self.growthrate #sigma = 0   
            self.disperser_pop.extend(disp.reproduce(avr_n_offspring, time_ratio))
        del old_population[:]
        
    def add_disperser(self, disperser):
        '''A given disperser is added to the disperser population'''
        self.disperser_pop.append(disperser)

    def add_resident(self, resident):
        '''A resident is added to the resident population'''
        self.population.append(resident)
        
    def get_pop(self):
        '''The resident population is returned and deleted from memory'''
        pop = self.population[:]
        del self.population[:]
        return pop       
        
    def join_pops(self):
        '''The disperser and resident population are merged'''
        self.population.extend(self.disperser_pop[:])
        del self.disperser_pop[:]
        
    def disperse(self):
        '''The individuals disperse (see def relocate)'''
        dispersers = [] 
        #print len(self.population), 'all'
        #print self.patch_coord
        old_population = self.population[:]
        del self.population[:]
        for i in old_population:
            i.relocate(self.two_pi, self.max_x)
            if i.distance == 0:
                self.population.append(i)
            else:
                dispersers.append(i)
        #print len(self.population), 'residents'
        #print len(dispersers), 'dispersers'
        #for d in dispersers: print d.coord
        return dispersers

    def append_individual(self, individual):
        '''A given individual is added to the population'''
        self.population.append(individual)
        
    def get_local_density(self):
        '''The local density is returned'''
        return len(self.population)
        
    def get_sum_dispersal_kernel(self):
        '''An array summing the kernels of the pop is returned'''
        sum_dispersal_kernels = np.zeros(self.nr_kernel_steps)
        for i in self.population:
            kernel = i.get_dispersal_kernel()
            kernel = self.rounding_neg_to_zero(kernel)    
            total = sum(kernel)
            if total > 0:
                kernel = self.scaling(kernel, total)
                sum_dispersal_kernels  += kernel
        return sum_dispersal_kernels     
        
    def decrease_density(self, forced_n):
        '''The population is shuffled and then reduced to the forced density'''
        if self.get_local_density()  >  forced_n:
            random.shuffle(self.population)
            self.population = self.population[:int(forced_n)]        
        
    def increase_density(self, random_individual_list):
        '''A random individual is added to the population (from the list)'''
        relocated_list=[]        
        for individual in random_individual_list:
            individual.set_new_coords(self.patch_coord[0],self.patch_coord[1])
            relocated_list.append(individual)
        self.population.extend(relocated_list)
            
    def get_coords(self):
        '''Returns the patch coordinates'''
        return self.patch_coord
        
    def get_random_individual_patch(self, ind_nr):
        '''Returns an individual with a given index from the pop'''
        individual = self.population[ind_nr]
        return cp.copy(individual)
     
class Metapop:
    '''Metapop class, contains all patches'''
    def __init__(self, 
                 dynamic, 
                 nr_kernel_steps, 
                 initial_pop, 
                 growthrate, 
                 carrying_capacity, 
                 mutation_rate, 
                 dispersal_mort, 
                 max_x, 
                 shuffling, beta):
        self.initial_pop = initial_pop
        self.nr_kernel_steps = nr_kernel_steps
        self.two_pi = 2*math.pi
        self.shuffling = shuffling
        self.beta = beta
        self.growthrate = growthrate
        self.carrying_capacity = carrying_capacity
        self.max_x = max_x
        self.mutation_rate = mutation_rate
        self.dispersal_mort = dispersal_mort
        self.dynamic = dynamic        
        #susceptibility to crowding
        self.a = (self.growthrate**(1/float(self.beta))-1)/float(self.carrying_capacity) 
        self.patches = []
        self.mean_n_immigrant_ref = self.dynamic.get_immigrant_mean()
        self.matrix_size = (self.nr_kernel_steps*2)-1
        Metapop.initialize_patches(self)
        
    def initialize_patches(self):
        '''Initialize the patches'''
        for y in xrange(self.max_x):
            for x in xrange(self.max_x):            
                self.patches.append(Patch([x, y], 
                                          self.max_x, 
                                          self.initial_pop, 
                                          self.carrying_capacity, 
                                          self.dynamic, 
                                          self.growthrate, 
                                          self.a, 
                                          self.mutation_rate, 
                                          self.nr_kernel_steps, 
                                          self.two_pi))
            
    def live(self, forced_n, time_ratio):
        '''Main procedure, first dispersers are removed from patches
        then they reproduce, then residents reproduce, then pops are merged, 
        then forced density is enforced. There is optional shuffling to 
        destroy kin structure'''
        self.time_ratio = time_ratio        
        #Virtual densities are selected from reference model 
        #and dispersers are removed from focal patch'''
        self.disperser_density = np.zeros((self.matrix_size, 
                                           self.matrix_size), 
                                           dtype = int) 
        dispersers = []
        for patch in self.patches: 
            dispersers.extend(patch.disperse()) #dispersers move out of patch 
        #Dispersers are checked for survival + density is increased'''
        surviving_dispersers = []
        for d in dispersers:  
            #check dispersal survival
            if d.dispersal_survival(self.dispersal_mort, d.distance): 
                if self.shuffling:   #pre reproduction shuffle
                    new_x = np.random.randint(0, self.max_x)
                    new_y = np.random.randint(0, self.max_x)
                    #if shuffled the individual will be compete with 
                    #individuals from any random patch.
                    d.set_new_coords(new_x, new_y)           
                if self.disperser_density[d.coord[0]][d.coord[1]] == 0:
                    self.disperser_density[d.coord[0]][d.coord[1]] = self.dynamic.sample_random_density()
                self.disperser_density[d.coord[0]][d.coord[1]]  += 1       
                surviving_dispersers.append(d)
        del dispersers[:]
        #print self.disperser_density
        #Dispersers get their experienced density assigned and
        #are placed in patch'''
        for d in surviving_dispersers:
            d.set_density(self.disperser_density[d.coord[0]][d.coord[1]])
            self.patches[self.max_x*d.coord[1]+d.coord[0]].add_disperser(d)
        del surviving_dispersers[:]
        
        if self.shuffling:    #pre reproduction shuffle
            self.shuffle_pop()        
            
        #dispersers and residents reproduce'''
        for p in self.patches:
            p.disperser_reproduction(self.time_ratio, self.beta)
            p.resident_reproduction(self.time_ratio, self.beta)
     
         #print self.patch.get_local_density(), len(self.new_dispersers)   

        #dispersers and residents are joined
        
        total_popsize = 0                
        all_above_forced_n = True
        for p in self.patches:
            p.join_pops()
            loc_dens = p.get_local_density()
            total_popsize += loc_dens
            if loc_dens < forced_n:
                all_above_forced_n = False
            
        if self.shuffling:        #post reproduction shuffle
            self.shuffle_pop()
        #the population is forced to a certain density   
        self.force_density(total_popsize, all_above_forced_n, forced_n)
        
    def force_density(self, total_popsize, all_above_forced_n, forced_n):
        '''Here the wanted density is enforced'''
        if total_popsize > 0:   
            if not all_above_forced_n:
                poplist = []
                total = 0     
                #create a list of all individuals and 
                #the patch in which they are located.
                for patchnr, patch in enumerate(self.patches): 
                    dens =  patch.get_local_density()
                    total  += dens
                    poplist.extend([[patchnr, ind, dens] for ind in xrange(dens)])
                                  
                for patchnr, patch in enumerate(self.patches):                                      
                    dens =  patch.get_local_density()
                    random_individual_list = [] 
                    if dens   <  forced_n:                     
                        for i in xrange(int(forced_n)-dens):
                            individual = random.choice(poplist)
                            random_individual_list.append(self.patches[individual[0]].get_random_individual_patch(individual[1])) #first argument is patchnr, second ind nr
                        patch.increase_density(random_individual_list) 
            for patch in self.patches:
                patch.decrease_density(forced_n)               
                
    def shuffle_pop(self):      
        '''The popualtion is shuffled to destroy kin structure'''
        #Put all individuals in list + shuffle        
        pop = []
        for p in self.patches:
            pop.extend(p.get_pop())             
        #Randomly redistribute individuals over patches'''
        for ind in pop:
            new_x = np.random.randint(0, self.max_x)
            new_y = np.random.randint(0, self.max_x)
            ind.set_new_coords(new_x, new_y)
            self.patches[self.max_x*ind.coord[1]+ind.coord[0]].add_resident(ind)
        del pop[:]
            
    def get_average_kernel(self):
        '''Calculates and returns the average kernel'''
        sum_kernels = np.zeros(self.nr_kernel_steps)
        density = 0
        for p in self.patches:
            sum_kernels += p.get_sum_dispersal_kernel()
            density += p.get_local_density()
        if density  >  0:
            average_dispersal_kernel = sum_kernels/float(density)
        else:average_dispersal_kernel = 'NA'
        return average_dispersal_kernel
        
    def reset(self):
        '''Deletes all patches'''
        del self.patches[:]
        
class Dynamics:
    '''Class that regulates information from reference model'''
    def __init__(self, title_d, title_i):        
        self.density_list = [int(line.strip()) for line in open(title_d)]
        self.immigrants_list = [int(line.strip()) for line in open(title_i)] 
        self.immigrant_mean = np.mean(self.immigrants_list)
        
    def get_immigrant_mean(self):
        '''Returns the average number of immigrants (calculated from list)'''
        return self.immigrant_mean
        
    def sample_random_density(self):
        '''Returns a random density from the list'''
        return random.choice(self.density_list)
        
    def sample_random_immigrants(self):
        '''Returns a random n_immigrants from the list'''
        return random.choice(self.immigrants_list)
        
class Simulation:
    '''Highest level class, contains main loop and analysis'''
    def __init__(self, 
                 maxtime, 
                 initial_pop, 
                 growthrate, 
                 carrying_capacity, 
                 max_x, 
                 nr_kernel_steps, 
                 mutation_rate, 
                 dispersal_mort, 
                 shuffling, 
                 beta, 
                 n_fraction_list):
        self.maxtime = maxtime
        self.initial_pop = initial_pop
        self.growthrate = growthrate
        self.beta = beta
        self.n_fraction_list = n_fraction_list
        self.carrying_capacity = carrying_capacity
        self.max_x = max_x
        self.nr_kernel_steps = nr_kernel_steps
        self.mutation_rate = mutation_rate
        self.dispersal_mort = dispersal_mort
        self.shuffling = shuffling
        title_d  =  'Opt_no_kin_D_l_%s_disp_m_%s_K_%s_beta_%s.out' 
        title_d  =  title_d % (str(self.growthrate), 
                             str(self.dispersal_mort), 
                             str(self.carrying_capacity), 
                             str(self.beta))
        title_i  =  'Opt_no_kin_I_l_%s_disp_m_%s_K_%s_beta_%s.out'
        title_i  =  title_i % (str(self.growthrate), 
                             str(self.dispersal_mort), 
                             str(self.carrying_capacity), 
                             str(self.beta))
        self.dynamic  =  Dynamics(title_d, title_i)
        if shuffling:
            title  =  'No_kin_swng_2D_%s_disp_m_%s_K_%s_mut_%s_beta_%s.out'
            title  =  title % (str(self.growthrate), 
                             str(self.dispersal_mort), 
                             str(self.carrying_capacity), 
                             str(self.mutation_rate), 
                             str(self.beta))
        else:
            title  =  'Kin_swng_2D_%s_disp_m_%s_K_%s_mut_%s_beta_%s.out'
            title  =  title % (str(self.growthrate), 
                             str(self.dispersal_mort), 
                             str(self.carrying_capacity), 
                             str(self.mutation_rate), 
                             str(self.beta))
        self.output = open(title, "a")        
        Simulation.run(self)
        
    def run(self):
        '''Main loop'''
        fraction_list_all=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21], [22, 23, 24, 25, 26], [27, 28, 29, 30], [31, 32, 33, 34], [35, 36, 37], [38, 39, 40], [41, 42, 43], [44, 45], [46, 47], [48, 49], [50]]
        for f in fraction_list_all[self.n_fraction_list]:
            self.metapop = Metapop(self.dynamic, 
                                   self.nr_kernel_steps, 
                                   self.initial_pop, 
                                   self.growthrate, 
                                   self.carrying_capacity, 
                                   self.mutation_rate, 
                                   self.dispersal_mort, 
                                   self.max_x, 
                                   self.shuffling, 
                                   self.beta)
            self.forced_n = f*self.carrying_capacity/float(10)
            for t in xrange(self.maxtime):
                if t % 10 == 0:
                    print self.forced_n, t                    
                self.time_ratio = t/float(self.maxtime)                
                self.metapop.live(self.forced_n, self.time_ratio)
            Simulation.analyse(self, self.forced_n)
            self.metapop.reset()
        self.output.close()
        
    def analyse(self, forced_density):
        '''Analysis results and writes them to output'''
        density = str(int(forced_density))
        average_dispersal_kernel = self.metapop.get_average_kernel()
        if average_dispersal_kernel != 'NA':
            self.output.write(density) 
            for k in average_dispersal_kernel:                
                self.output.write("\t"+str(round(k, 2)))  
            self.output.write("\n")
        #Write 'NA' 10 times 
        else: self.output.write(density+("\t"+average_dispersal_kernel)*10 +"\n") 
                            
if __name__  ==   '__main__':                
    SIMULATION = Simulation(maxtime = 1000, 
                            initial_pop = 50, #in each patch 
                            growthrate = int(round(float(sys.argv[1]))), 
                            carrying_capacity = 50, 
                            max_x = 19, 
                            nr_kernel_steps = 10, 
                            mutation_rate = 0.01, 
                            #per step, see function
                            dispersal_mort = float(sys.argv[2]), 
                            #if true, all kin structure is destroyed
                            shuffling = bool(round(float(sys.argv[4]))), 		
                            beta = int(round(float(sys.argv[3]))), 
                            n_fraction_list = int(round(float(sys.argv[5]))))                            
    #import cProfile
    #cProfile.run('SIMULATION.run()')                          
    #print 'Finished!'                          