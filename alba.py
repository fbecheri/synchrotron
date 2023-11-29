import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
#import tango

C = 299792458  # m/s
E_electron = 0.510998

class SynchrotronSimulation:
    def __init__(self):
        #self.dp = tango.DeviceProxy("fbecheri/synchsim/1")
        
        self.initAlba()
        # Observer position in units of radius
        self.observer_x = 10 # *r
        self.observer_y = 1
        self.observer_z = 0
        
    def initAlba(self):
        Alba_R, Alba_v = self.getAlba(E=3, l=269)
        self.setVelocityAndRadius(v=Alba_v, r=Alba_R)
        
    def getAlba(self, E, l):
        Alba_R = self.getRadiusFromCircunference(l)  # synchrotron radius in meters
        
        gamma = self.getGammaFromGeV(E)
        Alba_v = C * self.getBetafromGamma(gamma)
        return Alba_R, Alba_v
    
    def setVelocityAndRadius(self, v, r):
        
        self.setVelocity(v)
        self.setRadius(r)
        
        
    def setVelocity(self,v):
        self.v = v
        self.beta = v / C
        self.gamma = 1/np.sqrt(1-self.beta*self.beta)
        
    def setRadius(self, r):
        self.r = r
        #  circunference, radius, period and frequency
        self.sr_len, self.sr_period, self.sr_omega =  self.getRingParameters(r=self.r, v=self.v)
        
    def getBetafromGamma(self, gamma):
        # Beta
        return np.sqrt(1 - 1 / (gamma * gamma))
        
    def getGammaFromGeV(self, E):# GeV
        # Ring energy on electron energy
        return E / E_electron * 1000 # relativistic gamma associated to Alba
    
    def getRadiusFromCircunference(self, l):
        r = l / (2 * np.pi)  # synchrotron radius in meters
        return r
    
    def getRingParameters(self, r, v):
        sr_len = 2 * np.pi * r
        sr_period = v / sr_len  # period is seconds
        sr_omega = 2*np.pi / sr_period  # angular frequency in Hz
        
        return sr_len, sr_period, sr_omega
        
    def getChargeTrajectory(self, t_sent):
        # Charge position is parametrized with tau = gamma*t_sent
        q1 = self.r * np.sin(self.sr_omega * t_sent)
        q2 = self.r * np.cos(self.sr_omega * t_sent)
        q3 = 0
        return q1, q2, q3
        
    def getEcomponents(self, t_sent):
        q1, q2, q3 = self.getChargeTrajectory(t_sent)
        # Stationary observer at (2r, r, 0)
        # Null vector coordinates
        n1 = self.observer_x * self.r - q1
        n2 = self.observer_y * self.r - q2
        n3 = self.observer_z * self.r - q3
        # Received time by the observer
        t_received = t_sent + np.sqrt(n1 * n1 + n2 * n2 + n3 * n3) / C
        # Polar coordinates using vel and accel directions
        nv = n1 * np.cos(self.sr_omega * t_sent) - n2 * np.sin(self.sr_omega * t_sent)
        na = n1 * np.sin(self.sr_omega * t_sent) + n2 * np.cos(self.sr_omega * t_sent)
        # charge-observer distance at sent-time
        n0 = np.sqrt(n1 * n1 + n2 * n2 + n3 * n3)
        # acceleration in observer frame
        a = C * self.beta * self.beta * self.gamma * self.gamma / self.r
        # Initila factor of electromagnetic field
        F = 2 * a / (np.power(C * self.gamma * n0, 2) * np.power(1 - self.beta * nv / n0, 3))
        # Components of electric field
        Ev = F * (self.beta * na - na * nv / n0)
        Ea = F * ((nv * nv + n3 * n3) / n0 - self.beta * nv)
        E3 = F * (-na * n3 / n0)
        return Ev, Ea, E3, t_received
    
    def getEmodule(self, t_sent):
        Ev, Ea, E3, t_received = self.getEcomponents(t_sent)
        # Module of electric field and intensity
        E = np.sqrt(Ev * Ev + Ea * Ea + E3 * E3)
        return E, t_received
        
    def getIntensity(self, t_sent):
        E, t_received = self.getEmodule(t_sent)
        # Intensity
        I = E * E
        return I, t_received
    
    def configPlots(self, raws, col):
                    
        self.N_plotraws = raws
        self.N_subplots = col
        self.fig, self.axes = plt.subplots(raws, col, figsize=(12, 6))

    def simulate(self, raw, col, N=360, power_factor=1 ):
        dt_degree = self.sr_period / 360
            
        I_array = [] # Initialize spectrum of intensities
        t_array = [] # Initialize spectrum of received-time
        tau_array = [] # Initilaize spectrum of sent-time
        
        R1 = -N / power_factor # min_range for j-spectrum
        R2 = (N + 1) / power_factor # max_range for j-spectrum
        x_array = np.linspace(R1, R2, 2 * N + 1) # array for plotting
        
        for i in range(-N, N + 1):
            # Sequence of different sent-times
            t_sent = i * dt_degree / power_factor
            tau_array = tau_array + [t_sent]
            # Intensity and received time for each t_sent
            I, t_received = self.getIntensity(t_sent)
            I_array.append(I)
            t_array.append(t_received)
            
        print(self.sr_omega * t_sent)
            
        ## Write Tango
        #if col==0:
            #self.dp.intensity0_spectrum = I_array
        #elif col==1:
            #self.dp.intensity1_spectrum = I_array
        #elif col==2:
            #self.dp.intensity2_spectrum = I_array

        # (0) Plot the array of sent-time
        col = 0
        lbl = "Plot range: {} {} degree".format(R1, R2)
        self.axes[raw, col].plot(tau_array, I_array, label=lbl)
        #self.axes[raw, col].set_title(lbl)
        #self.axes[raw, col].set_xticks(np.linspace(R1, R2, 8))
        self.axes[raw, col].grid(True)
        self.axes[raw, col].legend()

         # (1) Plot the array of received-time
        col = 1
        lbl = "Plot range: {} {} degree".format(R1, R2)
        self.axes[raw, col].plot(t_array, I_array, label=lbl)
        #self.axes[raw, col].set_title(lbl)
        #self.axes[raw, col].set_xticks(np.linspace(R1, R2, 8))
        self.axes[raw, col].grid(True)
        self.axes[raw, col].legend()
        
        # Fast Fourier
        col = 2
        y = fft(I_array)
        t_sent_A = t_array[N] #-N * dt_degree / power_factor
        t_sent_B = t_array[N+1] #(N+1) * dt_degree / power_factor
        freq = np.fft.fftfreq(len(y), t_sent_B - t_sent_A)
        self.axes[raw, col].plot(freq, np.abs(y))
        

    def showPlot(self):

        # Show all the plots
        plt.show()

if __name__ == "__main__":
    simulator = SynchrotronSimulation()
    power_factor = np.power(10, 5)

    simulator.configPlots(3,3)

    # Raw, Col, N, power_factor
    # In the top-left, we show the radiation versus the sent0time,
    # ... along the full ring between -180 and 180
    # In the top center, the x-coordinates is the received time
    # On top-right, the Fast Fourier Transform
    simulator.simulate(0,0,180, 1)

    # 
    simulator.setVelocityAndRadius(C/2, 1)
    simulator.simulate(1,0,180, 1)

    # 
    simulator.setVelocityAndRadius(C/1000000, 1)
    simulator.simulate(2,0,180, 1)
    
    #Alba_R, Alba_v = simulator.getAlba(E=3, l=268)
    #simulator.setVelocityAndRadius(v=Alba_v/2.0, r=Alba_R)
    
    simulator.showPlot()
    
    
    
    #import numpy as np
    #import matplotlib.pyplot as plt
    #from scipy import signal
    #from scipy.fft import fft
    
    #t = np.linspace(0, 1, 1000, endpoint=False)
    #amplitude = (signal.square(2 * np.pi * 600 * t) + 1) / 2
    
    #sq_wave = plt.figure()
    #plt.plot(t, amplitude)
    #plt.show()
    
    #y = fft(amplitude)
    #freq = np.fft.fftfreq(len(y), t[1] - t[0])
    
    #plt.plot(freq, np.abs(y))
    #plt.show()
