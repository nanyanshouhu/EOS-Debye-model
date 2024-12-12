import sys
import math
import numpy as np
import subprocess
from scipy.optimize import curve_fit
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import pandas as pd
#import electron_free_energy
from scipy.integrate import quad

s = 0.617
NA = 6.02e23  # Avogadro's number
A = 231.04
with open('POSCAR', 'r') as f:
    lines = f.readlines()

# Get element and quantity information
elements = lines[5].split()
element_counts = [int(count) for count in lines[6].split()]

# A dictionary that contains the atomic masses of elements
MM_of_Elements = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107, 'N': 14.0067,
                  'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386,
                  'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078,
                  'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938045,
                  'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.409, 'Ga': 69.723, 'Ge': 72.64,
                  'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585,
                  'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.9063, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42,
                  'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.760, 'Te': 127.6,
                  'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116,
                  'Pr': 140.90465, 'Nd': 144.242, 'Pm': 146.9151, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25,
                  'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04,
                  'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
                  'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804,
                  'Po': 208.9824, 'At': 209.9871, 'Rn': 222.0176, 'Fr': 223.0197, 'Ra': 226.0254, 'Ac': 227.0278,
                  'Th': 232.03806, 'Pa': 231.03588, 'U': 238.02891, 'Np': 237.0482, 'Pu': 244.0642, 'Am': 243.0614,
                  'Cm': 247.0703, 'Bk': 247.0703, 'Cf': 251.0796, 'Es': 252.0829, 'Fm': 257.0951, 'Md': 258.0951,
                  'No': 259.1009, 'Lr': 262, 'Rf': 267, 'Db': 268, 'Sg': 271, 'Bh': 270, 'Hs': 269, 'Mt': 278,
                  'Ds': 281, 'Rg': 281, 'Cn': 285, 'Nh': 284, 'Fl': 289, 'Mc': 289, 'Lv': 292, 'Ts': 294, 'Og': 294,
                  'ZERO': 0}

# Calculate the total mass
total_mass = sum([MM_of_Elements[element] * count for element, count in zip(elements, element_counts)])

# Calculate the average mass per atom
average_mass_per_atom = total_mass / sum(element_counts)

print("average:", average_mass_per_atom)
N = sum(element_counts)  # Number of atoms
print(N)
M = average_mass_per_atom  # Specify the value of M1 per atom
h = 1.055e-34

# Define a function to calculate discrete point derivatives
def cal_deriv(x, y):                  # x and y are both lists
    diff_x = []                       # Used to store the difference between two numbers in the x list
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []                       # Used to store the difference between two numbers in the y list
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []                       # Used to store the slopes
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []                        # Used to store the first derivative
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))  # Calculate and store the result according to the definition of the discrete point derivative
    deriv.insert(0, slopes[0])        # The derivative at the (left) end is the slope with its nearest point
    deriv.append(slopes[-1])          # The derivative at the (right) end is the slope with its nearest point

    #for i in deriv:                   # Print the result for easy checking (can be commented out when calling)
    #    print(i)

    return deriv                      # Return the list storing the first derivative results

def cal_second_deriv(x, y):
    n = len(x)
    second_deriv = []

    for i in range(n):
        if i == 0:
            # Use forward difference for the first point
            h = x[i + 1] - x[i]
            d1 = (y[i + 1] - y[i]) / h
            d2 = (y[i + 2] - y[i + 1]) / h
            second_deriv.append(2 * (d2 - d1) / h)
        elif i == n - 1:
            # Use backward difference for the last point
            h = x[i] - x[i - 1]
            d1 = (y[i] - y[i - 1]) / h
            d2 = (y[i - 1] - y[i - 2]) / h
            second_deriv.append(2 * (d2 - d1) / h)
        else:
            # Use central difference for interior points
            h1 = x[i] - x[i - 1]
            h2 = x[i + 1] - x[i]
            d1 = (y[i] - y[i - 1]) / h1
            d2 = (y[i + 1] - y[i]) / h2
            second_deriv.append(2 * (d2 - d1) / (h1 + h2))

    return second_deriv
def read_data(filename):
    """ read the energy and Volume, the format shows as follow: 
    the first line are structure factors, the second line are volumes, and the third lines are energy:
        3.74 136.2000 -105.833996
        3.75 136.9300 -105.865334
        3.76 137.6600 -105.892136
        3.78 139.1300 -105.928546
        3.79 139.8600 -105.944722
        3.80 140.6000 -105.955402
        3.81 141.3400 -105.960574
        3.82 142.0900 -105.960563
        3.83 142.8300 -105.954437
        3.84 143.5800 -105.949877
    """
    data = np.loadtxt("curvefit.txt")     
    return data[:,1], data[:,2] 
def eos_murnaghan(vol, E0, B0, BP, V0):
    # First term in the equation
    term1 = (4 * B0 * V0) / (BP - 1)**2
    
    # Second term in the equation
    term2 = 1 - (3 / 2) * (BP - 1) * (1 - (vol / V0)**(1 / 3))
    
    # Exponential term
    exp_term = np.exp((3 / 2) * (BP - 1) * (1 - (vol / V0)**(1 / 3)))
    
    # Energy calculation
    E = E0 + term1 - term1 * term2 * exp_term
    
    return E
def fit_murnaghan(volume, energy):
    """ fittint Murnaghan equation，and return the optimized parameters 
    """
    # fitting with Quadratic first and then get the guess parameters.
    p_coefs = np.polyfit(volume, energy, 2)
    # the lowest point of parabola dE/dV = 0 ( p_coefs = [c,b,a] ) V(min) = -b/2a
    p_min = - p_coefs[1]/(2.*p_coefs[0])
    # warn if min volume not in result range 
    if (p_min < volume.min() or p_min > volume.max()):
        print ("Warning: minimum volume not in range of results")
    # estimate the energy based the the lowest point of parabola   
    E0 = np.polyval(p_coefs, p_min)
    # estimate the bulk modules
    B0 = 2.*p_coefs[2]*p_min
    # guess the parameter (set BP as 4)
    init_par = [E0, B0, 4, p_min]
    #print ("guess parameters:")
    #print (" V0     =  {:1.4f} A^3 ".format(init_par[3]))
    #print (" E0     =  {:1.4f} eV  ".format(init_par[0]))
    #print (" B(V0)  =  {:1.4f} eV/A^3".format(init_par[1]))
    #print (" B'(VO) =  {:1.4f} ".format(init_par[2]))
    best_par, cov_matrix = curve_fit(eos_murnaghan, volume, energy, p0 = init_par)
    residuals = energy - eos_murnaghan(volume, *best_par)
    ssr = np.sum(residuals**2)
    s_sq = ssr / (len(volume) - len(best_par))
    cov_diag = np.diag(cov_matrix)
    std_errors = np.sqrt(cov_diag)
    return best_par

def fit_and_plot(filename):
    """ read the data from the file，and fitting the curve based on Murnaghan equations, 
    and return the optimized parameters and E-V curve.
    """
    # read the file
    volume, energy = read_data(filename)   
    # fit the equations based on Murnaghan equation
    best_par = fit_murnaghan(volume, energy)
    # out put the optimized parameters   
    print ("Fit parameters:")
    print (" V0     =  {:1.4f} A^3 ".format(best_par[3]))
    print (" E0     =  {:1.4f} eV  ".format(best_par[0]))
    print (" B(V0)  =  {:1.4f} eV/A^3".format(best_par[1]))
    print (" B'(VO) =  {:1.4f} ".format(best_par[2]))
    data = np.concatenate((best_par[3].reshape(-1, 1),
                       best_par[0].reshape(-1, 1),
                       best_par[1].reshape(-1, 1),
                       best_par[2].reshape(-1, 1)), axis=1)
    np.savetxt('out_eosres.txt', data, delimiter=' ', fmt='%s')
    # generated the Murnaghan model based on the fitted parameters
    m_volume = np.linspace(volume.min(), volume.max(), 550) 
    m_energy = eos_murnaghan(m_volume, *best_par) 
    Data = np.loadtxt("curvefit.txt") 
    # plot the E-V curve
    V=np.array(Data[:,1])
    E0=np.array(best_par[0])
    E=np.array(Data[:,2])
    #P=np.array(cal_deriv(V,E)*(-V))
    B0=np.array(best_par[1])*160.217
    BP=np.array(best_par[2])
    #B2P=np.array(cal_second_deriv(P,B0))
    V0=np.array(best_par[3])
    kB=1.38e-23
    kB_j=8.617333262145e-5 
    I=np.array(Data[:,0])
    best_par_vf_list=[]
    for T in np.linspace(0, 1600, 161):  # Start from 0
        g_0 = 1.5
        sigma = 1.27262 + 1.67772 * np.log(V0 / V)
        gamma = g_0 * (V0 / V) ** sigma
        D = s * A * (V0 ** (1 / 6)) * ((B0 / M) ** (1 / 2)) * ((V0 / V) ** gamma)
        D = np.ravel(D)
    
        # Avoid division by zero for T=0
        T_safe = max(T, 1e-10)  # Use T_safe for calculations, ensuring it's never exactly 0
        Fvib = (9 / 8) * kB_j * D + kB_j * T_safe * (
            3 * np.log(1 - np.exp(-D / T_safe)) - debye_function(D / T_safe)
        )
        F = E + Fvib
        best_par_vf = fit_murnaghan(V, F)
        best_par_vf_list.append([T, best_par_vf[3], best_par_vf[0]])
    #for i, d in zip(I,D):
    #    wD = (d * kB) / (h*2*np.pi)
    #    w = np.linspace(0, wD, 3000)
    #    w = np.append(w, wD + (w[1] - w[0]))
    #    filename = "vdos_{}.out".format(i)
    #    with open(filename, 'w') as file:
    #        for j in range(len(w)):
    #            if w[j] <= wD:
    #                g = 9 * w[j]**2 / wD**3
    #            else:
    #                g = 0
                #file.write("{} {}\n".format(w[j], g))
    #pd.DataFrame(np.concatenate((V.reshape(-1,1), E.reshape(-1,1), B0.reshape(-1,1), BP.reshape(-1,1), B2P.reshape(-1,1),P.reshape(-1,1)),axis=1)).to_csv('E-V_fit.csv')
    best_par_vf=np.array(best_par_vf_list)
    return best_par, m_volume, m_energy, volume, energy, best_par_vf
def fit_curvefit(volume, energy):
    """
    Note: As per the current documentation (Scipy V1.1.0), sigma (yerr) must be:
        None or M-length sequence or MxM array, optional
    Therefore, replace:
        err_stdev = 0.2
    With:
        err_stdev = [0.2 for item in xdata]
    Or similar, to create an M-length sequence for this example.
    """
    volume, energy = read_data("curvefit.txt") 
    # fitting with Quadratic first and then get the guess parameters.
    p_coefs = np.polyfit(volume, energy, 2)
    # the lowest point of parabola dE/dV = 0 ( p_coefs = [c,b,a] ) V(min) = -b/2a
    p_min = - p_coefs[1]/(2.*p_coefs[0])
    # warn if min volume not in result range 
    if (p_min < volume.min() or p_min > volume.max()):
        print ("Warning: minimum volume not in range of results")
    # estimate the energy based the the lowest point of parabola   
    E0 = np.polyval(p_coefs, p_min)
    # estimate the bulk modules
    B0 = 2.*p_coefs[2]*p_min
    # guess the parameter (set BP as 4)
    init_par = [E0, B0, 4, p_min]
    pfit, pcov = \
         curve_fit(eos_murnaghan, volume, energy, p0 = init_par)
    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    
    pfit_curvefit = pfit/N
    perr_curvefit = np.array(error)/N
    return pfit_curvefit, perr_curvefit 
def debye_function(x):
    # Define the integrand function
    integrand = lambda t: t**3 / (np.exp(t) - 1)

    # Initialize the variable to store the total integral
    integral = np.zeros_like(x)

    # Calculate the integral using quad function
    for i, ele in enumerate(x):
            if ele < 155:  # Use 'ele' for comparison
                result, _ = quad(integrand, 0, ele)
                integral[i] = (3 / ele**3) * result
            else:
                # Perform a different calculation based on the condition
                integral[i] = 6.493939 * (3 / ele**3)  # Replace this with your actual calculation

    return integral
def debye_cv_function(x):
    # Define the integrand function with log-sum-exp trick
    integrand =lambda t: t ** 4 *np.exp(t)/ (np.exp(t) - 1.)**2

    cv_integral = np.zeros_like(x)
    # Calculate the integral using quad function
    for i, ele in enumerate(x):
        if ele < 155:  # Use 'ele' for comparison
            cv_integral[i], _ = quad(integrand, 0, ele, epsrel=1e-8)
        else:
            cv_integral[i] = (3 / ele**3) * 8.617333262145e-5 * sum(element_counts) * 4. / 5. * np.pi**4

    return cv_integral

def calculate_Fvib(filename):
    # Constants
    volume, energy = read_data(filename) 
    best_par = fit_murnaghan(volume, energy)
    best_par, m_volume, m_energy, volume, energy, best_par_vf = fit_and_plot(filename)
    E0 = best_par[0]
    V0 = best_par[3]
    B0 = best_par[1] * 160.217
    BP = best_par[2]
    result_list = []

    previous_V = None  # Used to calculate dVdT
    previous_T = None  # Used to calculate dVdT

    for idx, item in enumerate(best_par_vf):
        T = item[0]
        V = item[1]
        g_0 = 1.5
        sigma = 1.27262 + 1.67772 * np.log(V0 / V)
        gamma = g_0 * (V0 / V) ** sigma
        kB_j = 8.617333262145e-5
        D = s * A * (V0 ** (1 / 6)) * ((B0 / M) ** (1 / 2)) * ((V0 / V) ** gamma)
        D = np.ravel(D)
        # Calculate Fv
        Fd = (9 / 8) * kB_j * D + kB_j * T * (3 * np.log(1 - np.exp(-D / T)) - debye_function(D / T))
        Fv = E0 + (9 / 8) * kB_j * D + kB_j * T * (3 * np.log(1 - np.exp(-D / T)) - debye_function(D / T))
        Sv = kB_j * (-3 * np.log(1 - np.exp(-D / T)) + 4 * debye_function(D / T)) * 96450
        Cv = 9 * kB_j * (T / D) ** 3 * debye_cv_function(D / T) * 96450

        # Calculate dVdT and CTE
        if idx == 0:
            CTE = 0  # Set the first CTE value to 0
        else:
            dVdT = (V - previous_V) / (T - previous_T)
            CTE = dVdT / V

        previous_V = V
        previous_T = T

        result_list.append([T, V, float(Fv), float(Fd), float(Sv), float(Cv), CTE])

    result = np.array(result_list, dtype=object)
    return result

best_par, m_volume, m_energy, volume, energy, best_par_vf = fit_and_plot("curvefit.txt")
pfit, perr = fit_curvefit(volume, energy)
result=calculate_Fvib("curvefit.txt")
df = pd.DataFrame(np.array(result), columns=['T', 'V', 'Fv', 'Fd', 'Sv', 'Cv', 'CTE'])
df.to_csv('Thermal.csv', index=False)
result_final=pd.read_csv('Thermal.csv')
T=result_final['T'].to_list()
V=result_final['V'].to_list()
Fv=result_final['Fv'].to_list()
Sv=result_final['Sv'].to_list()
Cv=result_final['Cv'].to_list()
CTE=result_final['CTE'].to_list()
fig, axs = plt.subplots(2, 3, figsize=(10, 8))
fig.tight_layout(pad=4.0)

# Plot E-V curve
axs[0, 0].plot(volume, energy, 'o', fillstyle='none', color='black',markersize=15)
axs[0, 0].plot(m_volume, m_energy, '-r', linewidth=3)
axs[0, 0].set_xlabel(r"Volume [$\rm{A}^3$]",fontsize=10)
axs[0, 0].set_ylabel(r"Energy [$\rm{eV}$]",fontsize=10)
axs[0, 0].tick_params(labelsize=10)
axs[0, 0].tick_params(axis='x', labelsize=10)
axs[0, 0].tick_params(axis='y', labelsize=10)

# Plot T vs Vcte
axs[0, 1].plot(T, V, '-g', label='Vcte', linewidth=3)
axs[0, 1].set_xlabel("Temperature (K)", fontsize=12)
axs[0, 1].set_ylabel("Volume expansion (A3)", fontsize=12)
axs[0, 1].tick_params(labelsize=10)
axs[0, 1].tick_params(axis='x', labelsize=10)
axs[0, 1].tick_params(axis='y', labelsize=10)

# Plot T vs Fv
axs[0, 2].plot(T, Fv, '-g', label='Fv', linewidth=3)
axs[0, 2].set_xlabel("Temperature (K)", fontsize=12)
axs[0, 2].set_ylabel("Free energy (eV/atom)", fontsize=12)
axs[0, 2].tick_params(labelsize=10)
axs[0, 2].tick_params(axis='x', labelsize=10)
axs[0, 2].tick_params(axis='y', labelsize=10)

# Plot T vs Svib
axs[1, 0].plot(T, Sv, '-b', label='Svib', linewidth=3)
axs[1, 0].set_xlabel("Temperature (K)", fontsize=12)
axs[1, 0].set_ylabel("Svib (J/K)", fontsize=12)
axs[1, 0].tick_params(labelsize=10)
axs[1, 0].tick_params(axis='x', labelsize=10)
axs[1, 0].tick_params(axis='y', labelsize=10)

# Plot T vs Cvib
axs[1, 1].plot(T, Cv, '-m', label='Cvib', linewidth=3)
axs[1, 1].set_xlabel("Temperature (K)", fontsize=12)
axs[1, 1].set_ylabel("Cvib(J/K)", fontsize=12)
axs[1, 1].tick_params(labelsize=10)
axs[1, 1].tick_params(axis='x', labelsize=10)
axs[1, 1].tick_params(axis='y', labelsize=10)

# Plot T vs CTE
axs[1, 2].plot(T, CTE, '-b', label='CTE', linewidth=3)
axs[1, 2].set_xlabel("Temperature (K)", fontsize=12)
axs[1, 2].set_ylabel("CTE (1/K)", fontsize=12)
axs[1, 2].tick_params(labelsize=10)

axs[0, 0].legend(frameon=False)
axs[0, 1].legend(frameon=False)
axs[0, 2].legend(frameon=False)
axs[1, 0].legend(frameon=False)
axs[1, 1].legend(frameon=False)
axs[1, 1].legend(frameon=False)

plt.show()
plt.savefig('E-V_Debye.png', format='png', dpi=330)

print("\n# Fit parameters and parameter errors from curve_fit method :")
print("pfit = ", pfit)
print("perr = ", perr)
