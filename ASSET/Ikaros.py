import asset_asrl as ast
import asset_asrl.OptimalControl as oc
import asset_asrl.Astro.Constants as c
import spiceypy as spice
from asset_asrl.Astro.Date import datetime, datetime_to_jd, jd_to_datetime
from asset_asrl.Astro.AstroFrames import NBodyFrame
from asset_asrl.Astro.AstroModels import NBody_SpiceSolarSail, NBody
import matplotlib.pyplot as plt
import numpy as np
# norm = np.linalg.norm

vf = ast.VectorFunctions
Args = vf.Arguments

################################################################################
# INITIALIZATION OF SPICE AND FRAME STUFF
spacecraft_mass = 307
sail_mass= 16
sail_min_thickness = 7.5e-6
sail_size = 14*14

spice.furnsh('.\kernels\AstroDemo4\BasicKernel.txt')
init_datetime = datetime(2010, 5, 21)
JD0 = datetime_to_jd(init_datetime)
real_final_datetime = datetime(2010, 12, 8)
real_JDF = datetime_to_jd(real_final_datetime)
max_TOF = (real_JDF - JD0)*2
max_JDF = JD0 + max_TOF
print ("max_TOF days",max_TOF)

nFrame = NBodyFrame("SUN",c.MuSun,c.AU,JD0,max_JDF)
nFrame.AddSpiceBodies(["SUN", "EARTH", "VENUS"])

################################################################################
# PROBLEM SETUP
ref_coeff = 2
nFrame.Add_SolarSail(ref_coeff-1,sail_size,spacecraft_mass,zero_alpha=False,spherical_control=False)
nb_ss_ode = NBody_SpiceSolarSail(nFrame, ActiveAltBodies='All', Enable_J2=False, Enable_P1_Acc=False)
# nb_ode = NBody(nFrame, ActiveAltBodies='All', Enable_J2=False, Enable_P1_Acc=False)
earth_traj_total = nFrame.AltBodyTrajs["EARTH"]
earth_table = vf.InterpTable1D(earth_traj_total[:,6], earth_traj_total[:,:6], kind='cubic')
venus_traj_total = nFrame.AltBodyTrajs["VENUS"]
venus_table = vf.InterpTable1D(venus_traj_total[:,6], venus_traj_total[:,:6], kind='cubic')
print("venus final state:",venus_traj_total[-1])

def RendCon(table):
        args = Args(7)
        states = args.head(6)
        rend_time = args[6]

        target_states = table(rend_time)

        return target_states - states

def InterceptCon(table):
    args = Args(4)
    pos = args.head(3)
    int_time = args[3]
    return table(int_time)[:3] - pos

def FlybyDist(table):
    args = Args(4)
    pos = args.head(3)
    int_time = args[3]
    return (table(int_time)[:3] - pos).norm()

def Radial_Ulaw():
    r = Args(3)
    return r.normalized()
     

ikaros_ic = np.zeros((10))
# ikaros_ic = np.zeros((9))
ikaros_ic[0:7] = earth_traj_total[0]
ikaros_ic[7] = -1
print("sc_X0",ikaros_ic)
tf_ig = (real_JDF-JD0)*c.day/nb_ss_ode.tstar
print("tf_ig nd",tf_ig)
print("c.day",c.day)
print("nb_ode.tstar",nb_ss_ode.tstar)
print("tf_ig days",tf_ig*nb_ss_ode.tstar/c.day)

print("nb_ode.lstar",nb_ss_ode.lstar)

# ss_integ = nb_ss_ode.integrator(0.1,Radial_Ulaw(),[0,1,2])
ss_integ = nb_ss_ode.integrator(0.1)
ikaros_ig = ss_integ.integrate_dense(ikaros_ic,tf_ig)
ikaros_ig = np.array(ikaros_ig)
# print(ikaros_ig)
######### FIX NBODY GRAV (NBODYFRAME.PY LINE 173)
transfer_phase = nb_ss_ode.phase('LGL3')
transfer_phase.setTraj(ikaros_ig,2000)
x0_con = transfer_phase.addBoundaryValue('Front',[0,1,2,6], earth_traj_total[0,[0,1,2,6]])

flyby_range_km = 80800
flyby_range_nd = flyby_range_km*c.kilometer/nb_ss_ode.lstar
print("flyby_range",flyby_range_nd)
transfer_phase.setStaticParams([flyby_range_nd])
# tof_con = transfer_phase.addEqualCon('StaticParams',Args(1)[0]-tf_ig,[],[],[0])
transfer_phase.addLUVarBound('StaticParams', 0, c.RadiusVenus/nb_ss_ode.lstar, flyby_range_nd*2)
# transfer_phase.addLowerDeltaTimeBound(tf_ig*0.5)

# transfer_phase.addEqualCon('Back', InterceptCon(venus_table), [0,1,2,6],[],[])
# transfer_phase.addInequalCon('Back', FlybyDist(venus_table), [0,1,2,6],[],[0])
transfer_phase.addUpperFuncBound('Back',FlybyDist(venus_table), [0,1,2,6],flyby_range_nd)
# transfer_phase.setControlMode('HighestOrderSpline')
transfer_phase.addEqualCon("Path",Args(3).norm()-1.0,[7,8,9])
# transfer_phase.addLUVarBound("Path",7,0,c.pi)
# transfer_phase.addLUVarBound("Path",8,0,c.pi)
transfer_phase.solve()
# ocp = oc.OptimalControlProblem()
# ocp.addPhase(transfer_phase)
# ocp.solve()
# ocp.removePhase(transfer_phase)


# Ikaros solved intercept of Venus
ikaros_sol = np.array(transfer_phase.returnTraj())
sol_tof = ikaros_sol[-1,6]
print("ikaros solution tof (d)", sol_tof*nb_ss_ode.tstar/c.day)

ikaros_new_ic = ikaros_sol[0,0:7]
nb_base_ode = NBody(nFrame, ActiveAltBodies='All', Enable_J2=False, Enable_P1_Acc=False)
base_integ = nb_base_ode.integrator(0.1)
ikaros_no_ss = base_integ.integrate_dense(ikaros_new_ic,sol_tof,2001)
ikaros_no_ss = np.array(ikaros_no_ss)

earth_traj = base_integ.integrate_dense(earth_traj_total[0],sol_tof,2001)
earth_traj = np.array(earth_traj)

venus_traj = base_integ.integrate_dense(venus_traj_total[0],sol_tof,2001)
venus_traj = np.array(venus_traj)


opt_phase = nb_ss_ode.phase('LGL5')
opt_phase.setTraj(ikaros_sol,1000)

opt_phase.addBoundaryValue('Front',[0,1,2,3,4,5,6],ikaros_sol[0,[0,1,2,3,4,5,6]])
opt_phase.setStaticParams([flyby_range_nd])
# opt_phase.addLUVarBound('StaticParams', 0, tf_ig*0.5, tf_ig*1.5)
# opt_phase.addLowerDeltaTimeBound(tf_ig*0.5)
opt_phase.addUpperFuncBound('Back',FlybyDist(venus_table), [0,1,2,6],flyby_range_nd)
opt_phase.addLUVarBound('StaticParams', 0, c.RadiusVenus/nb_ss_ode.lstar, flyby_range_nd*2)
opt_phase.addLUVarBound('Path', 7, -1, 1)
opt_phase.addLUVarBound('Path', 8, -1, 1)
opt_phase.addLUVarBound('Path', 9, -1, 1)
# opt_phase.setControlMode('HighestOrderSpline')
# opt_phase.addEqualCon("Path",Args(3).norm()-1.0,[7,8,9])


opt_phase.addDeltaTimeObjective(1.0)
opt_phase.optimize()
# ocp.optimize()
ikaros_opt = np.array(opt_phase.returnTraj())


################################################################################
# VISUALIZATION
# Traj Plot
fig, ax = plt.subplots()

ax.scatter([0],[0],s=20,c='yellow',label="Sun")

ax.plot(earth_traj[:,0],earth_traj[:,1],c='blue', linestyle='dotted',label="Earth")
ax.scatter([earth_traj[-1,0]],[earth_traj[-1,1]],s=20,c='blue')

ax.plot(venus_traj[:,0],venus_traj[:,1],c='green', linestyle='dotted',label="Venus")
ax.scatter([venus_traj[-1,0]],[venus_traj[-1,1]],s=20,c='green')

ax.plot(ikaros_sol[:,0],ikaros_sol[:,1],c='red',label="Ikaros Solution with Solar Sail")
ax.scatter([ikaros_sol[-1,0]],[ikaros_sol[-1,1]],s=20,c='red')

# ax.plot(ikaros_no_ss[:,0],ikaros_no_ss[:,1],c='black', linestyle='dotted',label="Ikaros IC, no Solar Sail")
# ax.scatter([ikaros_no_ss[-1,0]],[ikaros_no_ss[-1,1]],s=20,c='black')

ax.plot(ikaros_opt[:,0],ikaros_opt[:,1],c='black', linestyle='dotted',label="Ikaros IC, optimized with SS")
ax.scatter([ikaros_opt[-1,0]],[ikaros_opt[-1,1]],s=20,c='black')

ax.set_xlabel('x (nd)')
ax.set_ylabel('y (nd)')
ax.set_title('IKAROS Transfer from Earth to Venus')
ax.axis('equal')
ax.legend()

# Disp Plot
fig1,ax1 = plt.subplots()
ikaros_delta_mags_base = np.zeros((len(ikaros_sol)))
ikaros_delta_mags_opt = np.zeros((len(ikaros_sol)))
for index,ss_state in enumerate(ikaros_sol[:,:3]):
    base_pos = ikaros_no_ss[index,:3]
    ikaros_delta_mags_base[index] = np.linalg.norm(ss_state - base_pos)
    opt_pos = ikaros_opt[index,:3]
    ikaros_delta_mags_opt[index] = np.linalg.norm(ss_state - opt_pos)

tVec_days = ikaros_sol[:,6]*nb_ss_ode.tstar/c.day
ax1.plot(tVec_days,ikaros_delta_mags_base*nb_ss_ode.lstar/1000, label="Sol vs Base")
ax1.plot(tVec_days,ikaros_delta_mags_opt*nb_ss_ode.lstar/1000, label="Sol vs Opt")
ax1.set_xlabel('time (d)')
ax1.set_ylabel('disp (km)')
ax1.set_title('Position Displacement Due to Solar Sail')
ax1.set_yscale('log')
ax1.legend()

# Control Plot
fig2,ax2 = plt.subplots()
controlVec1 = ikaros_opt[:,7]
controlVec2 = ikaros_opt[:,8]
controlVec3 = ikaros_opt[:,9]
ax2.plot(tVec_days,controlVec1,label="X")
ax2.plot(tVec_days,controlVec2,label="Y")
ax2.plot(tVec_days,controlVec3,label="Z")
# ax2.plot(tVec_days,controlVec1,label="Yaw")
# ax2.plot(tVec_days,controlVec2,label="Pitch")
ax2.set_xlabel('time (d)')
ax2.set_ylabel('Pointing Vector Components')
ax2.set_title('Solar Sail Pointing Vector Components')
ax2.legend()

# Dist to Venus Plot
fig3, ax3 = plt.subplots()
ikaros_venus_dist = np.zeros((len(ikaros_sol)))
for index,ss_state in enumerate(ikaros_sol[:,:7]):
    venus_pos = venus_table(ss_state[6])[:3]
    ikaros_venus_dist[index] = np.linalg.norm(ss_state[:3] - venus_pos)

ax3.plot(tVec_days,ikaros_venus_dist*nb_ss_ode.lstar/1000)
ax3.plot([tVec_days[0],tVec_days[-1]],[flyby_range_km,flyby_range_km])
ax3.set_xlabel('time (d)')
ax3.set_ylabel('Distance (km)')
ax3.set_title('IKAROS Distance From Venus')
print("Final Dist from Venus (km)",ikaros_venus_dist[-1])
ax3.set_yscale('log')

plt.show()