#!/usr/bin/env python
import ase
import schnetpack as sch
import numpy as np
import os, sys, shutil
from schnetpack import AtomsData, Properties
from schnetpack.data import AtomsConverter
import schnetpack.train as trn
from schnetpack.environment import APNetEnvironmentProvider
from schnetpack.md import System as sch_sys
from schnetpack.md.utils import MDUnits
import torch
from torch.optim import Adam
from simtk.openmm.app import *
from simtk.openmm import *
import simtk.openmm as mm
from simtk.unit import *

dA_min = -228.3671507582417632
dB_min = -344.2359084163039711
dAp_min = -228.9282132443254056
dBp_min = -343.8068210638700180
monomer_d1 = dA_min + dB_min
monomer_d2 = dAp_min + dBp_min
shift = monomer_d1 - monomer_d2
h2kj = 2625.5

def add_drude(pdb, modeller, xyz):
    for i in range(len(xyz)):
        pdb.positions[i] = xyz[i]*nanometer
    modeller_new = Modeller(pdb.topology, pdb.positions)
    return modeller_new

def get_qm_eng_force(traj):
    eng = []
    #force = []
    for d in traj:
        energy = np.array([float(list(d.info.values())[-1])], dtype=np.float32)
        eng.append(energy)
    #for d in traj_force:
    #    force.append(d.get_positions()*h2kj*MDUnits.angs2bohr)
    return np.asarray(eng)

def get_omm_energy(traj):
    temperature = 300*kelvin
    Topology.loadBondDefinitions("../pb_residues.xml")
    pdb = PDBFile("../emim.pdb")
    res_dict = []
    for res in pdb.topology.residues():
        res_list = []
        for i in range(len(res._atoms)):
            res_list.append(res._atoms[i].index)
        res_dict.append(res_list)

    integ_md = DrudeSCFIntegrator(0.0005*picoseconds)
    integ_md.setMinimizationErrorTolerance(0.01)
    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = ForceField("../pb.xml")
    modeller.addExtraParticles(forcefield)
    system = forcefield.createSystem(modeller.topology, constraints=None, rigidWater=True)
    nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
    customNonbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == CustomNonbondedForce][0]
    drudeForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == DrudeForce][0]

    hyper = CustomBondForce('step(r-rhyper)*((r-rhyper)*khyper)^powh')
    hyper.addGlobalParameter('khyper', 100.0)
    hyper.addGlobalParameter('rhyper', 0.02)
    hyper.addGlobalParameter('powh', 6)
    system.addForce(hyper)

    for i in range(drudeForce.getNumParticles()):
        param = drudeForce.getParticleParameters(i)
        drude = param[0]
        parent = param[1]
        hyper.addBond(drude, parent)

    real_atoms = []
    for i in range(system.getNumParticles()):
        if system.getParticleMass(i)/dalton > 1.0:
            real_atoms.append(i)

    for i in range(system.getNumParticles()):
        system.setParticleMass(i,0)

    platform = Platform.getPlatformByName('Reference')
    simmd = Simulation(modeller.topology, system, integ_md, platform)
    simmd.context.setPositions(modeller.positions)

    particleMap = {}
    for i in range(drudeForce.getNumParticles()):
        particleMap[drudeForce.getParticleParameters(i)[0]] = i

    flagexceptions = {}
    for i in range(nbondedForce.getNumExceptions()):
        (particle1, particle2, charge, sigma, epsilon) = nbondedForce.getExceptionParameters(i)
        string1=str(particle1)+"_"+str(particle2)
        string2=str(particle2)+"_"+str(particle1)
        flagexceptions[string1]=1
        flagexceptions[string2]=1

    flagexclusions = {}
    for i in range(customNonbondedForce.getNumExclusions()):
        (particle1, particle2) = customNonbondedForce.getExclusionParticles(i)
        string1=str(particle1)+"_"+str(particle2)
        string2=str(particle2)+"_"+str(particle1)
        flagexclusions[string1]=1
        flagexclusions[string2]=1

    for resi in simmd.topology.residues():
        for i in range(len(resi._atoms)-1):
            for j in range(i+1,len(resi._atoms)):
                (indi,indj) = (resi._atoms[i].index, resi._atoms[j].index)
                # here it doesn't matter if we already have this, since we pass the "True" flag
                nbondedForce.addException(indi,indj,0,1,0,True)
                # make sure we don't already exlude this customnonbond
                string1=str(indi)+"_"+str(indj)
                string2=str(indj)+"_"+str(indi)
                if string1 in flagexclusions and string2 in flagexclusions:
                    continue
                else:
                    customNonbondedForce.addExclusion(indi,indj)
                # add thole if we're excluding two drudes
                if indi in particleMap and indj in particleMap:
                    # make sure we don't already have this screened pair
                    if string1 in flagexceptions or string2 in flagexceptions:
                        continue
                    else:
                        drudei = particleMap[indi]
                        drudej = particleMap[indj]
                        drudeForce.addScreenedPair(drudei, drudej, 2.0)

    for i in range(system.getNumForces()):
        f = system.getForce(i)
        type(f)
        f.setForceGroup(i)

    state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
    position = state.getPositions()
    simmd.context.reinitialize()
    simmd.context.setPositions(position)
    
    omm_eng = []
    omm_forces = []
    z = 0
    for i in range(len(traj)):
        print(i)
        xyz_pos = traj[i].get_positions()
        omm_pos = []
        for k in range(len(xyz_pos)):
            omm_pos.append(Vec3(xyz_pos[k][0]/10.0, xyz_pos[k][1]/10.0, xyz_pos[k][2]/10.0))
        modeller_new = add_drude(pdb, modeller, omm_pos)
        modeller_new.addExtraParticles(forcefield)
        simmd.context.setPositions(modeller_new.positions)
        simmd.step(1)
        for j in range(system.getNumForces()):
            f = system.getForce(j)
            if f.__class__.__name__ == "CustomNonbondedForce":
                custom_nonbond = simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()/kilojoule_per_mole
                custom_nonbond_force = simmd.context.getState(getForces=True, groups=2**j).getForces(asNumpy=True)/kilojoule_per_mole*nanometer
            if f.__class__.__name__ == "NonbondedForce":
                nbond_eng = simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()/kilojoule_per_mole
                nbond_eng_force = simmd.context.getState(getForces=True, groups=2**j).getForces(asNumpy=True)/kilojoule_per_mole*nanometer
            if f.__class__.__name__ == "DrudeForce":
                drude_eng = simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()/kilojoule_per_mole
                drude_eng_force = simmd.context.getState(getForces=True, groups=2**j).getForces(asNumpy=True)/kilojoule_per_mole*nanometer
            if f.__class__.__name__ == "CustomBondForce":
                custom_bond = simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()/kilojoule_per_mole
                custom_bond_force = simmd.context.getState(getForces=True, groups=2**j).getForces(asNumpy=True)/kilojoule_per_mole*nanometer

        total_nonbond_eng = custom_nonbond + nbond_eng + drude_eng + custom_bond
        total_nonbond_eng = np.array([total_nonbond_eng], dtype=np.float32)
        total_nonbond_force = custom_nonbond_force + nbond_eng_force + drude_eng_force + custom_bond_force
        total_nonbond_force = total_nonbond_force[real_atoms]
        omm_eng.append(total_nonbond_eng)
        omm_forces.append(total_nonbond_force/10.0)
    return np.asarray(omm_eng), np.asarray(omm_forces), res_dict

def get_diabat_A(traj, res_dict):
    diabatA_nn = torch.load("./nhc_model")
    res_list = res_dict[0]
    dA_eng = []
    dA_forces = []
    converter = AtomsConverter(device="cuda")
    for i in range(len(traj)):
        new_traj = traj[i][res_list]
        inputs = converter(new_traj)
        result = diabatA_nn(inputs)
        energy = result['energy'].detach().cpu().numpy()[0][0]
        forces = result['forces'].detach().cpu().numpy()[0]
        dA_eng.append(np.array([energy], dtype=np.float32))
        for j in range(len(res_dict[1])): forces = np.append(forces, [[0.0, 0.0, 0.0]], axis=0)
        dA_forces.append(forces)
    return np.asarray(dA_eng), np.asarray(dA_forces)

def get_diabat_B(traj, res_dict):
    diabatB_nn = torch.load("./acetic_model")
    res_list = res_dict[1]
    dB_eng = []
    dB_forces = []
    converter = AtomsConverter(device="cuda")
    for i in range(len(traj)):
        new_traj = traj[i][res_list]
        inputs = converter(new_traj)
        result = diabatB_nn(inputs)
        energy = result['energy'].detach().cpu().numpy()[0][0]
        forces = result['forces'].detach().cpu().numpy()[0]
        dB_eng.append(np.array([energy], dtype=np.float32))
        for j in range(len(res_dict[0])): forces = np.insert(forces, 0, [0.0, 0.0, 0.0], axis=0)
        dB_forces.append(forces)
    return np.asarray(dB_eng), np.asarray(dB_forces)

def rm_cutoff_energy(traj, residual_eng, cutoff, res_dict):
    inds = len(traj[0].get_chemical_symbols())
    inds = [i for i in range(inds)]
    dists = []
    monA = res_dict[0]
    monB = res_dict[1]
    zero = []
    for i in range(len(traj)):
        c = True
        for k in range(len(monA)):
            dist = traj[i].get_distances(monA[k], monB)
            cutoffs = 0.5 * (np.cos(dist * np.pi / cutoff) + 1.0)
            cutoffs *= (dist < cutoff)
            if not np.all(cutoffs == 0):
                c = False
                if residual_eng[i] == 0.0:
                    bad.append(i)
                    print("Structure {} inside cutoff with 0.0 energy".format(i))
                    raise Exception
                break
        if c:
            zero.append(i)
            print(i)
            residual_eng[i] = np.array([0.0], dtype=np.float32)
    return residual_eng, zero

# loss function
def loss(batch, result):
    rho_tradeoff = 0.1

    # compute the mean squared error on the energies
    diff_energy = batch['energy']-result['energy']
    err_sq_energy = torch.mean(diff_energy ** 2)

    # build the combined loss function
    err_sq = err_sq_energy 

    return err_sq

def main():
    """
    traj = ase.io.read("test.xyz", index="0:100")
    
    eng = get_qm_eng_force(traj)
    qm_eng = eng

    omm_eng, omm_forces, res_dict = get_omm_energy(traj)
    
    residual_eng = qm_eng - omm_eng
    np.savetxt("residual_eng_add.txt", residual_eng)
    residual_eng = np.loadtxt("residual_eng_add.txt")

    good = np.where(residual_eng > -800)[0].tolist()
    residual_eng = residual_eng[good]
    new_traj = []
    for i in good: new_traj.append(traj[i])
    traj = new_traj
    residual_eng = np.expand_dims(residual_eng, -1)
     
    properties = []
    global_id = [np.array([i], dtype=np.int) for i in range(len(traj))]
    for i in range(len(traj)):
        syms = traj[i].get_atomic_numbers()
        properties.append({'energy': residual_eng[i], 'global_id': global_id[i], 'ZA': syms[res_dict[0]], 'ZB': syms[res_dict[1]]})
    """
    #if os.path.isfile("dimer_2_13.db"): os.remove("dimer_2_13.db")
    d_set = AtomsData("./dimer_2_13.db", available_properties=['energy', 'global_id', 'ZA', 'ZB'], environment_provider=APNetEnvironmentProvider())
    d_set.add_systems(traj, properties)
    
    #if os.path.isfile("split.npz"): os.remove("split.npz")
    train, val, test = sch.train_test_split(
        data = d_set,
        num_train = 0.8,
        num_val = 0.15,
        split_file = 'split.npz'
    )
    
    train_loader = sch.AtomsLoader(train, batch_size=100, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = sch.AtomsLoader(val, batch_size=100, shuffle=True, pin_memory=True, num_workers=4)
    
    #mean, stddev = train_loader.get_statistics("energy")

    apnet = sch.representation.APNet(
        n_ap=21,
        elements=frozenset((1, 6, 7, 8)),
        cutoff_radius=4.,
        cutoff_radius2=4.,
        sym_cut=4.5,
        cutoff=sch.nn.cutoff.CosineCutoff,
        morse_bond=[2, 3],
        morse_mu=0.95,
        morse_beta=30.0,
        morse_res=res_dict[0],
     )

    output = sch.atomistic.Pairwise(n_in=128, property='energy', derivative='forces', negative_dr=True, elements=frozenset((1, 6, 7, 8)), n_acsf=43, n_apf=21)
    model = sch.PairwiseModel(representation=apnet, output_modules=output)
    optimizer = Adam(model.parameters(), lr=0.0005)
    
    if os.path.isdir('checkpoints'): shutil.rmtree('checkpoints')
    if os.path.isfile('log.csv'): os.remove('log.csv')
   
    metrics = [sch.metrics.MeanAbsoluteError('energy')]

    hooks = [
        trn.CSVHook(log_path='./', metrics=metrics),
            trn.ReduceLROnPlateauHook(
            optimizer,
            patience=5, factor=0.8, min_lr=1e-6,
            stop_after_min=True
        )
    ]

    trainer = trn.Trainer(
        model_path='./',
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
    )

    print("Train")
    device = "cuda"
    n_epochs = 1000
    trainer.train(device=device)

if __name__ == "__main__":
    main()
