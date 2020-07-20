import simtk.openmm as mm
import simtk.unit as unit
import openmmtorch as ommt

# you may need to add libtorch to LD_LIBRARY_PATH before running the script
system = mm.System()
for i in range(3):
    system.addParticle(1.0)
f = ommt.TorchForce('../tests/central.pt')
system.addForce(f)
integrator = mm.VerletIntegrator(0.001)
platform = mm.Platform.getPlatformByName('CUDA')
context = mm.Context(system, integrator, platform)
positions = [mm.Vec3(3, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(3, 4, 0)]
context.setPositions(positions)
print(context.getState(getEnergy=True).getPotentialEnergy())
# should give 50.0 kJ/mol

