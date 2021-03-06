%module openmmtorch

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>

%{
#include "TorchForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace TorchPlugin {

class TorchForce : public OpenMM::Force {
public:
    TorchForce(const std::string& file);
    const std::string& getFile() const;
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;

    /*
     * Add methods for casting a Force to a TorchForce.
    */
    %extend {
        static TorchPlugin::TorchForce& cast(OpenMM::Force& force) {
            return dynamic_cast<TorchPlugin::TorchForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<TorchPlugin::TorchForce*>(&force) != NULL);
        }
    }
};

}
