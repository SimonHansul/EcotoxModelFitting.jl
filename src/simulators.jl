
"""
    generate_fitting_simulator(defaultparams::ComponentArray, prior::Prior, simulator::Function)::Function

Attempt to define a generic simulator function, based on the information given to the ModelFitting object. <br> 
This function is called internally when calling `PMCBackend`, but can be overwritten with a custom definition if needed. <br>
I am sure there are use-cases where this will fail. For the cases tested so far though, it worked fine and was quite helpful.

The generated "fitting simulator" 
    - Is a wrapper around the provided `simulator` argument
    - Expects the parameter values as vector of floats
    - Assures that parameters are assigned correctly to a copy of the defaultparams
    - Pre-allocates copies of the defaultparams with account for multithreading
    - Deals with priors provided as `Hyperdist` (currently not in a full hierarchical approach, TBC)
    - Defines a second method for the fitting_simulator which dispatches to the original `simulator` function (useful in conjunction with the `PMCBackend` struct)

"""
function generate_fitting_simulator(defaultparams, prior::Prior, simulator::Function)::Function

    # when using mult-threading, we create a copy of the parameter object for each thread
    pfit = [deepcopy(defaultparams) for _ in 1:Threads.nthreads()]

    # matching parameter labels to indices
    pfit_labels = ComponentArrays.labels(pfit[1])
    idxs = [findfirst(x -> x == l, pfit_labels) for l in prior.labels]

    function fitting_simulator(pvec::Vector{R}; kwargs...) where R <: Real
        
        psim = pfit[threadid()] # pick the parameter copy for the current thread

        psim[idxs[.!prior.is_hyper]] = pvec[.!prior.is_hyper] # assign "normal" parameters directly
        psim[idxs[prior.is_hyper]] = [gendist(h) for (gendist,h) in zip(prior.gendists, pvec[prior.is_hyper])] # assign hyperparameters through the appropriate gendist function

        return simulator(psim; kwargs...)
    end

    # define an additional method that takes the componentarray as argument, and dispatches to the original function
    fitting_simulator(p::CA; kwargs...) where CA <: ComponentArray = simulator(p; kwargs...)

    return fitting_simulator

end
