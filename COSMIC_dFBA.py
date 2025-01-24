import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

# Placeholder functions to replace MATLAB-specific operations
def change_objective(model, rxn_id, coefficient):
    model.objective = {rxn_id: coefficient}

def optimize_cb_model(model, method="max"):
    # Mock function to replace optimizeCbModel
    solution = model.optimize() if method == "max" else model.optimize_minimal()
    return solution

def simulate_model(model, rxns, rxn_in, v_in, priority, objs, fix_flx, fix_comp):
    """Simulate the metabolic model using specified uptake rates and report secretion fluxes."""
    m1 = model.copy()

    # Update reaction bounds
    for i, rxn_id in enumerate(rxn_in):
        m1.reactions.get_by_id(rxn_id).lower_bound = v_in[i]
        if rxn_id in fix_comp:
            idx = fix_comp.index(rxn_id)
            m1.reactions.get_by_id(rxn_id).upper_bound = fix_flx[idx] * v_in[i]

    for i, rxn_id in enumerate(priority):
        change_objective(m1, rxn_id, 1)
        sol = optimize_cb_model(m1)
        m1.reactions.get_by_id(rxn_id).lower_bound = objs[i] * sol.objective_value

    sol = optimize_cb_model(m1, method="min")
    x = sol.fluxes

    v = np.zeros(len(rxns))
    for i, rxn_id in enumerate(rxns):
        v[i] = x[rxn_id]

    v[3:] *= 0.35
    return v

def phase_progress(c, comp_names, phase_classifier):
    """Determine progress towards production phase."""
    cx = np.array([c[comp_names.index(comp)] for comp in phase_classifier['components']])
    L = np.array(phase_classifier['Beta']) * np.array(phase_classifier['scaling'])
    K = phase_classifier['Bias']
    E = phase_classifier['log_coeff']

    w = np.dot(cx, L) + K
    w = w * E[1] + E[0]
    f = np.exp(w) / (1 + np.exp(w))
    return f

def evaluate_fluxes(c, kinetic):
    """Evaluate fluxes using concentrations and Michaelis-Menten Rate Laws."""
    v = -kinetic[0] * c / (kinetic[1] + c)
    return np.minimum(v, 0)

def eval_reactor(c, v, c_in):
    """Evaluate ODE function (conservation of mass)."""
    D = 1
    c = np.maximum(c, 0)
    et = np.ones_like(c)
    et[0] = 0
    dc = np.zeros_like(c)

    for i in range(len(dc)):
        if i < 3:
            dc[i] = v[i] * c[i]
        else:
            dc[i] = v[i] * c[0] + D * (c_in[i] - et[i] * c[i])

    return dc

def cosmic_dfba(model, time_range, dfba_data, fix_flx, fix_comp, notes):
    """Perform dynamic FBA simulation."""
    np_components = len(dfba_data['components'])
    time = np.zeros(10000)
    profiles = np.zeros((10000, np_components))
    flux_growth = np.zeros((10000, np_components))
    flux_prod = np.zeros((10000, np_components))
    phase_transition = np.zeros(10000)

    ctr = 0
    rtol = 1e-2
    h = 1e-4
    t_curr = time_range[0]
    done = False

    c0 = np.array(dfba_data['initial_concentrations'])
    cmp_names = dfba_data['component_names']
    components = dfba_data['components']

    c_in = np.array([dfba_data['Perfusion']['concentrations'][dfba_data['Perfusion']['components'].index(cmp)] for cmp in cmp_names])
    
    kinetic_comps = dfba_data['kinetic']['components_names']
    kinetic_growth = np.array([dfba_data['kinetic']['Vm_growth'], dfba_data['kinetic']['Km']])
    kinetic_prod = np.array([dfba_data['kinetic']['Vm_prod'], dfba_data['kinetic']['Km']])

    p_growth = np.array(dfba_data['Objectives']['priority_growth'])
    f_growth = np.array(dfba_data['Objectives']['c_growth'])
    valid_growth = f_growth >= 0
    p_growth, f_growth = p_growth[valid_growth], f_growth[valid_growth]

    p_prod = np.array(dfba_data['Objectives']['priority_prod'])
    f_prod = np.array(dfba_data['Objectives']['c_prod'])
    valid_prod = f_prod >= 0
    p_prod, f_prod = p_prod[valid_prod], f_prod[valid_prod]

    classification_data = dfba_data['Phase_classification']

    with tqdm(total=time_range[1], desc="Simulating dFBA") as progress:
        while not done:
            # Single step
            kc = np.array([c0[components.index(comp)] for comp in kinetic_comps])
            vkg = evaluate_fluxes(kc, kinetic_growth)
            vkp = evaluate_fluxes(kc, kinetic_prod)

            vg = simulate_model(model, components, kinetic_comps, vkg, p_growth, f_growth, fix_flx, fix_comp)
            vp = simulate_model(model, components, kinetic_comps, vkp, p_prod, f_prod, fix_flx, fix_comp)

            f = phase_progress(c0, components, classification_data)
            v = (1 - f) * vg + f * vp
            dc = eval_reactor(c0, v, c_in)
            c1 = c0 + dc * h

            # Double step
            c1a = c0 + dc * (h / 2)
            kc = np.array([c1a[components.index(comp)] for comp in kinetic_comps])
            vkg = evaluate_fluxes(kc, kinetic_growth)
            vkp = evaluate_fluxes(kc, kinetic_prod)

            vg = simulate_model(model, components, kinetic_comps, vkg, p_growth, f_growth, fix_flx, fix_comp)
            vp = simulate_model(model, components, kinetic_comps, vkp, p_prod, f_prod, fix_flx, fix_comp)

            f = phase_progress(c0, components, classification_data)
            v = (1 - f) * vg + f * vp
            dc = eval_reactor(c1a, v, c_in)
            c2 = c1a + dc * (h / 2)

            # Check error
            rel_err = np.sqrt(np.mean(((c2 - c1) / rtol) ** 2))

            if rel_err < 1:
                ctr += 1
                c0 = c2
                profiles[ctr, :] = c0
                phase_transition[ctr] = f
                flux_growth[ctr, :] = vg
                flux_prod[ctr, :] = vp
                t_curr += h
                time[ctr] = t_curr
                h = h / max(np.sqrt(rel_err), 0.5)
                h = min(h, time_range[1] - t_curr)
                progress.update(h)
            else:
                h /= 1.5

            if abs(time_range[1] - t_curr) < 1e-7:
                done = True

    return {
        "time": time[:ctr + 1],
        "profiles": profiles[:ctr + 1, :],
        "flux_growth": flux_growth[:ctr + 1, :],
        "flux_prod": flux_prod[:ctr + 1, :],
        "phase_transition": phase_transition[:ctr + 1],
        "notes": notes,
        "condition": dfba_data['Condition']
    }
