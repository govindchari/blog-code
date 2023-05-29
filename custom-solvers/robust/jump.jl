using JuMP, ECOS, SCS
include("problem.jl")

function jump_solve(p::PROBLEM)
    model = Model(ECOS.Optimizer)
    set_silent(model)
    @variable(model, x[1:p.n])
    f = 0
    @objective(model, Min, -sum(p.pb .* x))
    @constraint(model, sum(x) == 1)
    @constraint(model, x .>= 0)
    @constraint(model, [sum(p.pb .* x) - p.a; p.phi_inv * p.S_sr * x] in SecondOrderCone())
    optimize!(model)
    @assert termination_status(model) == OPTIMAL "Infeasible Problem"

    return value.(x), objective_value(model), solve_time(model)
end