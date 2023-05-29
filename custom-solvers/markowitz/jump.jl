using JuMP, ECOS, SCS
include("problem.jl")

function jump_solve(pb, S)
    model = Model(ECOS.Optimizer)
    set_silent(model)
    @variable(model, x[1:p.n])
    @objective(model, Min, 0.5 * x' * S * x -sum(pb .* x))
    @constraint(model, sum(x) == 1)
    @constraint(model, x .>= 0)
    optimize!(model)
    @assert termination_status(model) == OPTIMAL "Infeasible Problem"

    return value.(x), objective_value(model), solve_time(model)
end