using IterativeSolvers, JuMP, OSQP, LinearAlgebra

include("utils.jl")

function admm(A, b, lambda, rho, N, obj_opt)
    check_iters = 25
    m, n = size(A)
    At = A'
    AtA = At * A
    Atb = At * b
    F = cholesky(AtA + rho * I(n))
    threshold_parameter = lambda / rho
    x = ones(n)
    z = ones(n)
    zprev = ones(n)
    y = ones(n)
    tau = 2
    mu = 10
    iter = zeros(n, N)
    i = 1
    while(i < N)
        x .= F \ (Atb + rho * z - y)
        z .= soft_threshold(x + y / rho, threshold_parameter)
        y .= y + rho * (x - z)

        # Stepsize update
        # r = norm(x - z)
        # s = norm(z - zprev)
        # if (1 == 1)
        #     if (r > mu * s)
        #         rho *= tau
        #         println("Rho: ", rho)
        #         F = cholesky(AtA + rho * I(n))
        #     elseif (s > mu * r)
        #         rho /= tau
        #         F = cholesky(AtA + rho * I(n))
        #         println("Rho: ", rho)
        #     end    
        # end

        iter[:,i] .= x
        zprev = copy(z)
        if (i % check_iters == 0)
            obj = 0.5 * (A * x - b)' * (A * x - b) + lambda * norm(x, 1)
            if (abs(obj - obj_opt) < 1e-5)
                break
            end
        end
        i += 1
    end
    return x, (0.5 * (A * x - b)' * (A * x - b) + lambda * norm(x, 1)), iter, i-1
end

function ista(A, b, lambda, N, obj_opt)
    check_iters = 25
    m, n = size(A)
    At = A'
    AtA = At * A
    Atb = At * b
    L = real(powm(A' * A)[1])
    step_size = 1 / L
    threshold_parameter = lambda / L
    iter = zeros(n,N)
    x = ones(n)
    i = 1
    while(i <= N)
        x = soft_threshold(x .- (step_size) .* (AtA * x .- Atb), threshold_parameter)
        iter[:,i] = x
        if (i % check_iters == 0)
            obj = 0.5 * (A * x - b)' * (A * x - b) + lambda * norm(x, 1)
            if (abs(obj - obj_opt) < 1e-5)
                break
            end
        end
        i += 1
    end
    return x, (0.5 * (A * x - b)' * (A * x - b) + lambda * norm(x, 1)), iter, i
end

function fista(A, b, lambda, N, obj_opt)
    check_iters = 25
    m, n = size(A)
    At = A'
    AtA = At * A
    Atb = At * b
    L = real(powm(A' * A)[1])
    step_size = 1 / L
    threshold_parameter = lambda / L
    iter = zeros(n,N)
    y = ones(n)
    x = ones(n)
    xprev = ones(n)
    yprev = ones(n)
    tprev = 1
    i = 1
    while (i < N)
        x .= soft_threshold(yprev .- (step_size) .* (AtA * yprev .- Atb), threshold_parameter)
        t = (1 + sqrt(1 + 4 * tprev ^2)) / 2
        y .= x .+ ((tprev - 1) / t) .* (x .- xprev)
        iter[:,i] .= x
        xprev .= x
        tprev = t
        yprev .= y
        if (i % check_iters == 0)
            obj = 0.5 * (A * x - b)' * (A * x - b) + lambda * norm(x, 1)
            if (abs(obj - obj_opt) < 1e-5)
                break
            end
        end
        i += 1
    end
    return x, (0.5 * (A * x - b)' * (A * x - b) + lambda * norm(x, 1)), iter, i
end

function jump(A, b, lambda)
    m,n = size(A)

    model = Model(OSQP.Optimizer)
    set_optimizer_attribute(model, "eps_rel", 5e-10)
    set_optimizer_attribute(model, "eps_abs", 5e-10)

    @variable(model, x[1:n])
    @variable(model, t[1:1])
    @constraint(model, [t;x] in MOI.NormOneCone(n + 1))
    @constraint(model, t .>= 0)
    @objective(model, Min, 0.5 * (A * x - b)' * (A * x - b) + lambda * sum(t))
    optimize!(model)
    return value.(x), objective_value(model), solve_time(model)
end
