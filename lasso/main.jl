using PyPlot, Random, IterativeSolvers

include("utils.jl")
include("algorithms.jl")

begin
    # Number of observations
    m = 200

    # Number of features
    n = 1000
    x_true = zeros(n)

    # True number of nonzeros in solution vector
    nnz = n / 2
    rng = MersenneTwister(1234);

    for i = 1 : length(nnz)
        x_true[rand(rng, 1:n)] = rand(rng, 1:100)
    end

    # Generate problem data
    A = rand(rng, m, n)
    b = A * x_true + rand(rng, m)

    # Lasso regularization parameter
    lambda = 10

    # ADMM Stepsize
    rho = 100

    # Number of ADMM iterations for high accuracy solution
    N_high_acc = 10000
    x_opt, obj_opt, admm_iter = admm(A, b, lambda, rho, N_high_acc, 0)

    # Max iterations for ISTA/FISTA
    N = 100000
    t_ista = @elapsed x_ista, obj_ista, ista_iter, ista_total_iters = ista(A, b, lambda, N, obj_opt)
    t_fista = @elapsed x_fista, obj_fista, fista_iter, fista_total_iters = fista(A, b, lambda, N, obj_opt)

    # Sample frequency for ISTA/FISTA
    sample_freq = 10
    n_fista_samples = Int64(floor(fista_total_iters / sample_freq))
    fista_res = zeros(n_fista_samples)

    # Get residuals for FISTA
    for i = 1 : n_fista_samples
        fista_res[i] = abs(obj_opt - obj(fista_iter[:,sample_freq * i]))
    end

    n_ista_samples = Int64(floor(ista_total_iters / sample_freq))
    ista_res = zeros(n_ista_samples)
    for i = 1 : n_ista_samples
        ista_res[i] = abs(obj_opt - obj(ista_iter[:,sample_freq * i]))
    end

    # Run ADMM with different stepsizes
    rho = [10,50,100]
    n_rho = size(rho)[1]
    N_admm = 2500
    x_admm = zeros(n_rho, n)
    obj_admm = zeros(n_rho)
    admm_iter = zeros(n_rho, n, N_admm)
    admm_total_iters = zeros(n_rho)
    t_admm = zeros(n_rho)
    for i = 1 : n_rho
        t_admm[i] = @elapsed x_admm[i,:], obj_admm[i], admm_iter[i,:,:], admm_total_iters[i] = admm(A, b, lambda, rho[i], N_admm, obj_opt)
    end
    
    admm_res = zeros(n_rho, N_admm)
    # Get residuals for ADMM
    for n = 1 : n_rho
        for i = 1 : N_admm
            admm_res[n,i] = abs(obj_opt - obj(admm_iter[n,:,i]))
        end
    end

    # Run OSQP
    x_osqp, obj_osqp, t_osqp = jump(A, b, lambda)
    println("ADMM Time: ", t_admm)
    println("OSQP Time: ", t_osqp)
    println("FISTA Time: ", t_fista)
    println("ISTA Time: ", t_ista)

    println("OSQP Distance to Optimal: ", abs(obj_opt - obj_osqp))
    println("High Acc Residual: ", get_residual(A, b, lambda, x_opt))

    ista_iters_array = LinRange(0, ista_total_iters, n_ista_samples)
    fista_iters_array = LinRange(0, fista_total_iters, n_fista_samples)

    figure(dpi=200)
    plot(ista_iters_array, ista_res, label="ISTA")
    plot(fista_iters_array, fista_res, label="FISTA")
    legend()
    yscale("log")
    title("ISTA/FISTA Distance to Optimal")
    xlabel("Iteration")
    ylabel(L"$f(x)-f^*(x)$")
    grid(true)

    figure(dpi=200)
    for i = 1 : n_rho
        plot(admm_res[i,1:Int64(admm_total_iters[i])], label=L"$\rho=$"*string(rho[i]))
    end
    legend()
    yscale("log")
    title("ADMM Distance to Optimal")
    xlabel("Iteration")
    ylabel(L"$f(x)-f^*(x)$")
    grid(true)

    show()
end