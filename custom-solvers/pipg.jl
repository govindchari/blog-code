using Printf, IterativeSolvers

include("pipg_structs.jl")
function solve_pipg(p::PIPG_STRUCT, sol::Vector{Float64}, verbose::Bool)
    t = @elapsed begin

        opts = p.opts
        stop = false
        k = 1
        HtH = p.H' * p.H

        @time begin
            norm2H = real(powm(HtH)[1])
            normP = real(powm(P)[1])
        end

        a = 2 / (sqrt(normP^2 + 4.0 * opts.omega * norm2H) + normP)
        b = opts.omega * a

        z = zeros(length(p.q))
        xi = zeros(length(p.q))
        w = zeros(length(p.h))
        eta = zeros(length(p.h))

        if (verbose)
            println("iter     objv       sum(x) - 1\n")
            println("-----------------------------\n")
        end

        while (stop == false)
            z = proj_D!(xi - a * (p.P * xi + p.q + p.H' * eta))
            w = eta + b * (p.H * (2 * z - xi) - p.h)
            w = w - proj_Kp(w)
            xi = (1 - opts.rho) * xi + opts.rho * z
            eta = (1 - opts.rho) * eta + opts.rho * w
            if (verbose)
                if mod(k, opts.check_iter) == 0
                    @printf("%3d   %10.3e  %9.2e\n",
                        k, dot(p.q, xi), sum(xi) - 1.0)
                    stop = (norm(xi - sol) / norm(sol)) <= 0.001 || k >= opts.max_iter
                end
            end
            k += 1
        end
    end
    return xi, t
end