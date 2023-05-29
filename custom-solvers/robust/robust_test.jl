using LinearAlgebra

include("problem.jl")
include("jump.jl")
include("proj.jl")
include("../pipg.jl")

n = 100
m = 30
F = rand(n, m)
D = Diagonal(rand(n))
S = F * F' + D
pb = rand(n)

a = sum(pb) / n - n
b = 0.1


p = PROBLEM(pb, S, a, b)

P = zeros(n, n)
q = -p.pb
H = [1.0 * ones(n)'; p.phi_inv * p.S_sr; p.pb']
h = [1.0; zeros(n); p.a]
for i = 1:size(H)[1]
    nrm = norm(H[i, :])
    H[i, :] .= H[i, :] ./ nrm
    h[i] /= nrm
end

pp = PIPG_STRUCT(P, q, H, h)

xecos, obj, tecos = jump_solve(p)
xpipg, tpipg = solve_pipg(pp, xecos, true)

println("PIPG Time: ", tpipg * 1000)
println("ECOS Time: ", tecos * 1000)