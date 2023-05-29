using LinearAlgebra, BenchmarkTools

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

p = PROBLEM(pb, S)

P = S
q = -p.pb
H = 1.0 * ones(1, n)
h = [1.0]

pp = PIPG_STRUCT(P, q, H, h)

xecos, obj, tecos = jump_solve(pb, S)
xpipg, tpipg = solve_pipg(pp, xecos, true)

println("PIPG Time: ", tpipg * 1000)
println("ECOS Time: ", tecos * 1000)