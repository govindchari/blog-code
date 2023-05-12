using LinearAlgebra

include("problem.jl")

pb = [1.0;0.0]
a = 0.5
b = 0.5
S = rand(2,2)
S = S' * S

p = PROBLEM(pb, S, a, b)
println(p.phi_inv)