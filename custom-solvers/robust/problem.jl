using Distributions
struct PROBLEM
    pb::Vector{Float64}
    S_sr::Matrix{Float64}
    a::Float64
    phi_inv::Float64
    n::Int64

    function PROBLEM(pb, S, a, b)
        @assert 0 <= b <= 0.5
        phi_inv = quantile(Normal(), b)
        F = cholesky(S)
        S_sr = Matrix(F.U)
        new(pb, S_sr, a, phi_inv, length(pb))
    end
end