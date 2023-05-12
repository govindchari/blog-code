using Distributions
struct PROBLEM
    pb::Vector{Float64}
    S::Matrix{Float64}
    a::Float64
    phi_inv::Float64

    function PROBLEM(pb, S, a, b)
        @assert 0 <= b <= 0.5
        phi_inv = quantile(Normal(), b)
        new(pb, S, a, phi_inv)
    end
end