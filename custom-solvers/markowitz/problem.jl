using Distributions
struct PROBLEM
    pb::Vector{Float64}
    S::Matrix{Float64}
    n::Int64

    function PROBLEM(pb, S)
        new(pb, S, length(pb))
    end
end