struct PIPG_OPTS
    omega::Float64
    rho::Float64
    check_iter::Int64
    max_iter::Int64

    function PIPG_OPTS()
        omega = 1000.0
        rho = 1.7
        check_iter = 50
        max_iter = 100000
        new(omega, rho, check_iter, max_iter)
    end
end
struct PIPG_STRUCT
    P::Matrix{Float64}
    q::Vector{Float64}
    H::Matrix{Float64}
    h::Vector{Float64}
    opts::PIPG_OPTS
    function PIPG_STRUCT(P, q, H, h)
        new(P, q, Matrix(H), h, PIPG_OPTS())
    end
end