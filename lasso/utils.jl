function soft_threshold(x, lambda)
    return sign.(x) .* max.(abs.(x) .- lambda, 0)
end

function obj(x)
    res = A * x - b
    return 0.5 * dot(res, res) + lambda * norm(x, 1)
end

function get_residual(A, b, lambda, x)
    m,n = size(A)
    res = zeros(n)
    fac = b - A * x
    tol = 1e-8
    for i = 1 : n
        daf = dot(A[:,i], fac)
        if (abs(x[i]) >= tol)
            res[i] = daf - lambda * sign(x[i])
        else
            res[i] = max((abs(daf) - lambda), 0)
        end
    end
    return norm(res)
end