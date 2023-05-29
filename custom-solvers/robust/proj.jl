function proj_Kp(w)
    c = w[2:end-1]
    s = w[end]
    nc = norm(c)
    if (nc <= s)
    elseif (nc <= -s)
        c .= 0
        s = 0
    else
        f = 0.5 * (nc + s)
        s = f
        c = f * (c ./ nc)
    end
    return [0.0; c; s]
end
function proj_D!(x)
    max.(x, 0)
end