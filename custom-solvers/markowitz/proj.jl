function proj_Kp(w)
    # Projection onto zero cone
    return [0.0]
end
function proj_D!(x)
    max.(x, 0)
end