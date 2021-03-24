## Compatibility layer to work with Julia 1.0

# https://github.com/JuliaLang/julia/pull/29679
@static if VERSION < v"1.1.0-DEV.472"
    isnothing(::Any) = false
    isnothing(::Nothing) = true
    export isnothing
end

# https://github.com/JuliaLang/julia/pull/32148
@static if VERSION < v"1.3.0"
    import Base: print
    print(io::IO, ::Nothing) = print(io, "nothing")
end

# https://github.com/JuliaLang/julia/pull/29173
@static if VERSION < v"1.1.0"
    import Future: copy!
end
