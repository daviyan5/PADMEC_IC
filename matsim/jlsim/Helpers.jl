
module Helpers

include("MeshGenerator.jl")

import .MeshGenerator
import LinearAlgebra as LA

using StaticArrays

# ----------- Print Helpers --------------------------------#
mutable struct IdxIter
    current :: Int64
    function IdxIter(current :: Int64)
        this = new()
        this.current = current
        return this
    end
    
end
global idx_iter = IdxIter(0)

function reset_verbose()
    idx_iter.current = 0
end


function verbose(msg :: String, type :: String, verbose :: Bool = true)
    if verbose == false
        return
    end
    lim = 100
    suf1 = " " * repeat("=", lim)
    suf2 = " " * repeat("+", lim)
    pref1 = "-- "
    if type == "OUT"
        println()
        msg *= suf1
    elseif type == "CHK"
        msg *= suf2
    elseif type == "INFO"
        msg = pref1 * msg
    end
    msg = (length(msg) > lim) ? msg[1:lim] : msg
    idx = lpad(idx_iter.current, 3, "0")
    idx_iter.current += 1
    print("[$(idx)]\t$(rpad(msg, lim))")
    println("\t$(rpad(type, 3))")

end

# ----------------------------------------------------------#

function get_random_tensor(a :: Number, b :: Number, sz :: Int64, n :: Int64 = 3, m :: Int64 = 3, only_diagonal :: Bool = false) :: Vector
    """
    Retorna um array de tamanho size cujos elementos são tensores aleatórios n x m, com valores entre a e b
    Se only_diagonal for True, os elementos fora da diagonal serão iguais a zero, e o tensor será n x n

    """

    if only_diagonal == true
        m = n
    end
    shape = tuple(n, m, sz)
    tensor = (b - a) .* rand(Float64, shape) .+ a

    if only_diagonal == true
        A = LA.Diagonal(ones(n, m))
        tensor = A .* tensor
    end

    tensor = Vector([SMatrix{n, m}(tensor[:, :, i]) for i in 1:sz])
    return tensor
end

function get_tensor(a :: Number, sz :: Int64, n :: Int64 = 3, m :: Int64 = 3, only_diagonal :: Bool = false) :: Vector
    return get_random_tensor(a, a, sz, n, m, only_diagonal)
end

function get_column(v :: SVector, i :: Int64)
    return v[i]
end


function get_column(v :: Vector, i :: Int64)
    return v[i]
end

function abs_b(v :: SVector) :: SVector
    return abs.(v)
end

function abs_b(v :: Vector) :: Vector
    return abs.(v)
end
end # module Helpers


import .Helpers

function testing()
    Helpers.verbose("Testing", "OUT", true)
    Helpers.verbose("Testing", "CHK", true)
    Helpers.verbose("Testing", "INFO", true)
    name = "mesh"
    Helpers.verbose("== Applying TPFA Scheme over $(name)...", "OUT", true)

    a = Helpers.get_tensor(1, 10, 3, 3, false)
    b = Helpers.get_random_tensor(10, 11, 11,  3, 3, true)
    show(a[1])
    println()
    show(b[1])
    println()
    
end

#testing()