using PyCall


function py()
    py"""
    import numpy as np

    def f():
        a = np.random.randint(0, 10, 100000)
        p = np.random.uniform(0, 1, 100000)
        v = np.zeros(10)
        np.add.at(v, a, p)
        return v
    """
    return py"f"()

end

function jl()
    a = rand(1:10, 100000)
    p = rand(100000)
    v = zeros(10)
    for i in 1:100000
        v[a[i]] += p[i]
    end
    return v
end
